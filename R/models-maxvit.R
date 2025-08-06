# R/model-maxvit.R - PyTorch compatible MaxViT implementation

conv_norm_act <- function(in_channels, mid_channels = 64, out_channels = 64, ...) {
  nn_sequential(
    "stem.0" = nn_sequential(
      "0" = nn_conv2d(in_channels, mid_channels, ..., bias = FALSE),
      "1" = nn_batch_norm2d(mid_channels, track_running_stats = TRUE),
      "2" = nn_gelu()
    ),
    "stem.1" = nn_sequential(
      "0" = nn_conv2d(mid_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = TRUE)
    )
  )
}

squeeze_excitation <- nn_module(
  initialize = function(in_channels, squeeze_factor = 16) {
    squeeze_channels <- max(1L, as.integer(in_channels / squeeze_factor))
    self$fc1 <- nn_conv2d(in_channels, squeeze_channels, 1)
    self$fc2 <- nn_conv2d(squeeze_channels, in_channels, 1)
  },
  forward = function(x) {
    scale <- x$mean(dim = c(3, 4), keepdim = TRUE)
    scale <- self$fc1(scale)
    scale <- torch_gelu(scale)
    scale <- self$fc2(scale)
    scale <- torch_sigmoid(scale)
    x * scale
  }
)

mbconv <- nn_module(
  initialize = function(in_channels, out_channels, expansion = 4, stride = 1) {
    hidden_dim <- in_channels * expansion
    self$use_expansion <- expansion != 1

    layers_list <- list()
    if (self$use_expansion) {
      layers_list[["conv_a"]] <- nn_sequential(
        "0" = nn_conv2d(in_channels, hidden_dim, 1, bias = FALSE),
        "1" = nn_batch_norm2d(hidden_dim, track_running_stats = TRUE)
      )
    }

    layers_list[["conv_b"]] <- nn_sequential(
      "0" = nn_conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = FALSE),
      "1" = nn_batch_norm2d(hidden_dim, track_running_stats = TRUE)
    )

    layers_list[["squeeze_excitation"]] <- squeeze_excitation(hidden_dim)
    layers_list[["pre_norm"]] <- nn_batch_norm2d(out_channels, track_running_stats = TRUE)
    self$layers <- nn_module_dict(layers_list)

    proj_dict <- list()
    proj_dict[["1"]] <- nn_conv2d(hidden_dim, out_channels, 1)
    self$proj <- nn_module_dict(proj_dict)

    self$use_res_connect <- stride == 1 && in_channels == out_channels
  },

  forward = function(x) {
    identity <- x
    out <- x

    if (self$use_expansion) {
      out <- self$layers[["conv_a"]](out)
      out <- torch_gelu(out)
    }

    out <- self$layers[["conv_b"]](out)
    out <- torch_gelu(out)
    out <- self$layers[["squeeze_excitation"]](out)
    out <- self$proj[["1"]](out)
    out <- self$layers[["pre_norm"]](out)

    if (self$use_res_connect) {
      out <- out + identity
    }

    out
  }
)

window_attention <- nn_module(
  initialize = function(dim, heads = 4) {
    self$attn_layer <- nn_sequential(
      "0" = nn_layer_norm(dim),
      "1" = nn_multihead_attention(embed_dim = dim, num_heads = heads, batch_first = TRUE)
    )
    self$mlp_layer <- nn_sequential(
      "0" = nn_layer_norm(dim),
      "1" = nn_linear(dim, dim * 4),
      "2" = nn_gelu(),
      "3" = nn_linear(dim * 4, dim)
    )
  },

  forward = function(x) {
    b <- x$size(1); c <- x$size(2); h <- x$size(3); w <- x$size(4)
    x <- x$permute(c(1, 3, 4, 2))$reshape(c(b, h * w, c))
    x <- x + self$attn_layer[[2]](self$attn_layer[[1]](x), x, x)[[1]]
    x <- x + self$mlp_layer(x)
    x <- x$reshape(c(b, h, w, c))$permute(c(1, 4, 2, 3))
    x
  }
)

grid_attention <- nn_module(
  initialize = function(dim, heads = 4) {
    self$attn_layer <- nn_sequential(
      "0" = nn_layer_norm(dim),
      "1" = nn_multihead_attention(embed_dim = dim, num_heads = heads, batch_first = TRUE)
    )
    self$mlp_layer <- nn_sequential(
      "0" = nn_layer_norm(dim),
      "1" = nn_linear(dim, dim * 4),
      "2" = nn_gelu(),
      "3" = nn_linear(dim * 4, dim)
    )
  },

  forward = function(x) {
    b <- x$size(1); c <- x$size(2); h <- x$size(3); w <- x$size(4)
    gh <- h %/% 2; gw <- w %/% 2
    x <- x$reshape(c(b, c, gh, 2, gw, 2))$permute(c(1, 3, 5, 4, 6, 2))$reshape(c(b * gh * gw, 4, c))
    x <- x + self$attn_layer[[2]](self$attn_layer[[1]](x), x, x)[[1]]
    x <- x + self$mlp_layer(x)
    x <- x$reshape(c(b, gh, gw, 2, 2, c))$permute(c(1, 6, 2, 4, 3, 5))$reshape(c(b, c, h, w))
    x
  }
)

maxvit_block <- nn_module(
  initialize = function(in_channels, out_channels, expansion = 4, stride = 1) {
    self$layers <- nn_module_dict(list(
      MBconv = mbconv(in_channels, out_channels, expansion, stride),
      window_attention = window_attention(out_channels),
      grid_attention = grid_attention(out_channels)
    ))
  },

  forward = function(x) {
    x <- self$layers$MBconv(x)
    x <- self$layers$window_attention(x)
    x <- self$layers$grid_attention(x)
    x
  }
)

maxvit_impl <- nn_module(
  initialize = function(num_classes = 1000) {
    self$stem <- conv_norm_act(3, mid_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
    self$blocks <- nn_sequential(
      "0" = maxvit_block(64, 64, expansion = 4, stride = 2),
      "1" = maxvit_block(64, 64, expansion = 4, stride = 1),
      "2" = maxvit_block(64, 64, expansion = 8, stride = 2),
      "3" = maxvit_block(64, 128, expansion = 8, stride = 1),
      "4" = maxvit_block(128, 128, expansion = 8, stride = 2),
      "5" = maxvit_block(128, 256, expansion = 8, stride = 1),
      "6" = maxvit_block(256, 256, expansion = 4, stride = 1),
      "7" = maxvit_block(256, 256, expansion = 4, stride = 1),
      "8" = maxvit_block(256, 256, expansion = 8, stride = 2),
      "9" = maxvit_block(256, 512, expansion = 8, stride = 1)
    )
    self$pool <- nn_adaptive_avg_pool2d(c(1, 1))
    self$fc <- nn_linear(512, num_classes)
  },

  forward = function(x) {
    x <- self$stem(x)
    x <- self$blocks(x)
    x <- self$pool(x)
    x <- x$flatten(start_dim = 2)
    self$fc(x)
  }
)

model_maxvit <- function(pretrained = FALSE, progress = TRUE, num_classes = 1000, ...) {
  model <- maxvit_impl(num_classes = num_classes)

  if (pretrained) {
    path <- download_and_cache("https://torch-cdn.mlverse.org/models/vision/v2/models/maxvit.pth")
    state_dict <- torch::load_state_dict(path)

    # Optionally filter num_batches_tracked if needed
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]

    # Actually load the weights
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

