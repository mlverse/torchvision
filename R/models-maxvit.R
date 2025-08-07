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

    self$layers <- nn_module_dict(list(
      pre_norm = nn_batch_norm2d(in_channels, track_running_stats = TRUE),
      conv_a = nn_sequential(
        "0" = nn_conv2d(in_channels, hidden_dim, 1, bias = FALSE),
        "1" = nn_batch_norm2d(hidden_dim, track_running_stats = TRUE),
        "2" = nn_gelu()
      ),
      conv_b = nn_sequential(
        "0" = nn_conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = FALSE),
        "1" = nn_batch_norm2d(hidden_dim, track_running_stats = TRUE),
        "2" = nn_gelu()
      ),
      squeeze_excitation = squeeze_excitation(hidden_dim),
      conv_c = nn_conv2d(hidden_dim, out_channels, 1, bias = TRUE)
    ))

    self$proj <- nn_sequential(
      "0" = nn_identity(),
      "1" = nn_batch_norm2d(out_channels, track_running_stats = FALSE)
    )

    self$use_res_connect <- stride == 1 && in_channels == out_channels
  },

  forward = function(x) {
    identity <- x
    out <- self$layers$pre_norm(x)
    out <- self$layers$conv_a(out)
    out <- self$layers$conv_b(out)
    out <- self$layers$squeeze_excitation(out)
    out <- self$layers$conv_c(out)
    out <- self$proj(out)

    if (self$use_res_connect) {
      out <- out + identity
    }

    out
  }
)

window_attention <- nn_module(
  initialize = function(dim, heads = 4) {
    self$attn <- nn_multihead_attention(embed_dim = dim, num_heads = heads, batch_first = TRUE)
    self$norm1 <- nn_layer_norm(dim)
    self$mlp <- nn_sequential(
      nn_linear(dim, dim * 4),
      nn_gelu(),
      nn_linear(dim * 4, dim)
    )
    self$norm2 <- nn_layer_norm(dim)
  },

  forward = function(x) {
    b <- x$size(1); c <- x$size(2); h <- x$size(3); w <- x$size(4)
    x <- x$permute(c(1, 3, 4, 2))$reshape(c(b, h * w, c))
    x <- x + self$attn(self$norm1(x), self$norm1(x), self$norm1(x))[[1]]
    x <- x + self$mlp(self$norm2(x))
    x <- x$reshape(c(b, h, w, c))$permute(c(1, 4, 2, 3))
    x
  }
)

grid_attention <- nn_module(
  initialize = function(dim, heads = 4) {
    self$attn <- nn_multihead_attention(embed_dim = dim, num_heads = heads, batch_first = TRUE)
    self$norm1 <- nn_layer_norm(dim)
    self$mlp <- nn_sequential(
      nn_linear(dim, dim * 4),
      nn_gelu(),
      nn_linear(dim * 4, dim)
    )
    self$norm2 <- nn_layer_norm(dim)
  },

  forward = function(x) {
    b <- x$size(1); c <- x$size(2); h <- x$size(3); w <- x$size(4)
    gh <- h %/% 2; gw <- w %/% 2
    x <- x$reshape(c(b, c, gh, 2, gw, 2))$permute(c(1, 3, 5, 4, 6, 2))$reshape(c(b * gh * gw, 4, c))
    x <- x + self$attn(self$norm1(x), self$norm1(x), self$norm1(x))[[1]]
    x <- x + self$mlp(self$norm2(x))
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

maxvit_stage <- nn_module(
  initialize = function(...) {
    self$layers <- nn_sequential(...)
  },
  forward = function(x) {
    self$layers(x)
  }
)

maxvit_impl <- nn_module(
  initialize = function(num_classes = 1000) {
    self$stem <- conv_norm_act(
      3, mid_channels = 64, out_channels = 64,
      kernel_size = 3, stride = 2, padding = 1
    )

    self$blocks <- nn_module_list(list(
      # stage 0
      maxvit_stage(
        "0" = maxvit_block(64, 64, expansion = 4, stride = 2),
        "1" = maxvit_block(64, 64, expansion = 4, stride = 1)
      ),
      # stage 1
      maxvit_stage(
        "0" = maxvit_block(64, 128, expansion = 8, stride = 2),
        "1" = maxvit_block(128, 128, expansion = 4, stride = 1)
      ),
      # stage 2
      maxvit_stage(
        "0" = maxvit_block(128, 256, expansion = 8, stride = 2),
        "1" = maxvit_block(256, 256, expansion = 4, stride = 1),
        "2" = maxvit_block(256, 256, expansion = 4, stride = 1),
        "3" = maxvit_block(256, 256, expansion = 4, stride = 1),
        "4" = maxvit_block(256, 256, expansion = 4, stride = 1)
      ),
      # stage 3
      maxvit_stage(
        "0" = maxvit_block(256, 512, expansion = 8, stride = 2),
        "1" = maxvit_block(512, 512, expansion = 4, stride = 1)
      )
    ))

    self$pool <- nn_adaptive_avg_pool2d(c(1, 1))
    self$fc <- nn_linear(512, num_classes)
  },

  forward = function(x) {
    x <- self$stem(x)
    for (stage in self$blocks) {
      x <- stage(x)
    }
    x <- self$pool(x)
    x <- x$flatten(start_dim = 2)
    self$fc(x)
  }
)

.rename_maxvit_state_dict <- function(state_dict) {
  renamed <- list()

  for (nm in names(state_dict)) {
    # Skip relative position bias
    if (grepl("relative_position", nm)) next

    new_nm <- nm

    # Attention + MLP layer renaming
    new_nm <- sub("attn_layer\\.0\\.", "norm1.", new_nm)
    new_nm <- sub("attn_layer\\.1\\.to_qkv\\.", "attn.in_proj_", new_nm)
    new_nm <- sub("attn_layer\\.1\\.merge\\.", "attn.out_proj.", new_nm)
    new_nm <- sub("mlp_layer\\.0\\.", "norm2.", new_nm)
    new_nm <- sub("mlp_layer\\.1\\.", "mlp.0.", new_nm)
    new_nm <- sub("mlp_layer\\.3\\.", "mlp.2.", new_nm)

    # Classifier replacements
    if (startsWith(new_nm, "classifier.2.")) {
      new_nm <- sub("^classifier\\.2\\.", "fc.", new_nm)
    }

    if (startsWith(new_nm, "classifier.3.") || startsWith(new_nm, "classifier.5.")) {
      next  # Skip these
    }

    renamed[[new_nm]] <- state_dict[[nm]]
  }

  renamed
}

model_maxvit <- function(pretrained = FALSE, progress = TRUE, num_classes = 1000, ...) {
  model <- maxvit_impl(num_classes = num_classes)

  if (pretrained) {
    path <- download_and_cache("https://torch-cdn.mlverse.org/models/vision/v2/models/maxvit.pth")
    state_dict <- torch::load_state_dict(path)
    state_dict <- .rename_maxvit_state_dict(state_dict)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]

    model_state <- model$state_dict()
    model_state <- model_state[!grepl("num_batches_tracked$", names(model_state))]

    # Compare shapes before loading
    compare_state_dict_shapes(model_state, state_dict)

    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}
