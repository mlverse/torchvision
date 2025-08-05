# R/model-maxvit.R

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

se_block <- nn_module(
  initialize = function(in_channels, squeeze_factor = 16) {
    squeeze_channels <- max(1L, as.integer(in_channels / squeeze_factor))
    self$fc1 <- nn_conv2d(in_channels, squeeze_channels, 1)
    self$act1 <- nn_gelu()
    self$fc2 <- nn_conv2d(squeeze_channels, in_channels, 1)
    self$act2 <- nn_sigmoid()
  },
  forward = function(x) {
    scale <- x$mean(dim = c(3, 4), keepdim = TRUE)
    scale <- self$fc1(scale) %>% self$act1() %>% self$fc2() %>% self$act2()
    x * scale
  }
)

mbconv_block <- nn_module(
  initialize = function(in_channels, out_channels, expansion = 4, stride = 1) {
    hidden_dim <- in_channels * expansion
    self$expand <- if (expansion != 1) nn_conv2d(in_channels, hidden_dim, 1, bias = FALSE) else NULL
    self$bn1 <- nn_batch_norm2d(hidden_dim)
    self$act1 <- nn_gelu()
    self$dwconv <- nn_conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = FALSE)
    self$bn2 <- nn_batch_norm2d(hidden_dim)
    self$act2 <- nn_gelu()
    self$se <- se_block(hidden_dim)
    self$project <- nn_conv2d(hidden_dim, out_channels, 1)
    self$bn3 <- nn_batch_norm2d(out_channels)
    self$use_res_connect <- stride == 1 && in_channels == out_channels
  },
  forward = function(x) {
    identity <- x
    out <- x
    if (!is.null(self$expand)) {
      out <- out %>% self$expand() %>% self$bn1() %>% self$act1()
    }
    out <- out %>% self$dwconv() %>% self$bn2() %>% self$act2()
    out <- self$se(out)
    out <- out %>% self$project() %>% self$bn3()
    if (self$use_res_connect) out <- out + identity
    out
  }
)

maxvit_block <- nn_module(
  initialize = function(dim, heads = 4) {
    self$norm1 <- nn_layer_norm(dim)
    self$attn_block <- nn_multihead_attention(embed_dim = dim, num_heads = heads, batch_first = TRUE)
    self$norm2 <- nn_layer_norm(dim)
    self$mlp <- nn_sequential(
      nn_linear(dim, dim * 4),
      nn_gelu(),
      nn_linear(dim * 4, dim)
    )
  },
  forward = function(x) {
    b <- x$size(1)
    c <- x$size(2)
    h <- x$size(3)
    w <- x$size(4)
    x <- x$permute(c(1, 3, 4, 2))$reshape(c(b, h * w, c))
    x <- self$norm1(x)
    attn_result <- self$attn_block(x, x, x)
    x <- x + attn_result[[1]]
    x <- x + self$mlp(self$norm2(x))
    x <- x$reshape(c(b, h, w, c))$permute(c(1, 4, 2, 3))
    x
  }
)

grid_attention <- nn_module(
  initialize = function(dim, heads = 4) {
    self$norm1 <- nn_layer_norm(dim)
    self$attn <- nn_multihead_attention(embed_dim = dim, num_heads = heads, batch_first = TRUE)
    self$norm2 <- nn_layer_norm(dim)
    self$mlp <- nn_sequential(
      nn_linear(dim, dim * 4),
      nn_gelu(),
      nn_linear(dim * 4, dim)
    )
  },
  forward = function(x) {
    b <- x$size(1)
    c <- x$size(2)
    h <- x$size(3)
    w <- x$size(4)

    gh <- h %/% 2
    gw <- w %/% 2
    x <- x$reshape(c(b, c, gh, 2, gw, 2))$
      permute(c(1, 3, 5, 4, 6, 2))$
      reshape(c(b * gh * gw, 4, c))

    x <- self$norm1(x)
    attn_result <- self$attn(x, x, x)
    x <- x + attn_result[[1]]
    x <- x + self$mlp(self$norm2(x))
    x <- x$reshape(c(b, gh, gw, 2, 2, c))$
      permute(c(1, 6, 2, 4, 3, 5))$
      reshape(c(b, c, h, w))
    x
  }
)

maxvit_stage <- nn_module(
  initialize = function(in_channels, out_channels, depth, expansions = rep(4, depth)) {
    self$blocks <- nn_module_list()
    self$blocks$append(mbconv_block(in_channels, out_channels, expansion = expansions[1], stride = 2))
    if (depth > 1) {
      for (i in 2:depth) {
        self$blocks$append(mbconv_block(out_channels, out_channels, expansion = expansions[i]))
      }
    }
    self$attn_blocks <- nn_module_list()
    for (i in 1:depth) {
      self$attn_blocks$append(maxvit_block(out_channels))
      self$attn_blocks$append(grid_attention(out_channels))
    }
  },
  forward = function(x) {
    for (block in self$blocks) {
      x <- block(x)
    }
    for (attn in self$attn_blocks) {
      x <- attn(x)
    }
    x
  }
)

maxvit_impl <- nn_module(
  initialize = function(num_classes = 1000) {
    self$stem <- conv_norm_act(3, mid_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)

    self$stages <- nn_sequential(
      maxvit_stage(64, 64, 2, expansions = c(4, 4)),
      maxvit_stage(64, 128, 2, expansions = c(8, 4)),
      maxvit_stage(128, 256, 4, expansions = c(8, 4, 4, 4)),
      maxvit_stage(256, 512, 2, expansions = c(8, 4))
    )

    self$pool <- nn_adaptive_avg_pool2d(c(1, 1))
    self$fc <- nn_linear(512, num_classes)
  },
  forward = function(x) {
    x <- self$stem(x)
    x <- self$stages(x)
    x <- self$pool(x)
    x <- x$flatten(start_dim = 2)
    self$fc(x)
  }
)

# Helper to adapt PyTorch weight names to the R implementation
.rename_maxvit_state_dict <- function(state_dict) {
  renamed <- list()

  for (nm in names(state_dict)) {
    # Skip irrelevant keys from Swin-like models
    if (grepl("window_attention|attn_layer|relative_position", nm)) next

    new_nm <- nm

    # === MBConv block mapping ===
    # blocks.X.layers.Y.layers.MBconv.layers.* → stages.X.blocks.Y.*
    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.MBconv\\.layers",
      "stages.\\1.blocks.\\2", new_nm
    )

    # blocks.X.layers.Y.layers.MBconv.* → stages.X.blocks.Y.*
    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.MBconv\\.",
      "stages.\\1.blocks.\\2.", new_nm
    )

    # === Pre-normalization ===
    new_nm <- sub("\\.pre_norm", ".bn3", new_nm)

    # === conv_a (expand) ===
    new_nm <- sub("\\.conv_a\\.0", ".expand", new_nm)
    new_nm <- sub("\\.conv_a\\.1", ".bn1", new_nm)

    # === conv_b (depthwise conv) ===
    new_nm <- sub("\\.conv_b\\.0", ".dwconv", new_nm)
    new_nm <- sub("\\.conv_b\\.1", ".bn2", new_nm)

    # === squeeze_excitation ===
    new_nm <- sub("\\.squeeze_excitation", ".se", new_nm)

    # === conv_c (projection) ===
    new_nm <- sub("\\.conv_c", ".project", new_nm)

    # === Fix incorrect proj.1 naming ===
    new_nm <- sub("\\.proj\\.1", ".project", new_nm)

    # === Attention blocks ===
    # Handle grid_attention -> attn_blocks
    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.grid_attention\\.mlp_layer\\.0",
      "stages.\\1.attn_blocks.\\2.norm1", new_nm
    )

    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.grid_attention\\.mlp_layer\\.1",
      "stages.\\1.attn_blocks.\\2.attn_block", new_nm
    )

    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.grid_attention\\.mlp_layer\\.3",
      "stages.\\1.attn_blocks.\\2.norm2", new_nm
    )

    # window_attention.attn_layer.1 → attn_block
    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.window_attention\\.attn_layer\\.1",
      "stages.\\1.attn_blocks.\\2.attn_block", new_nm
    )

    # Swin-style attention projection mappings
    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.window_attention\\.attn_layer\\.1\\.to_qkv\\.weight",
      "stages.\\1.attn_blocks.\\2.attn_block.in_proj_weight", new_nm
    )

    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.window_attention\\.attn_layer\\.1\\.to_qkv\\.bias",
      "stages.\\1.attn_blocks.\\2.attn_block.in_proj_bias", new_nm
    )

    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.window_attention\\.attn_layer\\.1\\.merge\\.weight",
      "stages.\\1.attn_blocks.\\2.attn_block.out_proj.weight", new_nm
    )

    new_nm <- sub(
      "^blocks\\.([0-9]+)\\.layers\\.([0-9]+)\\.layers\\.window_attention\\.attn_layer\\.1\\.merge\\.bias",
      "stages.\\1.attn_blocks.\\2.attn_block.out_proj.bias", new_nm
    )





    # === MLP ===
    new_nm <- sub("\\.mlp\\.fc1", ".mlp.0", new_nm)
    new_nm <- sub("\\.mlp\\.fc2", ".mlp.2", new_nm)

    # Store renamed key
    renamed[[new_nm]] <- state_dict[[nm]]
  }

  renamed
}



#' Constructs a MaxViT classification model
#'
#' This implementation is based on the "MaxViT: Multi-Axis Vision Transformer" paper.
#'
#' @param pretrained If TRUE, returns a model pre-trained on ImageNet-1K
#' @param progress If TRUE, displays a progress bar of the download to stderr
#' @param num_classes Number of output classes. Default is 1000 for ImageNet.
#' @return A `nn_module` representing the MaxViT model
#' @export
model_maxvit <- function(pretrained = FALSE, progress = TRUE, num_classes = 1000, ...) {
  model <- maxvit_impl(num_classes = num_classes)

  if (pretrained) {
    path <- download_and_cache("https://torch-cdn.mlverse.org/models/vision/v2/models/maxvit.pth")
    state_dict <- torch::load_state_dict(path)
    state_dict <- .rename_maxvit_state_dict(state_dict)
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}
