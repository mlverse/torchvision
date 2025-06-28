# EfficientNet R implementation that mirrors PyTorch structure
# This version is designed to load PyTorch .pth files directly without key remapping

library(torch)

# Helper function for MBConv block
mbconv_block <- nn_module(
  initialize = function(in_channels, out_channels, expand_ratio, stride, kernel_size, se_ratio, norm_layer = NULL) {
    self$use_residual <- stride == 1 && in_channels == out_channels

    if (is.null(norm_layer))
      norm_layer <- nn_batch_norm2d

    hidden_dim <- in_channels * expand_ratio
    layers <- list()

    if (expand_ratio != 1) {
      layers[[length(layers)+1]] <- nn_conv2d(in_channels, hidden_dim, 1, bias = FALSE)
      layers[[length(layers)+1]] <- norm_layer(hidden_dim)
      layers[[length(layers)+1]] <- nn_silu()
    }

    layers[[length(layers)+1]] <- nn_conv2d(hidden_dim, hidden_dim, kernel_size, stride = stride,
                                            padding = (kernel_size - 1) %/% 2,
                                            groups = hidden_dim, bias = FALSE)
    layers[[length(layers)+1]] <- norm_layer(hidden_dim)
    layers[[length(layers)+1]] <- nn_silu()

    squeeze_channels <- as.integer(in_channels * se_ratio)
    self$se <- nn_sequential(
      nn_adaptive_avg_pool2d(1),
      nn_conv2d(hidden_dim, squeeze_channels, 1),
      nn_silu(),
      nn_conv2d(squeeze_channels, hidden_dim, 1),
      nn_sigmoid()
    )

    layers[[length(layers)+1]] <- nn_conv2d(hidden_dim, out_channels, 1, bias = FALSE)
    layers[[length(layers)+1]] <- norm_layer(out_channels)

    self$block <- nn_sequential(!!!layers)
  },

  forward = function(x) {
    identity <- x
    out <- x
    for (layer in self$block) {
      out <- layer(out)
    }
    se_weight <- self$se(out)
    out <- out * se_weight
    if (self$use_residual)
      out <- out + identity
    out
  }
)

# Full EfficientNet backbone
efficientnet_backbone <- nn_module(
  initialize = function(cfg, dropout, num_classes = 1000, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- nn_batch_norm2d

    self$features <- nn_sequential(
      nn_conv2d(3, cfg$stem_out, kernel_size = 3, stride = 2, padding = 1, bias = FALSE),
      norm_layer(cfg$stem_out),
      nn_silu()
    )

    in_channels <- cfg$stem_out
    for (stage in cfg$stages) {
      for (block in stage) {
        mb <- mbconv_block(
          in_channels = in_channels,
          out_channels = block$out,
          expand_ratio = block$expand,
          stride = block$stride,
          kernel_size = block$kernel,
          se_ratio = block$se,
          norm_layer = norm_layer
        )
        block_name <- paste0("mbconv_", length(self$features))
        self$features$add_module(block_name, mb)

        in_channels <- block$out
      }
    }

    self$features$add_module("final_conv",
                             nn_conv2d(in_channels, cfg$final_out, kernel_size = 1, bias = FALSE))
    self$features$add_module("final_bn", norm_layer(cfg$final_out))
    self$features$add_module("final_act", nn_silu())

    self$avgpool <- nn_adaptive_avg_pool2d(1)
    self$classifier <- nn_sequential(
      nn_dropout(p = dropout),
      nn_linear(cfg$final_out, num_classes)
    )
  },

  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
    x
  }
)

# EfficientNet-B0 config
efficientnet_b0_config <- list(
  stem_out = 32,
  final_out = 1280,
  stages = list(
    list(list(out=16,  expand=1,  stride=1, kernel=3, se=0.25)),
    list(list(out=24,  expand=6,  stride=2, kernel=3, se=0.25),
         list(out=24,  expand=6,  stride=1, kernel=3, se=0.25)),
    list(list(out=40,  expand=6,  stride=2, kernel=5, se=0.25),
         list(out=40,  expand=6,  stride=1, kernel=5, se=0.25)),
    list(list(out=80,  expand=6,  stride=2, kernel=3, se=0.25),
         list(out=80,  expand=6,  stride=1, kernel=3, se=0.25),
         list(out=80,  expand=6,  stride=1, kernel=3, se=0.25)),
    list(list(out=112, expand=6,  stride=1, kernel=5, se=0.25),
         list(out=112, expand=6,  stride=1, kernel=5, se=0.25)),
    list(list(out=192, expand=6,  stride=2, kernel=5, se=0.25),
         list(out=192, expand=6,  stride=1, kernel=5, se=0.25),
         list(out=192, expand=6,  stride=1, kernel=5, se=0.25)),
    list(list(out=320, expand=6,  stride=1, kernel=3, se=0.25))
  )
)

# Model constructor
model_efficientnet_b0 <- function(pretrained = FALSE, path = NULL, ...) {
  model <- efficientnet_backbone(cfg = efficientnet_b0_config, dropout = 0.2, ...)

  if (pretrained) {
    if (is.null(path))
      stop("Please provide path to pretrained PyTorch .pth file")
    state_dict <- torch::load_state_dict(path)
    model$load_state_dict(state_dict)
  }

  model
}
