#' EfficientNet model variants (B0–B7)
#'
#' EfficientNet is a family of image classification models that uniformly scale
#' depth, width, and resolution using a compound scaling method.
#' This implementation is based on the paper
#' [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
#'
#' Each model variant supports optional loading of pretrained weights from ImageNet.
#'
#' @param pretrained (bool): If TRUE, returns a model pre-trained on ImageNet.
#' @param progress (bool): If TRUE, displays a progress bar of the download to stderr.
#' @param ... Additional arguments passed to the model constructor.
#'
#' @return A `nn_module` representing the EfficientNet model.
#' @name model_efficientnet
#' @rdname model_efficientnet
#' @family models
NULL

efficientnet_model_paths <- c(
  "efficientnet_b0" = "efficientnet_weights/efficientnet_b0_converted.pth",
  "efficientnet_b1" = "efficientnet_weights/efficientnet_b1_converted.pth",
  "efficientnet_b2" = "efficientnet_weights/efficientnet_b2_converted.pth",
  "efficientnet_b3" = "efficientnet_weights/efficientnet_b3_converted.pth",
  "efficientnet_b4" = "efficientnet_weights/efficientnet_b4_converted.pth",
  "efficientnet_b5" = "efficientnet_weights/efficientnet_b5_converted.pth",
  "efficientnet_b6" = "efficientnet_weights/efficientnet_b6_converted.pth",
  "efficientnet_b7" = "efficientnet_weights/efficientnet_b7_converted.pth"
)

mbconv_block <- torch::nn_module(
  "mbconv_block",
  initialize = function(in_channels, out_channels, expand_ratio = 1, stride = 1, kernel_size = 3, se_ratio = 0.25) {
    hidden_dim <- in_channels * expand_ratio
    self$use_residual <- (stride == 1 && in_channels == out_channels)

    layers <- list()

    if (expand_ratio != 1) {
      # 1x1 expansion
      layers <- append(layers, list(
        torch::nn_conv2d(in_channels, hidden_dim, kernel_size = 1, bias = FALSE),
        torch::nn_batch_norm2d(hidden_dim),
        torch::nn_silu()
      ))
    }

    # Depthwise convolution
    layers <- append(layers, list(
      torch::nn_conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride,
                       padding = kernel_size %/% 2, groups = hidden_dim, bias = FALSE),
      torch::nn_batch_norm2d(hidden_dim),
      torch::nn_swish()
    ))

    # Squeeze-and-Excitation
    se_hidden <- max(1L, as.integer(in_channels * se_ratio))
    self$se <- torch::nn_sequential(
      torch::nn_adaptive_avg_pool2d(output_size = 1),
      torch::nn_conv2d(hidden_dim, se_hidden, kernel_size = 1),
      torch::nn_relu(),
      torch::nn_conv2d(se_hidden, hidden_dim, kernel_size = 1),
      torch::nn_sigmoid()
    )

    # Projection layer
    layers <- append(layers, list(
      torch::nn_conv2d(hidden_dim, out_channels, kernel_size = 1, bias = FALSE),
      torch::nn_batch_norm2d(out_channels)
    ))

    self$block <- do.call(torch::nn_sequential, layers)
  },
  forward = function(x) {
    identity <- x

    out <- self$block(x)

    # Apply SE
    se_out <- self$se(out)
    out <- out * se_out

    if (self$use_residual) {
      out <- out + identity
    }

    out
  }
)

efficientnet_config_b0 <- function() {
  list(
    input_channels = 3,
    stem_out = 32,
    blocks = list(
      list(in_channels=32, out_channels=16, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=16, out_channels=24, kernel_size=3, stride=2, expand_ratio=6, num_repeat=2),
      list(in_channels=24, out_channels=40, kernel_size=5, stride=2, expand_ratio=6, num_repeat=2),
      list(in_channels=40, out_channels=80, kernel_size=3, stride=2, expand_ratio=6, num_repeat=3),
      list(in_channels=80, out_channels=112, kernel_size=5, stride=1, expand_ratio=6, num_repeat=3),
      list(in_channels=112, out_channels=192, kernel_size=5, stride=2, expand_ratio=6, num_repeat=4),
      list(in_channels=192, out_channels=320, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 1280
  )
}

efficientnet_config_b1 <- function() {
  list(
    input_channels = 3,
    stem_out = 32,
    blocks = list(
      list(in_channels=32, out_channels=16, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=16, out_channels=24, kernel_size=3, stride=2, expand_ratio=6, num_repeat=2),
      list(in_channels=24, out_channels=40, kernel_size=5, stride=2, expand_ratio=6, num_repeat=2),
      list(in_channels=40, out_channels=80, kernel_size=3, stride=2, expand_ratio=6, num_repeat=3),
      list(in_channels=80, out_channels=112, kernel_size=5, stride=1, expand_ratio=6, num_repeat=3),
      list(in_channels=112, out_channels=192, kernel_size=5, stride=2, expand_ratio=6, num_repeat=4),
      list(in_channels=192, out_channels=320, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 1280
  )
}

efficientnet_config_b2 <- function() {
  list(
    input_channels = 3,
    stem_out = 32,
    blocks = list(
      list(in_channels=32, out_channels=16, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=16, out_channels=24, kernel_size=3, stride=2, expand_ratio=6, num_repeat=2),
      list(in_channels=24, out_channels=40, kernel_size=5, stride=2, expand_ratio=6, num_repeat=3),
      list(in_channels=40, out_channels=80, kernel_size=3, stride=2, expand_ratio=6, num_repeat=3),
      list(in_channels=80, out_channels=112, kernel_size=5, stride=1, expand_ratio=6, num_repeat=4),
      list(in_channels=112, out_channels=192, kernel_size=5, stride=2, expand_ratio=6, num_repeat=4),
      list(in_channels=192, out_channels=320, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 1408
  )
}

efficientnet_config_b3 <- function() {
  list(
    input_channels = 3,
    stem_out = 40,
    blocks = list(
      list(in_channels=40, out_channels=24, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=24, out_channels=32, kernel_size=3, stride=2, expand_ratio=6, num_repeat=3),
      list(in_channels=32, out_channels=48, kernel_size=5, stride=2, expand_ratio=6, num_repeat=3),
      list(in_channels=48, out_channels=96, kernel_size=3, stride=2, expand_ratio=6, num_repeat=5),
      list(in_channels=96, out_channels=136, kernel_size=5, stride=1, expand_ratio=6, num_repeat=4),
      list(in_channels=136, out_channels=232, kernel_size=5, stride=2, expand_ratio=6, num_repeat=5),
      list(in_channels=232, out_channels=384, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 1536
  )
}

efficientnet_config_b4 <- function() {
  list(
    input_channels = 3,
    stem_out = 48,
    blocks = list(
      list(in_channels=48, out_channels=24, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=24, out_channels=32, kernel_size=3, stride=2, expand_ratio=6, num_repeat=4),
      list(in_channels=32, out_channels=56, kernel_size=5, stride=2, expand_ratio=6, num_repeat=4),
      list(in_channels=56, out_channels=112, kernel_size=3, stride=2, expand_ratio=6, num_repeat=6),
      list(in_channels=112, out_channels=160, kernel_size=5, stride=1, expand_ratio=6, num_repeat=5),
      list(in_channels=160, out_channels=272, kernel_size=5, stride=2, expand_ratio=6, num_repeat=6),
      list(in_channels=272, out_channels=448, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 1792
  )
}

efficientnet_config_b5 <- function() {
  list(
    input_channels = 3,
    stem_out = 48,
    blocks = list(
      list(in_channels=48, out_channels=24, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=24, out_channels=40, kernel_size=3, stride=2, expand_ratio=6, num_repeat=4),
      list(in_channels=40, out_channels=64, kernel_size=5, stride=2, expand_ratio=6, num_repeat=4),
      list(in_channels=64, out_channels=128, kernel_size=3, stride=2, expand_ratio=6, num_repeat=6),
      list(in_channels=128, out_channels=176, kernel_size=5, stride=1, expand_ratio=6, num_repeat=6),
      list(in_channels=176, out_channels=304, kernel_size=5, stride=2, expand_ratio=6, num_repeat=8),
      list(in_channels=304, out_channels=512, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 2048
  )
}

efficientnet_config_b6 <- function() {
  list(
    input_channels = 3,
    stem_out = 56,
    blocks = list(
      list(in_channels=56, out_channels=32, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=32, out_channels=48, kernel_size=3, stride=2, expand_ratio=6, num_repeat=5),
      list(in_channels=48, out_channels=72, kernel_size=5, stride=2, expand_ratio=6, num_repeat=5),
      list(in_channels=72, out_channels=144, kernel_size=3, stride=2, expand_ratio=6, num_repeat=7),
      list(in_channels=144, out_channels=200, kernel_size=5, stride=1, expand_ratio=6, num_repeat=6),
      list(in_channels=200, out_channels=344, kernel_size=5, stride=2, expand_ratio=6, num_repeat=8),
      list(in_channels=344, out_channels=576, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 2304
  )
}

efficientnet_config_b7 <- function() {
  list(
    input_channels = 3,
    stem_out = 64,
    blocks = list(
      list(in_channels=64, out_channels=32, kernel_size=3, stride=1, expand_ratio=1, num_repeat=1),
      list(in_channels=32, out_channels=48, kernel_size=3, stride=2, expand_ratio=6, num_repeat=5),
      list(in_channels=48, out_channels=80, kernel_size=5, stride=2, expand_ratio=6, num_repeat=5),
      list(in_channels=80, out_channels=160, kernel_size=3, stride=2, expand_ratio=6, num_repeat=8),
      list(in_channels=160, out_channels=224, kernel_size=5, stride=1, expand_ratio=6, num_repeat=7),
      list(in_channels=224, out_channels=384, kernel_size=5, stride=2, expand_ratio=6, num_repeat=10),
      list(in_channels=384, out_channels=640, kernel_size=3, stride=1, expand_ratio=6, num_repeat=1)
    ),
    head_channels = 2560
  )
}

efficientnet <- torch::nn_module(
  "efficientnet",
  initialize = function(config, num_classes = 1000) {
    feature_layers <- list()

    # Stem
    feature_layers <- append(feature_layers, list(
      torch::nn_conv2d(config$input_channels, config$stem_out, kernel_size = 3, stride = 2, padding = 1, bias = FALSE),
      torch::nn_batch_norm2d(config$stem_out),
      torch::nn_silu()
    ))

    # MBConv blocks
    in_channels <- config$stem_out
    for (block_cfg in config$blocks) {
      for (i in seq_len(block_cfg$num_repeat)) {
        stride <- ifelse(i == 1, block_cfg$stride, 1)

        feature_layers <- append(feature_layers, list(
          mbconv_block(
            in_channels = in_channels,
            out_channels = block_cfg$out_channels,
            kernel_size = block_cfg$kernel_size,
            stride = stride,
            expand_ratio = block_cfg$expand_ratio
          )
        ))

        in_channels <- block_cfg$out_channels
      }
    }

    # Head
    feature_layers <- append(feature_layers, list(
      torch::nn_conv2d(in_channels, config$head_channels, kernel_size = 1, bias = FALSE),
      torch::nn_batch_norm2d(config$head_channels),
      torch::nn_swish()
    ))

    # FINAL: convert list to sequential module
    self$features <- do.call(torch::nn_sequential, feature_layers)

    self$avgpool <- torch::nn_adaptive_avg_pool2d(output_size = 1)
    self$classifier <- torch::nn_linear(config$head_channels, num_classes)
  },

  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
    x
  }
)


.efficientnet <- function(arch, config, pretrained = FALSE, progress = TRUE, ...) {
  model <- efficientnet(config, ...)

  if (pretrained) {
    weight_path <- efficientnet_model_paths[arch]
    state_dict <- torch::load_state_dict(weight_path)
    model$load_state_dict(state_dict)
  }

  model
}


#' @describeIn model_efficientnet EfficientNet B0
#' @export
model_efficientnet_b0 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b0()
  .efficientnet("efficientnet_b0", config, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B1
#' @export
model_efficientnet_b1 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b1()
  .efficientnet("efficientnet_b1", config, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B2
#' @export
model_efficientnet_b2 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b2()
  .efficientnet("efficientnet_b2", config, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B3
#' @export
model_efficientnet_b3 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b3()
  .efficientnet("efficientnet_b3", config, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B4
#' @export
model_efficientnet_b4 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b4()
  .efficientnet("efficientnet_b4", config, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B5
#' @export
model_efficientnet_b5 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b5()
  .efficientnet("efficientnet_b5", config, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B6
#' @export
model_efficientnet_b6 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b6()
  .efficientnet("efficientnet_b6", config, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B7
#' @export
model_efficientnet_b7 <- function(pretrained = FALSE, progress = TRUE, ...) {
  config <- efficientnet_config_b7()
  .efficientnet("efficientnet_b7", config, pretrained, progress, ...)
}

