#' EfficientNet implementation
#'
#' EfficientNet models are based on the architecture described in
#' [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
#'
#' @inheritParams model_resnet18
#' @param ... Other parameters passed to the EfficientNet implementation.
#'
#' @family models
#' @name model_efficientnet
NULL

.efficientnet_urls <- c(
  'efficientnet_b0' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b0.pth',
  'efficientnet_b1' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b1.pth',
  'efficientnet_b2' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b2.pth',
  'efficientnet_b3' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b3.pth',
  'efficientnet_b4' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b4.pth',
  'efficientnet_b5' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b5.pth',
  'efficientnet_b6' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b6.pth',
  'efficientnet_b7' = 'https://torch-cdn.mlverse.org/models/vision/v3/models/efficientnet_b7.pth'
)

make_divisible <- function(v, divisor = 8) {
  new_v <- max(divisor, as.integer(v + divisor / 2) %/% divisor * divisor)
  if (new_v < 0.9 * v)
    new_v <- new_v + divisor
  new_v
}

round_filters <- function(filters, width_mult) {
  make_divisible(filters * width_mult)
}

round_repeats <- function(repeats, depth_mult) {
  as.integer(ceiling(repeats * depth_mult))
}

squeeze_excitation <- torch::nn_module(
  "squeeze_excitation",
  initialize = function(input_channels, squeeze_channels) {
    self$avg_pool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
    self$fc1 <- torch::nn_conv2d(input_channels, squeeze_channels, 1)
    self$act1 <- torch::nn_silu()
    self$fc2 <- torch::nn_conv2d(squeeze_channels, input_channels, 1)
    self$act2 <- torch::nn_sigmoid()
  },
  forward = function(x) {
    scale <- self$avg_pool(x)
    scale <- self$fc1(scale)
    scale <- self$act1(scale)
    scale <- self$fc2(scale)
    scale <- self$act2(scale)
    x * scale
  }
)

mbconv <- torch::nn_module(
  "mbconv",
  initialize = function(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio = 0.25, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    self$use_res_connect <- stride == 1 && in_channels == out_channels
    expanded_channels <- as.integer(in_channels * expand_ratio)

    self$expand_ratio <- expand_ratio
    if (expand_ratio != 1) {
      self$expand_conv <- torch::nn_conv2d(in_channels, expanded_channels, 1, bias = FALSE)
      self$bn0 <- norm_layer(expanded_channels)
    }

    padding <- (kernel_size - 1) %/% 2
    self$conv_dw <- torch::nn_conv2d(expanded_channels, expanded_channels, kernel_size,
                                     stride = stride, padding = padding, groups = expanded_channels,
                                     bias = FALSE)
    self$bn1 <- norm_layer(expanded_channels)
    squeeze_channels <- max(1, as.integer(in_channels * se_ratio))
    self$se <- squeeze_excitation(expanded_channels, squeeze_channels)
    self$project_conv <- torch::nn_conv2d(expanded_channels, out_channels, 1, bias = FALSE)
    self$bn2 <- norm_layer(out_channels)
    self$act <- torch::nn_silu()
  },
  forward = function(input) {
    x <- input
    if (self$expand_ratio != 1) {
      x <- self$expand_conv(x)
      x <- self$bn0(x)
      x <- self$act(x)
    }
    x <- self$conv_dw(x)
    x <- self$bn1(x)
    x <- self$act(x)
    x <- self$se(x)
    x <- self$project_conv(x)
    x <- self$bn2(x)
    if (self$use_res_connect)
      x <- x + input
    x
  }
)

efficientnet <- torch::nn_module(
  "efficientnet",
  initialize = function(width_mult = 1, depth_mult = 1, dropout = 0.2, num_classes = 1000, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    cfgs <- list(
      list(1, 3, 1, 32, 16, 1),
      list(6, 3, 2, 16, 24, 2),
      list(6, 5, 2, 24, 40, 2),
      list(6, 3, 2, 40, 80, 3),
      list(6, 5, 1, 80, 112, 3),
      list(6, 5, 2, 112, 192, 4),
      list(6, 3, 1, 192, 320, 1)
    )

    out_channels <- round_filters(32, width_mult)
    self$stem <- torch::nn_sequential(
      torch::nn_conv2d(3, out_channels, 3, stride = 2, padding = 1, bias = FALSE),
      norm_layer(out_channels),
      torch::nn_silu()
    )
    in_channels <- out_channels

    blocks <- list()
    for (cfg in cfgs) {
      names(cfg) <- c("expand_ratio", "kernel", "stride", "in", "out", "repeats")
      cfg <- as.list(cfg)
      cfg$out <- round_filters(cfg$out, width_mult)
      cfg$repeats <- round_repeats(cfg$repeats, depth_mult)
      for (i in seq_len(cfg$repeats)) {
        stride <- if (i == 1) cfg$stride else 1
        blocks[[length(blocks) + 1]] <- mbconv(in_channels, cfg$out, cfg$kernel, stride, cfg$expand_ratio)
        in_channels <- cfg$out
      }
    }
    self$blocks <- torch::nn_sequential(!!!blocks)
    head_channels <- round_filters(1280, width_mult)
    self$head <- torch::nn_sequential(
      torch::nn_conv2d(in_channels, head_channels, 1, bias = FALSE),
      norm_layer(head_channels),
      torch::nn_silu()
    )
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
    self$classifier <- torch::nn_sequential(
      torch::nn_dropout(p = dropout),
      torch::nn_linear(head_channels, num_classes)
    )
  },
  forward = function(x) {
    x <- self$stem(x)
    x <- self$blocks(x)
    x <- self$head(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
    x
  }
)

.efficientnet <- function(arch, width_mult, depth_mult, dropout, pretrained, progress, ...) {
  model <- efficientnet(width_mult, depth_mult, dropout, ...)
  if (pretrained) {
    state_dict_path <- download_and_cache(.efficientnet_urls[[arch]])
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }
  model
}

#' @describeIn model_efficientnet EfficientNet B0 model
#' @export
model_efficientnet_b0 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B1 model
#' @export
model_efficientnet_b1 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B2 model
#' @export
model_efficientnet_b2 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b2", 1.1, 1.2, 0.3, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B3 model
#' @export
model_efficientnet_b3 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b3", 1.2, 1.4, 0.3, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B4 model
#' @export
model_efficientnet_b4 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b4", 1.4, 1.8, 0.4, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B5 model
#' @export
model_efficientnet_b5 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b5", 1.6, 2.2, 0.4, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B6 model
#' @export
model_efficientnet_b6 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b6", 1.8, 2.6, 0.5, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B7 model
#' @export
model_efficientnet_b7 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .efficientnet("efficientnet_b7", 2.0, 3.1, 0.5, pretrained, progress, ...)
}
