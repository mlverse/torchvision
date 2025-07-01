#' EfficientNet Models
#'
#' Constructs EfficientNet model architectures as described in
#' \emph{EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}.
#' These models are designed for image classification tasks and provide a balance
#' between accuracy and computational efficiency through compound scaling.
#'
#' @section Task:
#' Image classification with 1000 output classes by default (ImageNet).
#'
#' @section Input Format:
#' The models expect input tensors of shape \code{(batch_size, 3, H, W)}, where H and W
#' should typically be 224 for B0 and scaled versions for B1–B7 (e.g., B7 uses 600x600).
#'
#' @section Variants and Scaling:
#' \tabular{lll}{
#'   \strong{Model} \tab \strong{Width} \tab \strong{Depth} \cr
#'   B0 \tab 1.0 \tab 1.0 \cr
#'   B1 \tab 1.0 \tab 1.1 \cr
#'   B2 \tab 1.1 \tab 1.2 \cr
#'   B3 \tab 1.2 \tab 1.4 \cr
#'   B4 \tab 1.4 \tab 1.8 \cr
#'   B5 \tab 1.6 \tab 2.2 \cr
#'   B6 \tab 1.8 \tab 2.6 \cr
#'   B7 \tab 2.0 \tab 3.1
#' }
#'
#' @inheritParams model_resnet18
#' @param ... Other parameters passed to the model implementation, such as
#' \code{num_classes} to change the output dimension.
#'
#' @examples
#' if (torch::cuda_is_available()) {
#'   model <- model_efficientnet_b0(pretrained = FALSE)
#'   input <- torch::torch_randn(1, 3, 224, 224)
#'   output <- model(input)
#'   print(output$shape)  # Should be (1, 1000) by default
#' }
#'
#' @family models
#'
#' @examples
#' \dontrun{
#' model <- model_efficientnet_b0()
#' input <- torch::torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' dim(output)
#' }
#'
#' @name model_efficientnet
NULL


conv_norm_act <- torch::nn_module(
  inherit = torch::nn_sequential,
  initialize = function(in_channels, out_channels, kernel_size = 3, stride = 1,
                        groups = 1, norm_layer = NULL, activation_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d
    if (is.null(activation_layer))
      activation_layer <- torch::nn_silu
    padding <- (kernel_size - 1) %/% 2
    super$initialize(
      torch::nn_conv2d(in_channels, out_channels, kernel_size, stride, padding,
                       groups = groups, bias = FALSE),
      norm_layer(out_channels),
      activation_layer(inplace = TRUE)
    )
    self$out_channels <- out_channels
  }
)

se_block <- torch::nn_module(
  initialize = function(in_channels, squeeze_channels) {
    self$avg_pool <- torch::nn_adaptive_avg_pool2d(output_size = 1)
    self$squeeze <- torch::nn_conv2d(in_channels, squeeze_channels,
                                     kernel_size = 1)
    self$relu <- torch::nn_relu()
    self$expand <- torch::nn_conv2d(squeeze_channels, in_channels,
                                    kernel_size = 1)
    self$sigmoid <- torch::nn_sigmoid()
  },
  forward = function(x) {
    scale <- self$avg_pool(x)
    scale <- self$squeeze(scale)
    scale <- self$relu(scale)
    scale <- self$expand(scale)
    scale <- self$sigmoid(scale)
    x * scale
  }
)



mbconv_block <- torch::nn_module(
  initialize = function(in_channels, out_channels, kernel_size, stride,
                        expand_ratio, se_ratio = 0.25, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d
    hidden_dim <- in_channels * expand_ratio
    self$use_res_connect <- stride == 1 && in_channels == out_channels
    layers <- list()

    # Expand
    if (expand_ratio != 1) {
      layers[[length(layers) + 1]] <- conv_norm_act(
        in_channels, hidden_dim, kernel_size = 1,
        norm_layer = norm_layer, activation_layer = torch::nn_silu
      )
    }

    # Depthwise
    layers[[length(layers) + 1]] <- conv_norm_act(
      hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride,
      groups = hidden_dim, norm_layer = norm_layer,
      activation_layer = torch::nn_silu
    )

    # SE Block
    if (!is.null(se_ratio) && se_ratio > 0) {
      squeeze_channels <- max(1L, hidden_dim %/% as.integer(1 / se_ratio))
      layers[[length(layers) + 1]] <- se_block(hidden_dim, squeeze_channels = squeeze_channels)
    }

    # Projection (fix: wrap in sequential to get block.2.0 / block.2.1)
    layers[[length(layers) + 1]] <- torch::nn_sequential(
      torch::nn_conv2d(hidden_dim, out_channels, 1, bias = FALSE),
      norm_layer(out_channels)
    )

    self$block <- torch::nn_sequential(!!!layers)
  },
  forward = function(x) {
    out <- self$block(x)
    if (self$use_res_connect)
      out <- out + x
    out
  }
)

efficientnet <- torch::nn_module(
  "efficientnet",
  initialize = function(width_coefficient = 1, depth_coefficient = 1,
                        dropout = 0.2, num_classes = 1000, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d
    b0_cfg <- list(
      list(expand = 1, channels = 16, repeats = 1, stride = 1, kernel = 3),
      list(expand = 6, channels = 24, repeats = 2, stride = 2, kernel = 3),
      list(expand = 6, channels = 40, repeats = 2, stride = 2, kernel = 5),
      list(expand = 6, channels = 80, repeats = 3, stride = 2, kernel = 3),
      list(expand = 6, channels = 112, repeats = 3, stride = 1, kernel = 5),
      list(expand = 6, channels = 192, repeats = 4, stride = 2, kernel = 5),
      list(expand = 6, channels = 320, repeats = 1, stride = 1, kernel = 3)
    )
    round_filters <- function(filters) {
      divisor <- 8
      filters <- filters * width_coefficient
      new_filters <- max(divisor, as.integer(filters + divisor/2) %/% divisor * divisor)
      if (new_filters < 0.9 * filters)
        new_filters <- new_filters + divisor
      as.integer(new_filters)
    }
    round_repeats <- function(repeats) {
      as.integer(ceiling(depth_coefficient * repeats))
    }
    out_channels <- round_filters(32)
    features <- list(conv_norm_act(3, out_channels, stride = 2, norm_layer = norm_layer, activation_layer = torch::nn_silu))
    in_channels <- out_channels

    for (cfg in b0_cfg) {
      oc <- round_filters(cfg$channels)
      r <- round_repeats(cfg$repeats)
      stage_blocks <- list()
      for (i in 1:r) {
        s <- if (i == 1) cfg$stride else 1
        stage_blocks[[i]] <- mbconv_block(in_channels, oc,
                                          kernel_size = cfg$kernel, stride = s, expand_ratio = cfg$expand,
                                          se_ratio = 0.25, norm_layer = norm_layer)
        in_channels <- oc
      }
      features[[length(features) + 1]] <- torch::nn_sequential(!!!stage_blocks)
    }

    final_channels <- round_filters(1280)
    features[[length(features) + 1]] <- conv_norm_act(in_channels, final_channels,
                                                      kernel_size = 1, norm_layer = norm_layer, activation_layer = torch::nn_silu)
    self$features <- torch::nn_sequential(!!!features)
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
    self$classifier <- torch::nn_sequential(
      torch::nn_dropout(dropout),
      torch::nn_linear(final_channels, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    self$classifier(x)
  }
)

effnet <- function(arch, width, depth, dropout, pretrained, progress, ...) {
  args <- rlang::list2(...)
  model <- do.call(efficientnet, append(args, list(
    width_coefficient = width,
    depth_coefficient = depth,
    dropout = dropout
  )))

  if (pretrained) {
    local_path <- here::here("tools", "models", paste0(arch, ".pth"))
    message("Using local weights: ", local_path)

    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked", names(state_dict))]

  }

  model
}


# Individual model variants

#' @describeIn model_efficientnet EfficientNet B0 model
#' @export
model_efficientnet_b0 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b0", 1.0, 1.0, 0.2, pretrained, progress, ...)
}


#' @describeIn model_efficientnet EfficientNet B1 model
#' @export
model_efficientnet_b1 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b1", 1.0, 1.1, 0.2, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B2 model
#' @export
model_efficientnet_b2 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b2", 1.1, 1.2, 0.3, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B3 model
#' @export
model_efficientnet_b3 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b3", 1.2, 1.4, 0.3, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B4 model
#' @export
model_efficientnet_b4 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b4", 1.4, 1.8, 0.4, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B5 model
#' @export
model_efficientnet_b5 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b5", 1.6, 2.2, 0.4, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B6 model
#' @export
model_efficientnet_b6 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b6", 1.8, 2.6, 0.5, pretrained, progress, ...)
}

#' @describeIn model_efficientnet EfficientNet B7 model
#' @export
model_efficientnet_b7 <- function(pretrained = FALSE, progress = TRUE, ...) {
  effnet("efficientnet_b7", 2.0, 3.1, 0.5, pretrained, progress, ...)
}

efficientnet_model_urls <- c(
  efficientnet_b0 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b0.pth",
  efficientnet_b1 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b1.pth",
  efficientnet_b2 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b2.pth",
  efficientnet_b3 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b3.pth",
  efficientnet_b4 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b4.pth",
  efficientnet_b5 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b5.pth",
  efficientnet_b6 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b6.pth",
  efficientnet_b7 = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b7.pth"
)
