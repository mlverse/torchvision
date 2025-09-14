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
#' \tabular{lllllll}{
#'   \strong{Model} \tab \strong{Width} \tab \strong{Depth} \tab
#'   \strong{Resolution} \tab \strong{Params (M)} \tab \strong{GFLOPs} \tab
#'   \strong{Top-1 Acc.} \cr
#'   B0 \tab 1.0 \tab 1.0 \tab 224 \tab 5.3 \tab 0.39 \tab 77.1 \cr
#'   B1 \tab 1.0 \tab 1.1 \tab 240 \tab 7.8 \tab 0.70 \tab 79.1 \cr
#'   B2 \tab 1.1 \tab 1.2 \tab 260 \tab 9.2 \tab 1.00 \tab 80.1 \cr
#'   B3 \tab 1.2 \tab 1.4 \tab 300 \tab 12.0 \tab 1.80 \tab 81.6 \cr
#'   B4 \tab 1.4 \tab 1.8 \tab 380 \tab 19.0 \tab 4.20 \tab 82.9 \cr
#'   B5 \tab 1.6 \tab 2.2 \tab 456 \tab 30.0 \tab 9.90 \tab 83.6 \cr
#'   B6 \tab 1.8 \tab 2.6 \tab 528 \tab 43.0 \tab 19.0 \tab 84.0 \cr
#'   B7 \tab 2.0 \tab 3.1 \tab 600 \tab 66.0 \tab 37.0 \tab 84.3
#' }
#'
#' @inheritParams model_resnet18
#' @param ... Other parameters passed to the model implementation, such as
#' \code{num_classes} to change the output dimension.
#'
#' @family classification_model
#'
#' @examples
#' \dontrun{
#' model <- model_efficientnet_b0()
#' image_batch <- torch::torch_randn(1, 3, 224, 224)
#' output <- model(image_batch)
#' imagenet_label(which.max(as.numeric(output)))
#' }
#'
#' \dontrun{
#' # Example of using EfficientNet-B5 with its native image size
#' model <- model_efficientnet_b5()
#' image_batch <- torch::torch_randn(1, 3, 456, 456)
#' output <- model(image_batch)
#' imagenet_label(which.max(as.numeric(output)))
#' }
#'
#' @name model_efficientnet
NULL

efficientnet_model_urls <- list(
  efficientnet_b0 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b0.pth", "5cb997d28f1a30cdf1732dd1e69a6647", "~22 MB"),
  efficientnet_b1 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b1.pth", "1f1fdcc560d5a91875bbec02c0a105e8", "~32 MB"),
  efficientnet_b2 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b2.pth", "c5440198bb37adc12d8d88f58863b2d3", "~37 MB"),
  efficientnet_b3 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b3.pth", "74967d21f6d437845ab5f2fd87f31df1", "~50 MB"),
  efficientnet_b4 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b4.pth", "a044c7adf3bf4c457581669eae41f713", "~80 MB"),
  efficientnet_b5 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b5.pth", "8bea08f18d29d5becce388f363247a26", "~122 MB"),
  efficientnet_b6 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b6.pth", "96844282790aff48afe3c56b434f4863", "~175 MB"),
  efficientnet_b7 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_b7.pth", "c3ddfc4df9851a8fe55312fdbe8f8c7e", "~270 MB")
)

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
    self$avgpool <- torch::nn_adaptive_avg_pool2d(output_size = 1)
    self$fc1 <- torch::nn_conv2d(in_channels, squeeze_channels, kernel_size = 1)
    self$activation <- torch::nn_relu()
    self$fc2 <- torch::nn_conv2d(squeeze_channels, in_channels, kernel_size = 1)
    self$scale_activation <- torch::nn_sigmoid()
  },
  forward = function(x) {
    scale <- self$avgpool(x)
    scale <- self$fc1(scale)
    scale <- self$activation(scale)
    scale <- self$fc2(scale)
    scale <- self$scale_activation(scale)
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

    # Correct SE block with proper squeeze logic
    if (!is.null(se_ratio) && se_ratio > 0) {
      squeeze_channels <- max(1, as.integer(in_channels * se_ratio))
      layers[[length(layers) + 1]] <- se_block(
        hidden_dim, squeeze_channels = squeeze_channels
      )
    }

    # Projection
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
        s <- ifelse(i == 1, cfg$stride, 1)
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
    r <- efficientnet_model_urls[[arch]]
    cli_inform("Model weights for {.cls {arch}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    state_dict <- state_dict[!grepl("num_batches_tracked", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
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
