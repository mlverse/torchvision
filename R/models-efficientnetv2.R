#' EfficientNetV2 Models
#'
#' Constructs EfficientNetV2 model architectures as described in
#' \emph{EfficientNetV2: Smaller Models and Faster Training}.
#'
#' @inheritParams model_resnet18
#' @param ... Other parameters passed to the model implementation, such as
#'   \code{num_classes} to change the output dimension.
#'
#' @family models
#' @name model_efficientnet_v2
#' @seealso \code{\link{model_efficientnet}}
#' @examples
#' \dontrun{
#' model <- model_efficientnet_v2_s()
#' input <- torch::torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' }
NULL

fused_mbconv_block <- torch::nn_module(
  initialize = function(in_channels, out_channels, kernel_size, stride,
                        expand_ratio, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d
    hidden_dim <- as.integer(in_channels * expand_ratio)
    self$use_res_connect <- stride == 1 && in_channels == out_channels
    layers <- list()

    if (expand_ratio != 1) {
      # Fused: expansion + spatial convolution in one step
      layers[[length(layers) + 1]] <- conv_norm_act(
        in_channels, hidden_dim, kernel_size = kernel_size, stride = stride,
        norm_layer = norm_layer, activation_layer = torch::nn_silu
      )
      # Project back to output channels
      layers[[length(layers) + 1]] <- torch::nn_sequential(
        torch::nn_conv2d(hidden_dim, out_channels, kernel_size = 1, stride = 1, bias = FALSE),
        norm_layer(out_channels)
      )
    } else {
      # When expand_ratio == 1, just do a single convolution
      layers[[length(layers) + 1]] <- torch::nn_sequential(
        torch::nn_conv2d(in_channels, out_channels, kernel_size = kernel_size,
                         stride = stride, padding = (kernel_size - 1) %/% 2, bias = FALSE),
        norm_layer(out_channels),
        torch::nn_silu(inplace = TRUE)
      )
    }

    self$block <- torch::nn_sequential(!!!layers)
  },
  forward = function(x) {
    out <- self$block(x)
    if (self$use_res_connect)
      out <- out + x
    out
  }
)

efficientnet_v2 <- torch::nn_module(
  "efficientnet_v2",
  initialize = function(cfgs, dropout = 0.2, num_classes = 1000,
                        norm_layer = NULL, firstconv_out = 24) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    features <- list(
      conv_norm_act(3, firstconv_out, stride = 2,
                    norm_layer = norm_layer, activation_layer = torch::nn_silu)
    )
    in_channels <- firstconv_out

    for (cfg in cfgs) {
      oc <- cfg$channels
      r <- cfg$repeats
      block_fn <- if (identical(cfg$block, "fused")) fused_mbconv_block else mbconv_block
      stage_blocks <- list()
      for (i in seq_len(r)) {
        s <- if (i == 1) cfg$stride else 1
        if (identical(cfg$block, "fused")) {
          stage_blocks[[i]] <- block_fn(
            in_channels, oc, kernel_size = cfg$kernel, stride = s,
            expand_ratio = cfg$expand, norm_layer = norm_layer
          )
        } else {
          # For mbconv blocks, use the v2 block with corrected SE calculation
          stage_blocks[[i]] <- mbconv_block_v2(
            in_channels, oc, kernel_size = cfg$kernel, stride = s,
            expand_ratio = cfg$expand, se_ratio = 0.25, norm_layer = norm_layer
          )
        }
        in_channels <- oc
      }
      features[[length(features) + 1]] <- torch::nn_sequential(!!!stage_blocks)
    }

    final_channels <- 1280
    features[[length(features) + 1]] <- conv_norm_act(
      in_channels, final_channels, kernel_size = 1,
      norm_layer = norm_layer, activation_layer = torch::nn_silu
    )
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

# mbconv_block for EfficientNetV2
mbconv_block_v2 <- torch::nn_module(
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

    # SE block
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

effnetv2 <- function(arch, cfgs, dropout, firstconv_out, pretrained, progress, ...) {
  args <- rlang::list2(...)

  model <- do.call(efficientnet_v2, append(args, list(
    cfgs = cfgs,
    dropout = dropout,
    firstconv_out = firstconv_out
  )))

  if (pretrained) {
    local_path <- here::here("tools", "models", paste0(arch, ".pth"))
    print(local_path)
    state_dict <- torch::load_state_dict(local_path)
    model$load_state_dict(state_dict)
  }

  model
}

#' @describeIn model_efficientnet_v2 EfficientNetV2-S model
#' @export
model_efficientnet_v2_s <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfgs <- list(
    list(block = "fused", expand = 1, channels = 24, repeats = 2, stride = 1, kernel = 3),
    list(block = "fused", expand = 4, channels = 48, repeats = 4, stride = 2, kernel = 3),
    list(block = "fused", expand = 4, channels = 64, repeats = 4, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 4, channels = 128, repeats = 6, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 160, repeats = 9, stride = 1, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 256, repeats = 15, stride = 2, kernel = 3)
  )
  effnetv2("efficientnet_v2_s", cfgs, 0.2, 24, pretrained, progress, ...)
}

#' @describeIn model_efficientnet_v2 EfficientNetV2-M model
#' @export
model_efficientnet_v2_m <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfgs <- list(
    list(block = "fused", expand = 1, channels = 24, repeats = 3, stride = 1, kernel = 3),
    list(block = "fused", expand = 4, channels = 48, repeats = 5, stride = 2, kernel = 3),
    list(block = "fused", expand = 4, channels = 80, repeats = 5, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 4, channels = 160, repeats = 7, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 176, repeats = 14, stride = 1, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 304, repeats = 18, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 512, repeats = 5, stride = 1, kernel = 3)
  )
  effnetv2("efficientnet_v2_m", cfgs, 0.3, 24, pretrained, progress, ...)
}

#' @describeIn model_efficientnet_v2 EfficientNetV2-L model
#' @export
model_efficientnet_v2_l <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfgs <- list(
    list(block = "fused", expand = 1, channels = 32, repeats = 4, stride = 1, kernel = 3),
    list(block = "fused", expand = 4, channels = 64, repeats = 7, stride = 2, kernel = 3),
    list(block = "fused", expand = 4, channels = 96, repeats = 7, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 4, channels = 192, repeats = 10, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 224, repeats = 19, stride = 1, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 384, repeats = 25, stride = 2, kernel = 3),
    list(block = "mbconv", expand = 6, channels = 640, repeats = 7, stride = 1, kernel = 3)
  )
  effnetv2("efficientnet_v2_l", cfgs, 0.4, 32, pretrained, progress, ...)
}

efficientnet_v2_model_urls <- c(
  efficientnet_v2_s = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_v2_s.pth",
  efficientnet_v2_m = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_v2_m.pth",
  efficientnet_v2_l = "https://torch-cdn.mlverse.org/models/vision/v2/models/efficientnet_v2_l.pth"
)
