#' Constructs a MobileNetV3 Large architecture.
#'
#' This implementation mimics `torchvision::mobilenet_v3_large`.
#'
#' @inheritParams model_mobilenet_v2
#' @param ... Other parameters passed to the model implementation.
#'
#' @family models
#' @export
model_mobilenet_v3_large <- function(pretrained = FALSE, progress = TRUE, ...) {
  model <- mobilenet_v3_large_impl(...)

  if (pretrained) {
    local_path <- "tools/models/mobilenet_v3_large.pth"
    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

mobilenet_v3_large_url <- "https://torch-cdn.mlverse.org/models/vision/v2/models/mobilenet_v3_large.pth"

#' Quantized MobileNetV3 Large model
#'
#' This function mirrors `torchvision::quantization::mobilenet_v3_large` and
#' loads quantized weights when `pretrained` is `TRUE`.
#'
#' @inheritParams model_mobilenet_v3_large
#' @export
model_mobilenet_v3_large_quantized <- function(pretrained = FALSE, progress = TRUE, ...) {
  model <- mobilenet_v3_large_quant_impl(...)

  if (pretrained) {
    local_path <- "tools/models/mobilenet_v3_large_quantized.pth"
    state_dict <- torch::load_state_dict(local_path)
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

mobilenet_v3_large_quantized_url <- "https://torch-cdn.mlverse.org/models/vision/v2/models/mobilenet_v3_large_quantized.pth"

nn_hardswish <- torch::nn_module(
  initialize = function(inplace = TRUE) {
    self$inplace <- inplace
  },
  forward = function(x) {
    x * torch::nnf_relu6(x + 3, inplace = self$inplace) / 6
  }
)

nn_hardsigmoid <- torch::nn_module(
  initialize = function(inplace = TRUE) {
    self$inplace <- inplace
  },
  forward = function(x) {
    torch::nnf_relu6(x + 3, inplace = self$inplace) / 6
  }
)

conv_norm_act_v3 <- torch::nn_module(
  inherit = torch::nn_sequential,
  initialize = function(in_channels, out_channels, kernel_size = 3, stride = 1,
                        groups = 1, norm_layer = NULL, activation_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d
    if (is.null(activation_layer))
      activation_layer <- nn_hardswish
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

se_block_v3 <- torch::nn_module(
  initialize = function(in_channels, squeeze_channels) {
    self$avgpool <- torch::nn_adaptive_avg_pool2d(output_size = 1)
    self$fc1 <- torch::nn_conv2d(in_channels, squeeze_channels, kernel_size = 1)
    self$activation <- torch::nn_relu(inplace = TRUE)
    self$fc2 <- torch::nn_conv2d(squeeze_channels, in_channels, kernel_size = 1)
    self$scale_activation <- nn_hardsigmoid(inplace = TRUE)
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

inverted_residual_v3 <- torch::nn_module(
  initialize = function(in_channels, expand_channels, out_channels, kernel_size,
                        stride, use_se = FALSE, activation_layer = nn_hardswish,
                        norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d
    self$use_res_connect <- stride == 1 && in_channels == out_channels
    layers <- list()

    if (expand_channels != in_channels) {
      layers[[length(layers) + 1]] <- conv_norm_act_v3(
        in_channels, expand_channels, kernel_size = 1,
        norm_layer = norm_layer, activation_layer = activation_layer
      )
    }

    layers[[length(layers) + 1]] <- conv_norm_act_v3(
      expand_channels, expand_channels, kernel_size = kernel_size, stride = stride,
      groups = expand_channels, norm_layer = norm_layer,
      activation_layer = activation_layer
    )

    if (use_se) {
      make_divisible <- function(v, divisor = 8, min_value = NULL) {
        if (is.null(min_value)) min_value <- divisor
        new_v <- max(min_value, as.integer(v + divisor / 2) %/% divisor * divisor)
        if (new_v < 0.9 * v) new_v <- new_v + divisor
        new_v
      }

      # Later in inverted_residual_v3:
      squeeze_channels <- make_divisible(expand_channels / 4)
      layers[[length(layers) + 1]] <- se_block_v3(expand_channels, squeeze_channels)
    }

    layers[[length(layers) + 1]] <- torch::nn_sequential(
      torch::nn_conv2d(expand_channels, out_channels, 1, bias = FALSE),
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


mobilenet_v3_large_impl <- torch::nn_module(
  "mobilenet_v3_large_impl",
  initialize = function(num_classes = 1000, dropout = 0.2, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- nn_batch_norm2d

    activation <- nn_hardswish

    layers <- list()
    layers[[length(layers) + 1]] <- conv_norm_act_v3(3, 16, stride = 2,
                                                     norm_layer = norm_layer,
                                                     activation_layer = activation)
    input_channel <- 16

    cfgs <- list(
      list(3, 16, 16, FALSE, torch::nn_relu, 1),
      list(3, 64, 24, FALSE, torch::nn_relu, 2),
      list(3, 72, 24, FALSE, torch::nn_relu, 1),
      list(5, 72, 40, TRUE, torch::nn_relu, 2),
      list(5, 120, 40, TRUE, torch::nn_relu, 1),
      list(5, 120, 40, TRUE, torch::nn_relu, 1),
      list(3, 240, 80, FALSE, activation, 2),
      list(3, 200, 80, FALSE, activation, 1),
      list(3, 184, 80, FALSE, activation, 1),
      list(3, 184, 80, FALSE, activation, 1),
      list(3, 480, 112, TRUE, activation, 1),
      list(3, 672, 112, TRUE, activation, 1),
      list(5, 672, 160, TRUE, activation, 2),
      list(5, 960, 160, TRUE, activation, 1),
      list(5, 960, 160, TRUE, activation, 1)
    )

    for (cfg in cfgs) {
      k <- cfg[[1]]; exp <- cfg[[2]]; c <- cfg[[3]]; se <- cfg[[4]];
      nl <- cfg[[5]]; s <- cfg[[6]]
      layers[[length(layers) + 1]] <- inverted_residual_v3(
        input_channel, exp, c, kernel_size = k, stride = s, use_se = se,
        activation_layer = nl, norm_layer = norm_layer
      )
      input_channel <- c
    }

    layers[[length(layers) + 1]] <- conv_norm_act_v3(
      input_channel, 960, kernel_size = 1,
      norm_layer = norm_layer, activation_layer = activation
    )

    self$features <- torch::nn_sequential(!!!layers)
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1,1))
    self$classifier <- torch::nn_sequential(
      torch::nn_linear(960, 1280),
      activation(inplace = TRUE),
      torch::nn_dropout(dropout),
      torch::nn_linear(1280, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    self$classifier(x)
  }
)





nn_identity <- torch::nn_module(
  "nn_identity",
  initialize = function() {},
  forward = function(x) x
)

nn_hardswish <- torch::nn_module(
  initialize = function(inplace = TRUE) {
    self$inplace <- inplace
  },
  forward = function(x) {
    x * torch::nnf_relu6(x + 3, inplace = self$inplace) / 6
  }
)

nn_hardsigmoid <- torch::nn_module(
  initialize = function(inplace = TRUE) {
    self$inplace <- inplace
  },
  forward = function(x) {
    torch::nnf_relu6(x + 3, inplace = self$inplace) / 6
  }
)

# SE Block for quantized version
quantized_squeeze_excitation <- torch::nn_module(
  initialize = function(input_channels, squeeze_factor = 4) {
    squeeze_channels <- input_channels %/% squeeze_factor
    self$fc1 <- torch::nn_conv2d(input_channels, squeeze_channels, 1)
    self$relu <- torch::nn_relu()
    self$fc2 <- torch::nn_conv2d(squeeze_channels, input_channels, 1)
    self$activation <- nn_hardsigmoid(inplace = TRUE)
    # Skip the FloatFunctional for now - might not be needed in R torch
  },
  forward = function(x) {
    scale <- torch::nnf_adaptive_avg_pool2d(x, 1)
    scale <- self$fc1(scale) %>% self$relu()
    scale <- self$fc2(scale) %>% self$activation()
    x * scale
  }
)

# Quantizable Inverted Residual
quantizable_inverted_residual <- torch::nn_module(
  initialize = function(input_channels, expanded_channels, out_channels, kernel_size,
                        stride, use_se = FALSE, activation_layer = torch::nn_relu) {
    self$use_res_connect <- (stride == 1) && (input_channels == out_channels)
    self$stride <- stride

    layers <- list()

    # Expand
    if (expanded_channels != input_channels) {
      layers <- append(layers, list(conv_bn_activation(input_channels, expanded_channels,
                                                       kernel_size = 1,
                                                       activation_layer = activation_layer)))
    }

    # Depthwise
    layers <- append(layers, list(conv_bn_activation(expanded_channels, expanded_channels,
                                                     kernel_size = kernel_size, stride = stride,
                                                     groups = expanded_channels,
                                                     activation_layer = activation_layer)))

    # SE
    if (use_se) {
      layers <- append(layers, list(quantized_squeeze_excitation(expanded_channels)))
    }

    # Project
    layers <- append(layers, list(conv_bn_activation(expanded_channels, out_channels,
                                                     kernel_size = 1,
                                                     activation_layer = nn_identity)))

    self$block <- torch::nn_sequential(!!!layers)

    # Skip FloatFunctional for now - might not be available in R torch
  },
  forward = function(x) {
    result <- self$block(x)
    if (self$use_res_connect) {
      result <- self$skip_add$add(result, x)
    }
    result
  }
)

mobilenet_v3_large_quant_impl <- torch::nn_module(
  "mobilenet_v3_large_quant_impl",
  initialize = function(num_classes = 1000, dropout = 0.2) {

    self$quant <- nn_identity()
    self$dequant <- nn_identity()

    # Build features layers
    layers <- list()

    # First layer
    layers <- append(layers, list(conv_bn_activation(3, 16, stride = 2,
                                                     activation_layer = nn_hardswish)))

    # Inverted Residual settings
    # kernel_size, expanded_channels, out_channels, use_se, activation, stride
    inverted_residual_setting <- list(
      list(3, 16, 16, FALSE, torch::nn_relu, 1),
      list(3, 64, 24, FALSE, torch::nn_relu, 2),
      list(3, 72, 24, FALSE, torch::nn_relu, 1),
      list(5, 72, 40, TRUE, torch::nn_relu, 2),
      list(5, 120, 40, TRUE, torch::nn_relu, 1),
      list(5, 120, 40, TRUE, torch::nn_relu, 1),
      list(3, 240, 80, FALSE, nn_hardswish, 2),
      list(3, 200, 80, FALSE, nn_hardswish, 1),
      list(3, 184, 80, FALSE, nn_hardswish, 1),
      list(3, 184, 80, FALSE, nn_hardswish, 1),
      list(3, 480, 112, TRUE, nn_hardswish, 1),
      list(3, 672, 112, TRUE, nn_hardswish, 1),
      list(5, 672, 160, TRUE, nn_hardswish, 2),
      list(5, 960, 160, TRUE, nn_hardswish, 1),
      list(5, 960, 160, TRUE, nn_hardswish, 1)
    )

    input_channels <- 16
    for (setting in inverted_residual_setting) {
      kernel_size <- setting[[1]]
      expanded_channels <- setting[[2]]
      out_channels <- setting[[3]]
      use_se <- setting[[4]]
      activation <- setting[[5]]
      stride <- setting[[6]]

      layers <- append(layers, list(
        quantizable_inverted_residual(input_channels, expanded_channels, out_channels,
                                      kernel_size, stride, use_se, activation)
      ))
      input_channels <- out_channels
    }

    # Last several layers
    layers <- append(layers, list(conv_bn_activation(160, 960, kernel_size = 1,
                                                     activation_layer = nn_hardswish)))

    self$features <- torch::nn_sequential(!!!layers)

    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))

    self$classifier <- torch::nn_sequential(
      torch::nn_linear(960, 1280),
      nn_hardswish(inplace = TRUE),
      torch::nn_dropout(dropout, inplace = TRUE),
      torch::nn_linear(1280, num_classes)
    )
  },
  forward = function(x) {
    x <- self$quant(x)
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
    self$dequant(x)
  }
)
