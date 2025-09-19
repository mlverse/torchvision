#' MobileNetV3 Large model architecture.
#'
#' This implementation mimics `torchvision::mobilenet_v3_large`.
#'
#' @inheritParams model_mobilenet_v2
#' @param ... Other parameters passed to the model implementation.
#'
#' @family classification_model
#' @rdname model_mobilenet_v3
#' @name model_mobilenet_v3
#' @export
model_mobilenet_v3_large <- function(pretrained = FALSE, progress = TRUE, ...) {
  # resources
  model <- mobilenet_v3_large_impl(...)

  if (pretrained) {
    r <- c("https://torch-cdn.mlverse.org/models/vision/v2/models/mobilenet_v3_large.pth",
           "71625955bc3be9516032a6d5bab49199", "~21 MB")
    cli_inform("Model weights for {.cls {class(model)[1]}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "mobilenet")
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

#' Quantized MobileNetV3 Large model architecture.
#'
#' This function mirrors `torchvision::quantization::mobilenet_v3_large` and
#' loads quantized weights when `pretrained` is `TRUE`.
#'
#' @inheritParams model_mobilenet_v3_large
#' @family classification_model
#' @rdname model_mobilenet
#' @name model_mobilenet
#' @export
model_mobilenet_v3_large_quantized <- function(pretrained = FALSE, progress = TRUE, ...) {
  # resources
  r <- c("https://torch-cdn.mlverse.org/models/vision/v2/models/mobilenet_v3_large_quantized.pth",
         "06af6062e42ad3c80e430219a6560ca0", "~13 MB")
  model <- mobilenet_v3_large_quant_impl(...)

  if (pretrained) {
    cli_inform("Model weights for {.cls {class(model)[1]}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1], prefix = "mobilenet")
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}


# ORIGINAL WORKING REGULAR MODEL COMPONENTS
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

# ORIGINAL WORKING REGULAR MODEL IMPLEMENTATION
mobilenet_v3_large_impl <- torch::nn_module(
  "mobilenet_v3_large_impl",
  initialize = function(num_classes = 1000, dropout = 0.2, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

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


###########################################################################################
# QUANTIZED MODEL COMPONENTS
nn_identity <- torch::nn_module(
  "nn_identity",
  initialize = function() {},
  forward = function(x) x
)
# Fixed Quantized Conv-BN-Activation block to match .pth structure
conv_bn_act_quantized <- torch::nn_module(
  initialize = function(in_channels, out_channels, kernel_size = 3, stride = 1,
                        groups = 1, norm_layer = NULL, activation_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d
    if (is.null(activation_layer))
      activation_layer <- nn_hardswish

    padding <- (kernel_size - 1) %/% 2

    # Create nested structure to match .pth: features.X.0.0.weight, features.X.0.bn.weight, etc.
    self$`0` <- torch::nn_module()
    self$`0`$`0` <- torch::nn_conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                     groups = groups, bias = FALSE)
    self$`0`$bn <- norm_layer(out_channels)
    self$`0`$`2` <- activation_layer(inplace = TRUE)

    # Quantization stubs
    self$`0`$activation_post_process <- nn_identity()

    self$out_channels <- out_channels
  },
  forward = function(x) {
    x <- self$`0`$`0`(x)
    x <- self$`0`$bn(x)
    x <- self$`0`$`2`(x)
    x
  }
)

# Updated Quantized Inverted Residual Block to match structure
inverted_residual_quantized <- torch::nn_module(
  initialize = function(in_channels, expand_channels, out_channels, kernel_size,
                        stride, use_se = FALSE, activation_layer = nn_hardswish,
                        norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    self$use_res_connect <- stride == 1 && in_channels == out_channels

    # Create block structure to match expected .pth keys
    block_idx <- 0

    # Expansion layer (if needed)
    if (expand_channels != in_channels) {
      self[[as.character(block_idx)]] <- conv_bn_act_quantized(
        in_channels, expand_channels, kernel_size = 1,
        norm_layer = norm_layer, activation_layer = activation_layer
      )
      block_idx <- block_idx + 1
    }

    # Depthwise layer
    self[[as.character(block_idx)]] <- conv_bn_act_quantized(
      expand_channels, expand_channels, kernel_size = kernel_size, stride = stride,
      groups = expand_channels, norm_layer = norm_layer,
      activation_layer = activation_layer
    )
    block_idx <- block_idx + 1

    # SE block (if used)
    if (use_se) {
      make_divisible <- function(v, divisor = 8, min_value = NULL) {
        if (is.null(min_value)) min_value <- divisor
        new_v <- max(min_value, as.integer(v + divisor / 2) %/% divisor * divisor)
        if (new_v < 0.9 * v) new_v <- new_v + divisor
        new_v
      }

      squeeze_channels <- make_divisible(expand_channels / 4)
      self[[as.character(block_idx)]] <- se_block_quantized(expand_channels, squeeze_channels)
      block_idx <- block_idx + 1
    }

    # Project layer (no activation) - direct conv and bn
    self[[as.character(block_idx)]] <- torch::nn_module()
    self[[as.character(block_idx)]]$`0` <- torch::nn_conv2d(expand_channels, out_channels, 1, bias = FALSE)
    self[[as.character(block_idx)]]$bn <- norm_layer(out_channels)

    # Store the number of blocks for forward pass
    self$num_blocks <- block_idx + 1

    # Add quantization stubs
    self$activation_post_process <- nn_identity()
  },
  forward = function(x) {
    out <- x
    for (i in 0:(self$num_blocks - 1)) {
      layer <- self[[as.character(i)]]
      if (!is.null(layer)) {
        out <- layer(out)
      }
    }

    if (self$use_res_connect)
      out <- out + x
    out
  }
)

# Quantized SE Block
se_block_quantized <- torch::nn_module(
  initialize = function(in_channels, squeeze_channels) {
    self$avgpool <- torch::nn_adaptive_avg_pool2d(output_size = 1)
    self$fc1 <- torch::nn_conv2d(in_channels, squeeze_channels, kernel_size = 1)
    self$activation <- torch::nn_relu(inplace = TRUE)
    self$fc2 <- torch::nn_conv2d(squeeze_channels, in_channels, kernel_size = 1)
    self$scale_activation <- nn_hardsigmoid(inplace = TRUE)

    # Quantization stubs
    self$activation_post_process <- nn_identity()
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

# QUANTIZED MODEL IMPLEMENTATION
mobilenet_v3_large_quant_impl <- torch::nn_module(
  "mobilenet_v3_large_quant_impl",
  initialize = function(num_classes = 1000, dropout = 0.2, norm_layer = NULL) {
    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    # Quantization stubs
    self$quant <- nn_identity()
    self$dequant <- nn_identity()

    activation <- nn_hardswish

    # Create features as a module list to match the quantized structure
    self$features <- torch::nn_module_list()

    # First layer - create the nested structure manually to match features.0.0.bn.weight
    self$features[["0"]] <- torch::nn_module()
    self$features[["0"]]$`0` <- torch::nn_module()
    self$features[["0"]]$`0`$`0` <- torch::nn_conv2d(3, 16, 3, stride = 2, padding = 1, bias = FALSE)
    self$features[["0"]]$`0`$bn <- norm_layer(16)
    self$features[["0"]]$`0`$`2` <- activation(inplace = TRUE)
    self$features[["0"]]$`0`$activation_post_process <- nn_identity()

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

    layer_idx <- 1
    for (cfg in cfgs) {
      k <- cfg[[1]]; exp <- cfg[[2]]; c <- cfg[[3]]; se <- cfg[[4]];
      nl <- cfg[[5]]; s <- cfg[[6]]

      self$features[[as.character(layer_idx)]] <- inverted_residual_quantized(
        input_channel, exp, c, kernel_size = k, stride = s, use_se = se,
        activation_layer = nl, norm_layer = norm_layer
      )
      input_channel <- c
      layer_idx <- layer_idx + 1
    }

    # Final conv layer
    self$features[[as.character(layer_idx)]] <- conv_bn_act_quantized(
      input_channel, 960, kernel_size = 1,
      norm_layer = norm_layer, activation_layer = activation
    )

    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1,1))

    # Classifier with quantization stubs
    self$classifier <- torch::nn_module()
    self$classifier$`0` <- torch::nn_linear(960, 1280)
    self$classifier$`1` <- activation(inplace = TRUE)
    self$classifier$`2` <- torch::nn_dropout(dropout)
    self$classifier$`3` <- torch::nn_linear(1280, num_classes)
  },
  forward = function(x) {
    x <- self$quant(x)

    # Forward through first layer manually
    x <- self$features[["0"]]$`0`$`0`(x)
    x <- self$features[["0"]]$`0`$bn(x)
    x <- self$features[["0"]]$`0`$`2`(x)

    # Forward through remaining features
    for (i in 1:(length(self$features) - 1)) {
      x <- self$features[[as.character(i)]](x)
    }

    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)

    # Forward through classifier
    x <- self$classifier$`0`(x)
    x <- self$classifier$`1`(x)
    x <- self$classifier$`2`(x)
    x <- self$classifier$`3`(x)

    self$dequant(x)
  }
)
