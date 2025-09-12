#' MobileNetV3 Model
#'
#' MobileNetV3 is a state-of-the-art lightweight convolutional neural network architecture
#' designed for mobile and embedded vision applications. This implementation follows the
#' design and optimizations presented in the original paper:[MobileNetV3: Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
#'
#' The model includes two variants:
#' - `model_mobilenet_v3_large()`
#' - `model_mobilenet_v3_small()`
#'
#' Both variants utilize efficient blocks such as inverted residuals, squeeze-and-excitation (SE) modules,
#' and hard-swish activations for improved accuracy and efficiency.
#'
#' ## Model Summary and Performance for pretrained weights
#' ```
#' | Model                  | Top-1 Acc | Top-5 Acc | Params  | GFLOPS | File Size | Notes                               |
#' |------------------------|-----------|-----------|---------|--------|-----------|-------------------------------------|
#' | MobileNetV3 Large      | 74.04%    | 91.34%    | 5.48M   | 0.22   | 21.1 MB   | Trained from scratch, simple recipe |
#' | MobileNetV3 Small      | 67.67%    | 87.40%    | 2.54M   | 0.06   | 9.8 MB    | Improved recipe over original paper |
#' ```
#'
#' @examples
#' \dontrun{
#' # 1. Download sample image (dog)
#' norm_mean <- c(0.485, 0.456, 0.406) # ImageNet normalization constants, see
#' # https://pytorch.org/vision/stable/models.html
#' norm_std  <- c(0.229, 0.224, 0.225)
#' img_url <- "https://en.wikipedia.org/wiki/Special:FilePath/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(img_url)
#'
#' # 2. Convert to tensor (RGB only), resize and normalize
#' input <- img %>%
#'  transform_to_tensor() %>%
#'  transform_resize(c(224, 224)) %>%
#'  transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)
#'
#' # 3. Load pretrained models
#' model_small <- model_mobilenet_v3_small(pretrained = TRUE)
#' model_small$eval()
#'
#' # 4. Forward pass
#' output_s <- model_small(batch)
#'
#' # 5. Top-5 printing helper
#' topk <- output_s$topk(k = 5, dim = 2)
#' indices <- as.integer(topk[[2]][1, ])
#' scores <- as.numeric(topk[[1]][1, ])
#'
#' # 6. Show Top-5 predictions
#' glue::glue("{seq_along(indices)}. {imagenet_label(indices)} ({round(scores, 2)}%)")
#'
#' # 7. Same with large model
#' model_large <- model_mobilenet_v3_large(pretrained = TRUE)
#' model_large$eval()
#' output_l <- model_large(input)
#' topk <- output_l$topk(k = 5, dim = 2)
#' indices <- as.integer(topk[[2]][1, ])
#' scores <- as.numeric(topk[[1]][1, ])
#' glue::glue("{seq_along(indices)}. {imagenet_label(indices)} ({round(scores, 2)}%)")
#' }
#'
#' @importFrom torch nn_module nn_conv2d nn_batch_norm2d nn_relu nn_hardswish nn_hardsigmoid nn_identity nn_sequential
#' @importFrom torch nn_adaptive_avg_pool2d nn_linear nn_dropout torch_clamp torch_flatten load_state_dict
#'
#' @inheritParams model_mobilenet_v2
#' @param num_classes number of output classes (default: 1000).
#' @param width_mult width multiplier for model scaling (default: 1.0).
#'
#' @family classification_model
#' @rdname model_mobilenet_v3
#' @name model_mobilenet_v3
NULL

make_divisible <- function(v, divisor = 8, min_value = NULL) {
  if (is.null(min_value)) min_value <- divisor
  new_v <- max(min_value, floor((v + divisor / 2) / divisor) * divisor)
  if (new_v < 0.9 * v) new_v <- new_v + divisor
  new_v
}

HardSwish <- nn_module(
  "HardSwish",
  forward = function(x) {
    x * torch_clamp(x + 3, min = 0, max = 6) / 6
  }
)

HardSigmoid <- nn_module(
  "HardSigmoid",
  forward = function(x) {
    torch_clamp(x + 3, min = 0, max = 6) / 6
  }
)

Conv2dNormActivation <- nn_module(
  "Conv2dNormActivation",
  initialize = function(in_channels, out_channels, kernel_size, stride = 1, groups = 1,
                        norm_layer = nn_batch_norm2d, activation_layer = nn_relu,
                        dilation = 1) {
    padding <- floor(((kernel_size - 1) / 2) * dilation)
    self$conv <- nn_conv2d(
      in_channels, out_channels, kernel_size,
      stride = stride, padding = padding, groups = groups,
      dilation = dilation, bias = FALSE
    )
    self$bn <- norm_layer(out_channels)
    if (is.character(activation_layer) && activation_layer == "hardswish") {
      self$activation <- HardSwish()
    } else if (identical(activation_layer, nn_hardswish)) {
      self$activation <- HardSwish()
    } else if (identical(activation_layer, nn_hardsigmoid)) {
      self$activation <- HardSigmoid()
    } else {
      self$activation <- activation_layer()
    }
  },
  forward = function(x) {
    x %>% self$conv() %>% self$bn() %>% self$activation()
  }
)

SELayer <- nn_module(
  "SELayer",
  initialize = function(input_channels, squeeze_channels) {
    self$avg_pool <- nn_adaptive_avg_pool2d(1)
    self$fc1 <- nn_conv2d(input_channels, squeeze_channels, 1)
    self$relu <- nn_relu()
    self$fc2 <- nn_conv2d(squeeze_channels, input_channels, 1)
    self$hsigmoid <- nn_hardsigmoid()
  },
  forward = function(x) {
    scale <- x %>%
      self$avg_pool() %>%
      self$fc1() %>%
      self$relu() %>%
      self$fc2() %>%
      self$hsigmoid()
    x * scale
  }
)

InvertedResidual <- nn_module(
  "InvertedResidual",
  initialize = function(input_channels, expanded_channels, out_channels, kernel, stride,
                        use_se, use_hs, dilation = 1, norm_layer = nn_batch_norm2d) {

    if (!(stride %in% c(1, 2))) stop("illegal stride value")
    self$use_res_connect <- (stride == 1) && (input_channels == out_channels)

    layers <- list()
    activation_layer <- if (use_hs) nn_hardswish else nn_relu

    if (expanded_channels != input_channels) {
      layers <- c(layers, list(
        Conv2dNormActivation(
          input_channels, expanded_channels, kernel_size = 1,
          norm_layer = norm_layer, activation_layer = activation_layer
        )
      ))
    }

    stride_ <- if (dilation > 1) 1 else stride
    layers <- c(layers, list(
      Conv2dNormActivation(
        expanded_channels, expanded_channels, kernel_size = kernel,
        stride = stride_, groups = expanded_channels, dilation = dilation,
        norm_layer = norm_layer, activation_layer = activation_layer
      )
    ))

    if (use_se) {
      squeeze_channels <- make_divisible(expanded_channels / 4)
      layers <- c(layers, list(SELayer(expanded_channels, squeeze_channels)))
    }

    layers <- c(layers, list(
      Conv2dNormActivation(
        expanded_channels, out_channels, kernel_size = 1,
        norm_layer = norm_layer, activation_layer = nn_identity
      )
    ))

    self$block <- nn_sequential(!!!layers)
    self$out_channels <- out_channels
  },
  forward = function(x) {
    out <- self$block(x)
    if (self$use_res_connect) {
      out + x
    } else {
      out
    }
  }
)

InvertedResidualConfig <- function(input_c, kernel, expanded_c, out_c, use_se, use_hs, stride, dilation = 1, width_mult = 1.0) {
  input_channels <- make_divisible(input_c * width_mult)
  expanded_channels <- make_divisible(expanded_c * width_mult)
  out_channels <- make_divisible(out_c * width_mult)

  list(
    input_channels = input_channels,
    kernel = kernel,
    expanded_channels = expanded_channels,
    out_channels = out_channels,
    use_se = use_se,
    use_hs = use_hs,
    stride = stride,
    dilation = dilation
  )
}

MobileNetV3 <- nn_module(
  "MobileNetV3",
  initialize = function(inverted_residual_setting, last_channel, num_classes = 1000,
                        dropout = 0.2, norm_layer = nn_batch_norm2d) {

    layers <- list()

    firstconv_out <- inverted_residual_setting[[1]]$input_channels
    layers <- c(layers, list(
      Conv2dNormActivation(
        3, firstconv_out, kernel_size = 3, stride = 2,
        norm_layer = norm_layer, activation_layer = nn_hardswish
      )
    ))

    for (conf in inverted_residual_setting) {
      layers <- c(layers, list(
        InvertedResidual(
          input_channels = conf$input_channels,
          expanded_channels = conf$expanded_channels,
          out_channels = conf$out_channels,
          kernel = conf$kernel,
          stride = conf$stride,
          use_se = conf$use_se,
          use_hs = conf$use_hs,
          dilation = conf$dilation,
          norm_layer = norm_layer
        )
      ))
    }

    lastconv_in <- inverted_residual_setting[[length(inverted_residual_setting)]]$out_channels
    lastconv_out <- 6 * lastconv_in
    layers <- c(layers, list(
      Conv2dNormActivation(
        lastconv_in, lastconv_out, kernel_size = 1,
        norm_layer = norm_layer, activation_layer = nn_hardswish
      )
    ))

    self$features <- nn_sequential(!!!layers)
    self$avgpool <- nn_adaptive_avg_pool2d(1)

    self$classifier <- nn_sequential(
      nn_linear(lastconv_out, last_channel),
      HardSwish(),
      nn_dropout(p = dropout, inplace = TRUE),
      nn_linear(last_channel, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch_flatten(x, start_dim = 1)
    x <- self$classifier(x)
    x
  }
)

mobilenet_v3_large_config <- function(width_mult = 1.0, reduced_tail = FALSE, dilated = FALSE) {
  reduce_divider <- ifelse(reduced_tail, 2, 1)
  dilation <- ifelse(dilated, 2, 1)
  b <- function(in_c, k, exp_c, out_c, se, act, s, d = 1) {
    InvertedResidualConfig(in_c, k, exp_c, out_c, se, act == "HS", s, d, width_mult)
  }

  list(
    b(16, 3, 16, 16, FALSE, "RE", 1, 1),
    b(16, 3, 64, 24, FALSE, "RE", 2, 1),
    b(24, 3, 72, 24, FALSE, "RE", 1, 1),
    b(24, 5, 72, 40, TRUE, "RE", 2, 1),
    b(40, 5, 120, 40, TRUE, "RE", 1, 1),
    b(40, 5, 120, 40, TRUE, "RE", 1, 1),
    b(40, 3, 240, 80, FALSE, "HS", 2, 1),
    b(80, 3, 200, 80, FALSE, "HS", 1, 1),
    b(80, 3, 184, 80, FALSE, "HS", 1, 1),
    b(80, 3, 184, 80, FALSE, "HS", 1, 1),
    b(80, 3, 480, 112, TRUE, "HS", 1, 1),
    b(112, 3, 672, 112, TRUE, "HS", 1, 1),
    b(112, 5, 672, 160 / reduce_divider, TRUE, "HS", 2, dilation),
    b(160 / reduce_divider, 5, 960 / reduce_divider, 160 / reduce_divider, TRUE, "HS", 1, dilation),
    b(160 / reduce_divider, 5, 960 / reduce_divider, 160 / reduce_divider, TRUE, "HS", 1, dilation)
  )
}

mobilenet_v3_small_config <- function(width_mult = 1.0, reduced_tail = FALSE, dilated = FALSE) {
  reduce_divider <- ifelse(reduced_tail, 2, 1)
  dilation <- ifelse(dilated, 2, 1)
  b <- function(in_c, k, exp_c, out_c, se, act, s, d = 1) {
    InvertedResidualConfig(in_c, k, exp_c, out_c, se, act == "HS", s, d, width_mult)
  }

  list(
    b(16, 3, 16, 16, TRUE, "RE", 2, 1),
    b(16, 3, 72, 24, FALSE, "RE", 2, 1),
    b(24, 3, 88, 24, FALSE, "RE", 1, 1),
    b(24, 5, 96, 40, TRUE, "HS", 2, 1),
    b(40, 5, 240, 40, TRUE, "HS", 1, 1),
    b(40, 5, 240, 40, TRUE, "HS", 1, 1),
    b(40, 5, 120, 48, TRUE, "HS", 1, 1),
    b(48, 5, 144, 48, TRUE, "HS", 1, 1),
    b(48, 5, 288, 96 / reduce_divider, TRUE, "HS", 2, dilation),
    b(96 / reduce_divider, 5, 576 / reduce_divider, 96 / reduce_divider, TRUE, "HS", 1, dilation),
    b(96 / reduce_divider, 5, 576 / reduce_divider, 96 / reduce_divider, TRUE, "HS", 1, dilation)
  )
}

#' @describeIn model_mobilenet_v3 MobileNetV3 Large model with about 5.5 million parameters.
#' @export
model_mobilenet_v3_large <- function(
  pretrained = FALSE,
  progress = TRUE,
  num_classes = 1000,
  width_mult = 1.0
) {

  config <- mobilenet_v3_large_config(width_mult)
  last_channel <- make_divisible(1280 * width_mult)
  model <- MobileNetV3(config, last_channel, num_classes = num_classes)
  if (pretrained) {
    state_dict_path <- download_and_cache("https://torch-cdn.mlverse.org/models/vision/v2/models/mobilenet_v3_large.pth", prefix = "mobilenet_v3_large")
    state_dict <- load_state_dict(state_dict_path)
    new_names <- names(state_dict)

    new_names <- gsub("^features\\.([0-9]+)\\.0\\.", "features.\\1.conv.", new_names)
    new_names <- gsub("^features\\.([0-9]+)\\.1\\.", "features.\\1.bn.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+)\\.0\\.", "\\1.conv.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+)\\.1\\.", "\\1.bn.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+\\.fc1)\\.", "\\1.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+\\.fc2)\\.", "\\1.", new_names)

    names(state_dict) <- new_names
    model$load_state_dict(state_dict)
  }
  model$eval()
  model
}

#' @describeIn model_mobilenet_v3 MobileNetV3 Small model with about 2.5 million parameters.
#' @export
model_mobilenet_v3_small <- function(
  pretrained = FALSE,
  progress = TRUE,
  num_classes = 1000,
  width_mult = 1.0
) {

  config <- mobilenet_v3_small_config(width_mult)
  last_channel <- make_divisible(1024 * width_mult)
  model <- MobileNetV3(config, last_channel, num_classes = num_classes)
  if (pretrained) {
    state_dict_path <- download_and_cache("https://torch-cdn.mlverse.org/models/vision/v2/models/mobilenet_v3_small.pth", prefix = "mobilenet_v3_small")
    state_dict <- load_state_dict(state_dict_path)
    new_names <- names(state_dict)

    new_names <- gsub("^features\\.([0-9]+)\\.0\\.", "features.\\1.conv.", new_names)
    new_names <- gsub("^features\\.([0-9]+)\\.1\\.", "features.\\1.bn.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+)\\.0\\.", "\\1.conv.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+)\\.1\\.", "\\1.bn.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+\\.fc1)\\.", "\\1.", new_names)
    new_names <- gsub("(features\\.[0-9]+\\.block\\.[0-9]+\\.fc2)\\.", "\\1.", new_names)

    names(state_dict) <- new_names
    model$load_state_dict(state_dict)
  }
  model$eval()
  model
}
