#' MobileNetV2 implementation
#'
#' Constructs a MobileNetV2 architecture from
#' [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381).
#'
#' @inheritParams model_resnet18
#' @param ... Other parameters passed to the model implementation.
#'
#' @family classification_model
#' @export
model_mobilenet_v2 <- function(pretrained = FALSE, progress = TRUE, ...) {
  # resources
  r <- c("https://torch-cdn.mlverse.org/models/vision/v2/models/mobilenet_v2.pth", "06af6062e42ad3c80e430219a6560ca0", "~13 MB")
  model <- mobilenet_v2(...)

  if (pretrained) {
    cli_inform("Model weights for {.cls {class(model)[1]}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}


mobilenet_v2 <- torch::nn_module(
  "mobilenet_v2",
  initialize = function(
    num_classes = 1000,
    width_mult = 1.0,
    inverted_residual_setting = NULL,
    round_nearest = 8,
    block = NULL,
    norm_layer = NULL
  ) {

    if (is.null(block))
      block <- inverted_residual

    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    input_channel <- 32
    last_channel <- 1280

    if (is.null(inverted_residual_setting)) {
      inverted_residual_setting <- list(
        # t, c, n, s
        c(1, 16, 1, 1),
        c(6, 24, 2, 2),
        c(6, 32, 3, 2),
        c(6, 64, 4, 2),
        c(6, 96, 3, 1),
        c(6, 160, 3, 2),
        c(6, 320, 1, 1)
      )
    }

    # only check the first element, assuming user knows t,c,n,s are required
    if (length(inverted_residual_setting) == 0 || length(inverted_residual_setting[[1]]) != 4)
      cli_abort("{.var inverted_residual_setting} should be non-empty or a 4-element list, got {.val {inverted_residual_setting}}")


    # building first layer
    input_channel <- mobilenetv2.make_divisible(input_channel * width_mult, round_nearest)
    self$last_channel <- mobilenetv2.make_divisible(last_channel * max(1.0, width_mult), round_nearest)

    features <- list(conv_bn_activation(3, input_channel, stride = 2, norm_layer = norm_layer))

    # building inverted residual blocks
    for (i in inverted_residual_setting) {
      names(i) <- c("t", "c", "n", "s")
      i <- as.list(i)
      output_channel <- mobilenetv2.make_divisible(i$c * width_mult, round_nearest)
      for (k in 0:(i$n -1)) {
        stride <-  if (k == 0) i$s else 1
        features[[length(features) + 1]] <- block(
          input_channel,
          output_channel,
          stride,
          expand_ratio=i$t,
          norm_layer=norm_layer
        )

        input_channel <- output_channel
      }
    }

    # building last several layers
    features[[length(features) + 1]] <- conv_bn_activation(
      input_channel,
      self$last_channel,
      kernel_size=1,
      norm_layer=norm_layer
    )

    # make it nn.Sequential
    self$features = torch::nn_sequential(!!!features)

    # building classifier
    self$classifier = torch::nn_sequential(
      torch::nn_dropout(0.2),
      torch::nn_linear(self$last_channel, num_classes)
    )

    # weight initialization
    for (m in self$modules) {
      if (inherits(m, "nn_conv2d")) {
        torch::nn_init_kaiming_normal_(m$weight, mode='fan_out')
        if (!is.null(m$bias)) {
          torch::nn_init_zeros_(m$bias)
        }
      } else if (inherits(m, c("nn_batch_norm2d", "nn_group_norm"))) {
        torch::nn_init_ones_(m$weight)
        torch::nn_init_zeros_(m$bias)
      } else if (inherits(m, c("nn_linear"))) {
        torch::nn_init_normal_(m$weight, 0, 0.01)
        torch::nn_init_zeros_(m$bias)
      }
    }

  },
  forward = function(x) {
    x <- self$features(x)
    # Cannot use "squeeze" as batch-size can be 1
    x <- torch::nnf_adaptive_avg_pool2d(x, c(1, 1))
    x <- torch::torch_flatten(x, start_dim = 2L)
    x <- self$classifier(x)
    x
  }
)

inverted_residual <- torch::nn_module(
  "inverted_residual",
  initialize = function(inp, oup, stride, expand_ratio, norm_layer = NULL) {

    self$stride <- stride

    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    hidden_dim <- as.integer(round(inp * expand_ratio))
    self$use_res_connect = self$stride == 1 && inp == oup

    layers <- list()

    if (expand_ratio != 1) {
      layers[[length(layers) + 1]] <- conv_bn_activation(
        inp,
        hidden_dim,
        kernel_size=1,
        norm_layer=norm_layer
      )
    }

    layers <- append(layers, list(
      conv_bn_activation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
      # pw-linear
      torch::nn_conv2d(hidden_dim, oup, 1, 1, 0, bias=FALSE),
      norm_layer(oup)
    ))

    self$conv <- torch::nn_sequential(!!!layers)
    self$out_channels <- oup
    self$.is_cn <- stride > 1
  },
  forward = function(x) {
    if (self$use_res_connect)
      x + self$conv(x)
    else
      self$conv(x)
  }
)

sequential <- torch::nn_module(
  classname = "nn_sequential",
  initialize = function(...) {
    modules <- rlang::list2(...)
    for (i in seq_along(modules)) {
      self$add_module(name = i - 1, module = modules[[i]])
    }
  }, forward = function(input) {
    for (module in private$modules_) {
      input <- module(input)
    }
    input
  }
)

conv_bn_activation <- torch::nn_module(
  "conv_bn_activation",
  inherit = sequential,
  initialize = function(
    in_planes,
    out_planes,
    kernel_size = 3,
    stride = 1,
    groups = 1,
    norm_layer = NULL,
    activation_layer = NULL,
    dilation = 1
  ) {

    padding <- (kernel_size - 1) %/% 2 * dilation

    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    if (is.null(activation_layer))
      activation_layer <- torch::nn_relu6

    super$initialize(
      torch::nn_conv2d(
        in_planes, out_planes, kernel_size, stride, padding,
        dilation=dilation,
        groups=groups,
        bias=FALSE
      ),
      norm_layer(out_planes),
      activation_layer(inplace=TRUE)
    )

    self$out_channels <- out_planes
  }
)

#' This function is taken from the original tf repo.
#' It ensures that all layers have a channel number that is divisible by 8
#' It can be seen here:
#' https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#' @noRd
mobilenetv2.make_divisible <- function(v, divisor, min_value = NULL) {

  if (is.null(min_value))
    min_value <- divisor

  new_v <- max(min_value, as.integer(v + divisor/2) %/% divisor * divisor)

  # Make sure that round down does not go down by more than 10%.
  if (new_v < 0.9 * v)
    new_v <- new_v + divisor

  new_v
}
