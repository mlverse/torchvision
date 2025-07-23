VGG <- torch::nn_module(
  "VGG",
  initialize = function(features, num_classes=1000, init_weights=TRUE) {
    self$features <- features
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(7,7))
    self$classifier <- torch::nn_sequential(
      torch::nn_linear(512 * 7 * 7, 4096),
      torch::nn_relu(TRUE),
      torch::nn_dropout(),
      torch::nn_linear(4096, 4096),
      torch::nn_relu(TRUE),
      torch::nn_dropout(),
      torch::nn_linear(4096, num_classes),
    )

    if (init_weights)
      self$.initialize_weights()

  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x,start_dim = 2)
    x <- self$classifier(x)
    x
  },
  .initialize_weights = function() {

    for (m in self$modules) {

      if (inherits(m, "nn_conv2d")) {
        torch::nn_init_kaiming_normal_(m$weight, mode = "fan_out", nonlinearity="relu")
        if (!is.null(m$bias))
          torch::nn_init_constant_(m$bias, 0)
      } else if (inherits(m, "nn_batch_norm2d")) {
        torch::nn_init_constant_(m$weight, 1)
        torch::nn_init_constant_(m$bias, 0)
      } else if (inherits(m, "nn_linear")) {
        torch::nn_init_normal_(m$weight, 0, 0.01)
        torch::nn_init_constant_(m$bias, 0)
      }

    }

  }
)

vgg_make_layers <- function(cfg, batch_norm) {
  layers <- list()
  in_channels <- 3
  for (v in cfg) {

    if (v == "M") {

      layers[[length(layers) + 1]] <- torch::nn_max_pool2d(
        kernel_size = 2,
        stride = 2
      )

    } else {

      v <- as.integer(v)
      layers[[length(layers) + 1]] <- torch::nn_conv2d(
        in_channels = in_channels, out_channels = v,
        kernel_size = 3, padding = 1
      )

      if (batch_norm)
        layers[[length(layers) + 1]] <- torch::nn_batch_norm2d(num_features = v)

      layers[[length(layers) + 1]] <- torch::nn_relu(inplace = TRUE)
      in_channels <- v

    }

  }
  torch::nn_sequential(!!!layers)
}

vgg_cfgs <- list(
  "A" = list(64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
  "B" = list(64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
  "D" = list(64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"),
  "E" = list(64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M")
)

vgg_model_urls <- list(
  "vgg11"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg11.pth", "3afa0c0e9eecb80c0f4ebc4303e19f68", "~510 MB"),
  "vgg13"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg13.pth", "1733a72a8b22e37a57fb9cb9d4db3a74", "~510 MB"),
  "vgg16"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg16.pth", "2329d5702a1d78713179f6127a280fcd", "~530 MB"),
  "vgg19"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg19.pth", "6acea77a847bb1549171504b872efb96", "~575 MB"),
  "vgg11_bn"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg11_bn.pth", "76ca955d0aa3e61152bdd726949d8fed", "~510 MB"),
  "vgg13_bn"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg13_bn.pth", "612e11f29432ea79ae94a66b93db9109", "~510 MB"),
  "vgg16_bn"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg16_bn.pth", "5ee7dc31de40677452eec2fdffce7e17", "~530 MB"),
  "vgg19_bn"= c("https://torch-cdn.mlverse.org/models/vision/v2/models/vgg19_bn.pth", "7112c457070af40fdddda8e75f9b727b", "~575 MB")
)

vgg <- function(arch, cfg, batch_norm, pretrained, progress, ...) {

  args <- rlang::list2(...)

  if (pretrained)
    args$init_weights <- FALSE

  layers <- vgg_make_layers(cfg = vgg_cfgs[[cfg]], batch_norm = batch_norm)
  args <- append(args, list(features = layers))
  model <- do.call(VGG, args)

  if (pretrained) {
    r <- vgg_model_urls[[arch]]
    cli_inform("Model weights for {.cls {arch}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)

  }

  model
}

#' VGG implementation
#'
#'
#' VGG models implementations based on
#' [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)
#'
#' @param pretrained (bool): If TRUE, returns a model pre-trained on ImageNet
#' @param progress (bool): If TRUE, displays a progress bar of the download
#'   to stderr
#' @param ... other parameters passed to the VGG model implementation.
#'
#' @name model_vgg
#' @rdname model_vgg
NULL

#' @describeIn model_vgg VGG 11-layer model (configuration "A")
#'
#' @family models
#' @export
model_vgg11 <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg('vgg11', 'A', FALSE, pretrained, progress, ...)
}

#' @describeIn model_vgg VGG 11-layer model (configuration "A") with batch normalization
#' @export
model_vgg11_bn <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg("vgg11_bn", "A", TRUE, pretrained, progress, ...)
}

#' @describeIn model_vgg VGG 13-layer model (configuration "B")
#' @export
model_vgg13 <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg('vgg13', 'B', FALSE, pretrained, progress, ...)
}

#' @describeIn model_vgg VGG 13-layer model (configuration "B") with batch normalization
#' @export
model_vgg13_bn <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg("vgg13_bn", "B", TRUE, pretrained, progress, ...)
}

#' @describeIn model_vgg VGG 13-layer model (configuration "D")
#' @export
model_vgg16 <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg('vgg16', 'D', FALSE, pretrained, progress, ...)
}

#' @describeIn model_vgg VGG 13-layer model (configuration "D") with batch normalization
#' @export
model_vgg16_bn <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg("vgg16_bn", "D", TRUE, pretrained, progress, ...)
}

#' @describeIn model_vgg VGG 19-layer model (configuration "E")
#' @export
model_vgg19 <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg('vgg19', 'E', FALSE, pretrained, progress, ...)
}

#' @describeIn model_vgg VGG 19-layer model (configuration "E") with batch normalization
#' @export
model_vgg19_bn <- function(pretrained = FALSE, progress = TRUE, ...) {
  vgg("vgg19_bn", "E", TRUE, pretrained, progress, ...)
}






