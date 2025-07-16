#' DeepLabV3 implementation
#'
#' DeepLabV3 model family based on
#'   [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587).
#'
#' @param pretrained (bool): If TRUE, returns a model pre-trained on COCO.
#' @param progress (bool): If TRUE, displays a progress bar of the download to stderr.
#' @param num_classes (int): number of output classes of the model.
#' @param aux_loss (bool): If TRUE, include an auxiliary classifier.
#' @param ... other parameters passed to the ResNet backbone.
#'
#' @family models
#' @name model_deeplabv3
#' @rdname model_deeplabv3
NULL


deeplabv3_model_urls <- c(
  'deeplabv3_resnet50_coco' = 'https://torch-cdn.mlverse.org/models/vision/v1/models/deeplabv3_resnet50_coco.pth',
  'deeplabv3_resnet101_coco' = 'https://torch-cdn.mlverse.org/models/vision/v1/models/deeplabv3_resnet101_coco.pth'
)

ASPPConv <- torch::nn_module(
  "ASPPConv",
  initialize = function(in_channels, out_channels, dilation) {
    self$conv <- torch::nn_conv2d(in_channels, out_channels, kernel_size = 3,
                                  padding = dilation, dilation = dilation, bias = FALSE)
    self$bn <- torch::nn_batch_norm2d(out_channels)
    self$relu <- torch::nn_relu(inplace = TRUE)
  },
  forward = function(x) {
    x <- self$conv(x)
    x <- self$bn(x)
    x <- self$relu(x)
    x
  }
)

ASPPPooling <- torch::nn_module(
  "ASPPPooling",
  initialize = function(in_channels, out_channels) {
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
    self$conv <- torch::nn_conv2d(in_channels, out_channels, 1, bias = FALSE)
    self$bn <- torch::nn_batch_norm2d(out_channels)
    self$relu <- torch::nn_relu(inplace = TRUE)
  },
  forward = function(x) {
    size <- tail(x$shape, 2)
    x <- self$avgpool(x)
    x <- self$conv(x)
    x <- self$bn(x)
    x <- self$relu(x)
    torch::nnf_interpolate(x, size = size, mode = 'bilinear', align_corners = FALSE)
  }
)

ASPP <- torch::nn_module(
  "ASPP",
  initialize = function(in_channels, out_channels, atrous_rates) {
    convs <- list(
      torch::nn_sequential(
        torch::nn_conv2d(in_channels, out_channels, 1, bias = FALSE),
        torch::nn_batch_norm2d(out_channels),
        torch::nn_relu(inplace = TRUE)
      )
    )
    for (rate in atrous_rates) {
      convs[[length(convs) + 1]] <- ASPPConv(in_channels, out_channels, rate)
    }
    convs[[length(convs) + 1]] <- ASPPPooling(in_channels, out_channels)
    self$convs <- torch::nn_module_list(convs)
    self$project <- torch::nn_sequential(
      torch::nn_conv2d(length(convs) * out_channels, out_channels, 1, bias = FALSE),
      torch::nn_batch_norm2d(out_channels),
      torch::nn_relu(inplace = TRUE),
      torch::nn_dropout(0.5)
    )
  },
  forward = function(x) {
    res <- list()
    for (i in seq_along(self$convs)) {
      res[[i]] <- self$convs[[i]](x)
    }
    x <- torch::torch_cat(res, dim = 2)
    self$project(x)
  }
)

FCNHead <- torch::nn_module(
  "FCNHead",
  initialize = function(in_channels, channels, num_classes) {
    self$block <- torch::nn_sequential(
      torch::nn_conv2d(in_channels, channels, 3, padding = 1, bias = FALSE),
      torch::nn_batch_norm2d(channels),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(channels, num_classes, 1)
    )
  },
  forward = function(x) {
    self$block(x)
  }
)

DeepLabHead <- torch::nn_module(
  "DeepLabHead",
  initialize = function(in_channels, num_classes) {
    self$block <- torch::nn_sequential(
      ASPP(in_channels, 256, atrous_rates = c(12, 24, 36)),
      torch::nn_conv2d(256, 256, kernel_size = 3, padding = 1, bias = FALSE),
      torch::nn_batch_norm2d(256),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(256, num_classes, kernel_size = 1)
    )
  },
  forward = function(x) {
    self$block(x)
  }
)

ResNetBackbone <- torch::nn_module(
  "ResNetBackbone",
  initialize = function(resnet, aux = FALSE) {
    self$resnet <- resnet
    self$aux <- aux
  },
  forward = function(x) {
    result <- list()
    x <- self$resnet$conv1(x)
    x <- self$resnet$bn1(x)
    x <- self$resnet$relu(x)
    x <- self$resnet$maxpool(x)
    x <- self$resnet$layer1(x)
    x <- self$resnet$layer2(x)
    x <- self$resnet$layer3(x)
    if (self$aux)
      result$aux <- x
    x <- self$resnet$layer4(x)
    result$out <- x
    result
  }
)

DeepLabV3 <- torch::nn_module(
  "DeepLabV3",
  initialize = function(backbone, classifier, aux_classifier = NULL) {
    self$backbone <- backbone
    self$classifier <- classifier
    self$aux_classifier <- aux_classifier
  },
  forward = function(x) {
    input_shape <- tail(x$shape, 2)
    features <- self$backbone(x)
    x <- features$out
    x <- self$classifier(x)
    x <- torch::nnf_interpolate(x, size = input_shape, mode = 'bilinear', align_corners = FALSE)
    out <- list(out = x)
    if (!is.null(self$aux_classifier) && !is.null(features$aux)) {
      aux <- self$aux_classifier(features$aux)
      aux <- torch::nnf_interpolate(aux, size = input_shape, mode = 'bilinear', align_corners = FALSE)
      out$aux <- aux
    }
    out
  }
)

.deeplabv3_resnet <- function(arch, block, layers, pretrained, progress, num_classes, aux_loss, ...) {
  if (pretrained) {
    aux_loss <- TRUE
    if (num_classes != 21)
      cli::cli_abort("pretrained models expect num_classes = 21")
  }
  if (is.null(aux_loss))
    aux_loss <- FALSE

  backbone <- resnet(block, layers, replace_stride_with_dilation = c(FALSE, TRUE, TRUE), ...)
  backbone <- ResNetBackbone(backbone, aux_loss)
  classifier <- DeepLabHead(2048, num_classes)
  aux_classifier <- if (aux_loss) FCNHead(1024, 256, num_classes) else NULL
  model <- DeepLabV3(backbone, classifier, aux_classifier)

  if (pretrained) {
    url <- deeplabv3_model_urls[paste0(arch, '_coco')]
    state_dict_path <- download_and_cache(url)
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}

#' @describeIn model_deeplabv3 DeepLabV3 model with a ResNet-50 backbone
#' @export
model_deeplabv3_resnet50 <- function(pretrained = FALSE, progress = TRUE,
                                     num_classes = 21, aux_loss = NULL, ...) {
  .deeplabv3_resnet('deeplabv3_resnet50', bottleneck, c(3,4,6,3),
                    pretrained, progress, num_classes, aux_loss, ...)
}

#' @describeIn model_deeplabv3 DeepLabV3 model with a ResNet-101 backbone
#' @export
model_deeplabv3_resnet101 <- function(pretrained = FALSE, progress = TRUE,
                                      num_classes = 21, aux_loss = NULL, ...) {
  .deeplabv3_resnet('deeplabv3_resnet101', bottleneck, c(3,4,23,3),
                    pretrained, progress, num_classes, aux_loss, ...)
}
