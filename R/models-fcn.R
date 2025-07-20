#' FCN - Fully Convolutional Network
#'
#' Constructs an FCN (Fully Convolutional Network) model for semantic segmentation,
#' based on a ResNet backbone as described in
#' \href{https://arxiv.org/abs/1411.4038}{Fully Convolutional Networks for Semantic Segmentation}.
#'
#' @param pretrained If TRUE, returns a model pre-trained on COCO train2017.
#' @param progress If TRUE, displays a progress bar of the download.
#' @param num_classes Number of output classes. Default: 21.
#' @param aux_loss If TRUE, includes the auxiliary classifier. If NULL, defaults to TRUE when `pretrained = TRUE`.
#' @param pretrained_backbone If TRUE, uses a backbone pre-trained on ImageNet.
#' @param ... Additional arguments passed to the backbone implementation.
#'
#' @return An `nn_module` representing the FCN model.
#'
#' @family models
#' @name model_fcn_resnet
#' @rdname model_fcn_resnet
#'
#' @examples
#' model <- model_fcn_resnet50(pretrained = FALSE)
#' input <- torch::torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' str(output)
#'
#' model <- model_fcn_resnet101(pretrained = FALSE, aux_loss = TRUE)
#' input <- torch::torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' str(output)
NULL

fcn_model_urls <- c(
  'fcn_resnet50_coco' = 'https://torch-cdn.mlverse.org/models/vision/v2/models/fcn_resnet50_coco.pth',
  'fcn_resnet101_coco' = 'https://torch-cdn.mlverse.org/models/vision/v2/models/fcn_resnet101_coco.pth',
  'resnet50' = 'https://torch-cdn.mlverse.org/models/vision/v2/models/resnet50.pth',
  'resnet101' = 'https://torch-cdn.mlverse.org/models/vision/v2/models/resnet101.pth'
)

fcn_head <- function(in_channels, channels, num_classes) {
  torch::nn_sequential(
    torch::nn_conv2d(in_channels, channels, kernel_size = 3, padding = 1, bias = FALSE),
    torch::nn_batch_norm2d(channels),
    torch::nn_relu(inplace = TRUE),
    torch::nn_dropout(0.1),
    torch::nn_conv2d(channels, num_classes, kernel_size = 1)
  )
}

fcn_backbone <- torch::nn_module(
  "fcn_backbone",
  initialize = function(block, layers, replace_stride_with_dilation = c(FALSE, FALSE, FALSE), ...) {
    args <- list(...)
    args$pretrained <- NULL  # remove if present

    resnet_model <- do.call(resnet, c(list(block = block, layers = layers,
                                           replace_stride_with_dilation = replace_stride_with_dilation),
                                      args))

    self$conv1 <- resnet_model$conv1
    self$bn1 <- resnet_model$bn1
    self$relu <- resnet_model$relu
    self$maxpool <- resnet_model$maxpool

    self$layer1 <- resnet_model$layer1
    self$layer2 <- resnet_model$layer2
    self$layer3 <- resnet_model$layer3
    self$layer4 <- resnet_model$layer4

    self$out_channels <- 512 * block$public_fields$expansion
    self$aux_channels <- 256 * block$public_fields$expansion
  },

  forward = function(x) {
    x <- self$conv1(x)
    x <- self$bn1(x)
    x <- self$relu(x)
    x <- self$maxpool(x)

    x <- self$layer1(x)
    x <- self$layer2(x)
    x <- self$layer3(x)
    aux <- x
    x <- self$layer4(x)
    list(out = x, aux = aux)
  }
)

fcn <- torch::nn_module(
  "fcn",
  initialize = function(backbone, classifier, aux_classifier = NULL) {
    self$backbone <- backbone
    self$classifier <- classifier
    self$aux_classifier <- aux_classifier
  },

  forward = function(x) {
    input_shape <- x$shape[3:4]
    features <- self$backbone(x)
    x <- features$out
    x <- self$classifier(x)
    x <- torch::nnf_interpolate(x, size = input_shape, mode = "bilinear", align_corners = FALSE)
    result <- list(out = x)
    if (!is.null(self$aux_classifier)) {
      x_aux <- features$aux
      x_aux <- self$aux_classifier(x_aux)
      x_aux <- torch::nnf_interpolate(x_aux, size = input_shape, mode = "bilinear", align_corners = FALSE)
      result$aux <- x_aux
    }
    result
  }
)

#' @rdname model_fcn_resnet
#' @export
model_fcn_resnet50 <- function(pretrained = FALSE, progress = TRUE, num_classes = 21,
                               aux_loss = NULL, pretrained_backbone = TRUE, ...) {
  if (is.null(aux_loss)) aux_loss <- pretrained

  backbone <- fcn_backbone(bottleneck, c(3, 4, 6, 3),
                           replace_stride_with_dilation = c(FALSE, FALSE, FALSE),
                           ...)

  if (pretrained_backbone) {
    state_dict_path <- download_and_cache(fcn_model_urls['resnet50'])
    state_dict <- torch::load_state_dict(state_dict_path)
    backbone$load_state_dict(state_dict, strict = FALSE)
  }

  classifier <- fcn_head(backbone$out_channels, 512, num_classes)
  aux_classifier <- if (aux_loss) fcn_head(backbone$aux_channels, 256, num_classes) else NULL

  model <- fcn(backbone, classifier, aux_classifier)

  if (pretrained) {
    state_dict_path <- download_and_cache(fcn_model_urls['fcn_resnet50_coco'])
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}

#' @rdname model_fcn_resnet
#' @export
model_fcn_resnet101 <- function(pretrained = FALSE, progress = TRUE, num_classes = 21,
                                aux_loss = NULL, pretrained_backbone = TRUE, ...) {
  if (is.null(aux_loss)) aux_loss <- pretrained

  backbone <- fcn_backbone(bottleneck, c(3, 4, 23, 3),
                           replace_stride_with_dilation = c(FALSE, FALSE, FALSE),
                           ...)

  if (pretrained_backbone) {
    state_dict_path <- download_and_cache(fcn_model_urls['resnet101'])
    state_dict <- torch::load_state_dict(state_dict_path)
    backbone$load_state_dict(state_dict, strict = FALSE)
  }

  classifier <- fcn_head(backbone$out_channels, 512, num_classes)
  aux_classifier <- if (aux_loss) fcn_head(backbone$aux_channels, 256, num_classes) else NULL

  model <- fcn(backbone, classifier, aux_classifier)

  if (pretrained) {
    state_dict_path <- download_and_cache(fcn_model_urls['fcn_resnet101_coco'])
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}
