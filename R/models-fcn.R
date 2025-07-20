#' FCN - Fully Convolutional Network
#'
#' Constructs an FCN (Fully Convolutional Network) model for semantic segmentation,
#' based on a ResNet backbone as described in
#' \href{https://arxiv.org/abs/1411.4038}{Fully Convolutional Networks for Semantic Segmentation}.
#'
#' The 21 output classes follow the PASCAL VOC convention:
#' `background`, `aeroplane`, `bicycle`, `bird`, `boat`,
#' `bottle`, `bus`, `car`, `cat`, `chair`,
#' `cow`, `dining table`, `dog`, `horse`, `motorbike`,
#' `person`, `potted plant`, `sheep`, `sofa`, `train`,
#' `tv/monitor`.
#'
#' @inheritParams model_resnet18
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
#' \dontrun{
#' model <- model_fcn_resnet50(pretrained = TRUE)
#' input <- torch::torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' # extract predicted classes for the first element of the batch
#' preds <- output$out$argmax(dim = 2)[1, ..]
#' # build a boolean mask for the "dog" class
#' dog_id <- which(voc_segmentation_classes == "dog")
#' dog_mask <- preds$eq(dog_id)$unsqueeze(1)
#' img <- input[1, ..]
#' segmented <- draw_segmentation_masks(img, dog_mask)
#' tensor_image_browse(segmented)
#'
#' model <- model_fcn_resnet101(pretrained = FALSE, aux_loss = TRUE)
#' input <- torch::torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' preds <- output$out$argmax(dim = 2)[1, ..]
#' person_id <- which(voc_segmentation_classes == "person")
#' person_mask <- preds$eq(person_id)$unsqueeze(1)
#' img <- input[1, ..]
#' segmented <- draw_segmentation_masks(img, person_mask)
#' tensor_image_browse(segmented)
#' }
NULL

voc_segmentation_classes <- c(
  "background", "aeroplane", "bicycle", "bird", "boat",
  "bottle", "bus", "car", "cat", "chair",
  "cow", "dining table", "dog", "horse", "motorbike",
  "person", "potted plant", "sheep", "sofa", "train",
  "tv/monitor"
)

fcn_model_urls <- list(
  fcn_resnet50_coco = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/fcn_resnet50_coco.pth",
    "c79a7e2675d73817cfe6ba383be6ca7d", "135 MB"),
  fcn_resnet101_coco = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/fcn_resnet101_coco.pth",
    "369109597fa68546df1231ae2fe0f66f", "207 MB")
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
    state_dict_path <- download_and_cache(resnet_model_urls["resnet50"])
    state_dict <- torch::load_state_dict(state_dict_path)
    backbone$load_state_dict(state_dict, strict = FALSE)
  }

  classifier <- fcn_head(backbone$out_channels, 512, num_classes)
  aux_classifier <- if (aux_loss) fcn_head(backbone$aux_channels, 256, num_classes) else NULL

  model <- fcn(backbone, classifier, aux_classifier)

  if (pretrained) {
    r <- fcn_model_urls$fcn_resnet50_coco
    name <- "model_fcn_resnet50"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!is.na(r[2])) {
      if (!tools::md5sum(state_dict_path) == r[2])
        runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }
    state_dict <- torch::load_state_dict(state_dict_path)
    strict_loading <- num_classes == 21 && aux_loss
    model$load_state_dict(state_dict, strict = strict_loading)
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
    state_dict_path <- download_and_cache(resnet_model_urls["resnet101"])
    state_dict <- torch::load_state_dict(state_dict_path)
    backbone$load_state_dict(state_dict, strict = FALSE)
  }

  classifier <- fcn_head(backbone$out_channels, 512, num_classes)
  aux_classifier <- if (aux_loss) fcn_head(backbone$aux_channels, 256, num_classes) else NULL

  model <- fcn(backbone, classifier, aux_classifier)

  if (pretrained) {
    r <- fcn_model_urls$fcn_resnet101_coco
    name <- "model_fcn_resnet101"
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!is.na(r[2])) {
      if (!tools::md5sum(state_dict_path) == r[2])
        runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    }
    state_dict <- torch::load_state_dict(state_dict_path)
    strict_loading <- num_classes == 21 && aux_loss
    model$load_state_dict(state_dict, strict = strict_loading)
  }

  model
}
