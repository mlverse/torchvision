#' ConvNeXt Segmentation Models
#'
#' @description
#' Semantic segmentation models that use a ConvNeXt backbone with either
#' an FCN (Fully Convolutional Network) head or a UPerNet (Unified Perceptual
#' Parsing Network) head.
#'
#' These models follow the architecture patterns from mmsegmentation and
#' can be used for semantic segmentation tasks.
#'
#' @section Available FCN Models:
#' \itemize{
#'   \item `model_convnext_tiny_fcn()`
#'   \item `model_convnext_small_fcn()`
#'   \item `model_convnext_base_fcn()`
#' }
#'
#' @section Available UPerNet Models:
#' \itemize{
#'   \item `model_convnext_tiny_upernet()`
#'   \item `model_convnext_small_upernet()`
#'   \item `model_convnext_base_upernet()`
#' }
#'
#' @param num_classes Number of output segmentation classes. Default: 21 (PASCAL VOC).
#' @param aux_loss Logical. If TRUE, includes an auxiliary classifier branch.
#'   Default: FALSE.
#' @param pretrained_backbone Logical. If TRUE, loads ImageNet pretrained
#'
#'   weights for the ConvNeXt backbone. Default: FALSE.
#' @param pool_scales Numeric vector. Pooling scales used in the Pyramid Pooling
#'   Module for UPerNet models. Default: c(1, 2, 3, 6).
#' @param ... Additional arguments passed to the backbone.
#'
#' @return An `nn_module` representing the segmentation model.
#'
#' @examples
#' \dontrun{
#' library(magrittr)
#' norm_mean <- c(0.485, 0.456, 0.406) # ImageNet normalization constants
#' norm_std  <- c(0.229, 0.224, 0.225)
#'
#' # Use a publicly available image
#' wmc <- "https://upload.wikimedia.org/wikipedia/commons/thumb/"
#' url <- "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg"
#' img <- base_loader(paste0(wmc, url))
#'
#' input <- img %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(520, 520)) %>%
#'   transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)
#'
#' # ConvNeXt Tiny FCN segmentation
#' model <- model_convnext_tiny_fcn(num_classes = 21, pretrained_backbone = TRUE)
#' model$eval()
#' output <- model(batch)
#'
#' # Visualize result
#' segmented <- draw_segmentation_masks(input, output$out$squeeze(1))
#' tensor_image_display(segmented)
#'
#' # ConvNeXt Tiny UPerNet segmentation
#' model <- model_convnext_tiny_upernet(num_classes = 21, pretrained_backbone = TRUE)
#' model$eval()
#' output <- model(batch)
#'
#' # Visualize result
#' segmented <- draw_segmentation_masks(input, output$out$squeeze(1))
#' tensor_image_display(segmented)
#' }
#'
#' @family semantic_segmentation_model
#' @name model_convnext_segmentation
NULL


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

validate_segmentation_num_classes <- function(num_classes) {
  if (num_classes <= 0) {
    cli_abort("{.var num_classes} must be positive")
  }
}


# ------------------------------------------------------------------------------
# ConvNeXt Backbone for Segmentation (extracts multi-scale features)
# ------------------------------------------------------------------------------

convnext_segmentation_backbone <- torch::nn_module(

  "convnext_segmentation_backbone",
  initialize = function(convnext_model) {
    self$model <- convnext_model
  },
  forward = function(x) {
    # Extract features at different scales
    c2 <- x %>%
      self$model$downsample_layers[[1]]() %>%
      self$model$stages[[1]]()

    c3 <- c2 %>%
      self$model$downsample_layers[[2]]() %>%
      self$model$stages[[2]]()

    c4 <- c3 %>%
      self$model$downsample_layers[[3]]() %>%
      self$model$stages[[3]]()

    c5 <- c4 %>%
      self$model$downsample_layers[[4]]() %>%
      self$model$stages[[4]]()

    list(c2 = c2, c3 = c3, c4 = c4, c5 = c5)
  }
)


# ------------------------------------------------------------------------------
# FCN Head
# ------------------------------------------------------------------------------

convnext_fcn_head <- function(in_channels, channels, num_classes) {
  torch::nn_sequential(
    torch::nn_conv2d(in_channels, channels, kernel_size = 3, padding = 1, bias = FALSE),
    torch::nn_batch_norm2d(channels),
    torch::nn_relu(inplace = TRUE),
    torch::nn_dropout(0.1),
    torch::nn_conv2d(channels, num_classes, kernel_size = 1)
  )
}


# FCN Segmentation Model
convnext_fcn <- torch::nn_module(
  "convnext_fcn",
  initialize = function(backbone, classifier, aux_classifier = NULL) {
    self$backbone <- backbone
    self$classifier <- classifier
    self$aux_classifier <- aux_classifier
  },
  forward = function(x) {
    input_shape <- x$shape[3:4]
    features <- self$backbone(x)

    # Main classifier uses c5 (deepest features)
    out <- self$classifier(features$c5)
    out <- torch::nnf_interpolate(out, size = input_shape, mode = "bilinear", align_corners = FALSE)

    result <- list(out = out)

    # Auxiliary classifier uses c4 (intermediate features)
    if (!is.null(self$aux_classifier)) {
      aux_out <- self$aux_classifier(features$c4)
      aux_out <- torch::nnf_interpolate(aux_out, size = input_shape, mode = "bilinear", align_corners = FALSE)
      result$aux <- aux_out
    }

    result
  }
)


# ------------------------------------------------------------------------------
# Pyramid Pooling Module (PPM) for UPerNet
# ------------------------------------------------------------------------------

ppm_module <- torch::nn_module(
  "PPM",
  initialize = function(in_channels, channels, pool_scales = c(1, 2, 3, 6)) {
    self$pool_scales <- pool_scales
    self$stages <- torch::nn_module_list()

    for (scale in pool_scales) {
      self$stages$append(
        torch::nn_sequential(
          torch::nn_adaptive_avg_pool2d(scale),
          torch::nn_conv2d(in_channels, channels, kernel_size = 1, bias = FALSE),
          torch::nn_batch_norm2d(channels),
          torch::nn_relu(inplace = TRUE)
        )
      )
    }
  },
  forward = function(x) {
    target_size <- x$shape[3:4]
    ppm_outs <- list()

    for (i in seq_along(self$stages)) {
      ppm_out <- self$stages[[i]](x)
      ppm_out <- torch::nnf_interpolate(
        ppm_out,
        size = as.integer(target_size),
        mode = "bilinear",
        align_corners = FALSE
      )
      ppm_outs[[i]] <- ppm_out
    }

    ppm_outs
  }
)


# ------------------------------------------------------------------------------
# UPerNet Head
# ------------------------------------------------------------------------------

upernet_head <- torch::nn_module(
  "UPerNetHead",
  initialize = function(in_channels, channels, num_classes, pool_scales = c(1, 2, 3, 6)) {
    # in_channels is a vector: c(c2_channels, c3_channels, c4_channels, c5_channels)
    self$in_channels <- in_channels
    self$channels <- channels
    self$num_classes <- num_classes

    # PSP module on deepest features (c5)
    self$psp_modules <- ppm_module(in_channels[4], channels, pool_scales)
    self$bottleneck <- torch::nn_sequential(
      torch::nn_conv2d(in_channels[4] + length(pool_scales) * channels, channels, kernel_size = 3, padding = 1, bias = FALSE),
      torch::nn_batch_norm2d(channels),
      torch::nn_relu(inplace = TRUE)
    )

    # Lateral convolutions for FPN
    self$lateral_convs <- torch::nn_module_list()
    self$fpn_convs <- torch::nn_module_list()

    for (i in 1:3) {  # c2, c3, c4
      l_conv <- torch::nn_sequential(
        torch::nn_conv2d(in_channels[i], channels, kernel_size = 1, bias = FALSE),
        torch::nn_batch_norm2d(channels),
        torch::nn_relu(inplace = TRUE)
      )
      fpn_conv <- torch::nn_sequential(
        torch::nn_conv2d(channels, channels, kernel_size = 3, padding = 1, bias = FALSE),
        torch::nn_batch_norm2d(channels),
        torch::nn_relu(inplace = TRUE)
      )
      self$lateral_convs$append(l_conv)
      self$fpn_convs$append(fpn_conv)
    }

    # Final bottleneck
    self$fpn_bottleneck <- torch::nn_sequential(
      torch::nn_conv2d(4 * channels, channels, kernel_size = 3, padding = 1, bias = FALSE),
      torch::nn_batch_norm2d(channels),
      torch::nn_relu(inplace = TRUE)
    )

    # Classification head
    self$cls_seg <- torch::nn_conv2d(channels, num_classes, kernel_size = 1)
  },

  psp_forward = function(x) {
    psp_outs <- list(x)
    psp_outs <- c(psp_outs, self$psp_modules(x))
    psp_out <- torch::torch_cat(psp_outs, dim = 2)
    self$bottleneck(psp_out)
  },

  forward = function(features) {
    # features is a list: list(c2, c3, c4, c5)
    inputs <- list(features$c2, features$c3, features$c4, features$c5)

    # Build laterals
    laterals <- list()
    for (i in 1:3) {
      laterals[[i]] <- self$lateral_convs[[i]](inputs[[i]])
    }
    # Add PSP output as the last lateral
    laterals[[4]] <- self$psp_forward(inputs[[4]])

    # Build top-down path
    for (i in 4:2) {
      prev_shape <- laterals[[i - 1]]$shape[3:4]
      laterals[[i - 1]] <- laterals[[i - 1]] + torch::nnf_interpolate(
        laterals[[i]],
        size = as.integer(prev_shape),
        mode = "bilinear",
        align_corners = FALSE
      )
    }

    # Build FPN outputs
    fpn_outs <- list()
    for (i in 1:3) {
      fpn_outs[[i]] <- self$fpn_convs[[i]](laterals[[i]])
    }
    fpn_outs[[4]] <- laterals[[4]]

    # Resize all to the largest feature map size
    target_size <- fpn_outs[[1]]$shape[3:4]
    for (i in 2:4) {
      fpn_outs[[i]] <- torch::nnf_interpolate(
        fpn_outs[[i]],
        size = as.integer(target_size),
        mode = "bilinear",
        align_corners = FALSE
      )
    }

    # Concatenate and apply bottleneck
    fpn_cat <- torch::torch_cat(fpn_outs, dim = 2)
    feats <- self$fpn_bottleneck(fpn_cat)

    # Classification
    self$cls_seg(feats)
  }
)


# UPerNet Segmentation Model
convnext_upernet <- torch::nn_module(
  "convnext_upernet",
  initialize = function(backbone, decode_head, aux_classifier = NULL) {
    self$backbone <- backbone
    self$decode_head <- decode_head
    self$aux_classifier <- aux_classifier
  },
  forward = function(x) {
    input_shape <- x$shape[3:4]
    features <- self$backbone(x)

    # Main decoder
    out <- self$decode_head(features)
    out <- torch::nnf_interpolate(out, size = input_shape, mode = "bilinear", align_corners = FALSE)

    result <- list(out = out)

    # Auxiliary classifier uses c4
    if (!is.null(self$aux_classifier)) {
      aux_out <- self$aux_classifier(features$c4)
      aux_out <- torch::nnf_interpolate(aux_out, size = input_shape, mode = "bilinear", align_corners = FALSE)
      result$aux <- aux_out
    }

    result
  }
)


# ------------------------------------------------------------------------------
# ConvNeXt FCN Model Factories
# ------------------------------------------------------------------------------

#' @describeIn model_convnext_segmentation ConvNeXt Tiny with FCN head
#' @export
model_convnext_tiny_fcn <- function(num_classes = 21,
                                    aux_loss = FALSE,
                                    pretrained_backbone = FALSE,
                                    ...) {
  validate_segmentation_num_classes(num_classes)

  convnext <- model_convnext_tiny_1k(pretrained = pretrained_backbone, ...)
  backbone <- convnext_segmentation_backbone(convnext)

  # ConvNeXt Tiny dims: c(96, 192, 384, 768)
  classifier <- convnext_fcn_head(768, 512, num_classes)
  aux_classifier <- if (aux_loss) convnext_fcn_head(384, 256, num_classes) else NULL

  convnext_fcn(backbone, classifier, aux_classifier)
}


#' @describeIn model_convnext_segmentation ConvNeXt Small with FCN head
#' @export
model_convnext_small_fcn <- function(num_classes = 21,
                                     aux_loss = FALSE,
                                     pretrained_backbone = FALSE,
                                     ...) {
  validate_segmentation_num_classes(num_classes)

  convnext <- model_convnext_small_22k(pretrained = pretrained_backbone, ...)
  backbone <- convnext_segmentation_backbone(convnext)

  # ConvNeXt Small dims: c(96, 192, 384, 768)
  classifier <- convnext_fcn_head(768, 512, num_classes)
  aux_classifier <- if (aux_loss) convnext_fcn_head(384, 256, num_classes) else NULL

  convnext_fcn(backbone, classifier, aux_classifier)
}


#' @describeIn model_convnext_segmentation ConvNeXt Base with FCN head
#' @export
model_convnext_base_fcn <- function(num_classes = 21,
                                    aux_loss = FALSE,
                                    pretrained_backbone = FALSE,
                                    ...) {
  validate_segmentation_num_classes(num_classes)

  convnext <- model_convnext_base_1k(pretrained = pretrained_backbone, ...)
  backbone <- convnext_segmentation_backbone(convnext)

  # ConvNeXt Base dims: c(128, 256, 512, 1024)
  classifier <- convnext_fcn_head(1024, 512, num_classes)
  aux_classifier <- if (aux_loss) convnext_fcn_head(512, 256, num_classes) else NULL

  convnext_fcn(backbone, classifier, aux_classifier)
}


# ------------------------------------------------------------------------------
# ConvNeXt UPerNet Model Factories
# ------------------------------------------------------------------------------

#' @describeIn model_convnext_segmentation ConvNeXt Tiny with UPerNet head
#' @export
model_convnext_tiny_upernet <- function(num_classes = 21,
                                        aux_loss = FALSE,
                                        pretrained_backbone = FALSE,
                                        pool_scales = c(1, 2, 3, 6),
                                        ...) {
  validate_segmentation_num_classes(num_classes)

  convnext <- model_convnext_tiny_1k(pretrained = pretrained_backbone, ...)
  backbone <- convnext_segmentation_backbone(convnext)

  # ConvNeXt Tiny dims: c(96, 192, 384, 768)
  decode_head <- upernet_head(
    in_channels = c(96, 192, 384, 768),
    channels = 512,
    num_classes = num_classes,
    pool_scales = pool_scales
  )
  aux_classifier <- if (aux_loss) convnext_fcn_head(384, 256, num_classes) else NULL

  convnext_upernet(backbone, decode_head, aux_classifier)
}


#' @describeIn model_convnext_segmentation ConvNeXt Small with UPerNet head
#' @export
model_convnext_small_upernet <- function(num_classes = 21,
                                         aux_loss = FALSE,
                                         pretrained_backbone = FALSE,
                                         pool_scales = c(1, 2, 3, 6),
                                         ...) {
  validate_segmentation_num_classes(num_classes)

  convnext <- model_convnext_small_22k(pretrained = pretrained_backbone, ...)
  backbone <- convnext_segmentation_backbone(convnext)

  # ConvNeXt Small dims: c(96, 192, 384, 768)
  decode_head <- upernet_head(
    in_channels = c(96, 192, 384, 768),
    channels = 512,
    num_classes = num_classes,
    pool_scales = pool_scales
  )
  aux_classifier <- if (aux_loss) convnext_fcn_head(384, 256, num_classes) else NULL

  convnext_upernet(backbone, decode_head, aux_classifier)
}


#' @describeIn model_convnext_segmentation ConvNeXt Base with UPerNet head
#' @export
model_convnext_base_upernet <- function(num_classes = 21,
                                        aux_loss = FALSE,
                                        pretrained_backbone = FALSE,
                                        pool_scales = c(1, 2, 3, 6),
                                        ...) {
  validate_segmentation_num_classes(num_classes)

  convnext <- model_convnext_base_1k(pretrained = pretrained_backbone, ...)
  backbone <- convnext_segmentation_backbone(convnext)

  # ConvNeXt Base dims: c(128, 256, 512, 1024)
  decode_head <- upernet_head(
    in_channels = c(128, 256, 512, 1024),
    channels = 512,
    num_classes = num_classes,
    pool_scales = pool_scales
  )
  aux_classifier <- if (aux_loss) convnext_fcn_head(512, 256, num_classes) else NULL

  convnext_upernet(backbone, decode_head, aux_classifier)
}
