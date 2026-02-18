#' ConvNeXt Detection Models (Faster R-CNN style)
#'
#' @description
#' Object detection models that use a ConvNeXt backbone with a Feature
#' Pyramid Network (FPN) and the same detection head as the Faster R-CNN
#' models implemented in `model_fasterrcnn_*`.
#'
#' These helpers mirror the architecture used in
#' `model_fasterrcnn_resnet50_fpn()`, but swap the ResNet backbone for
#' ConvNeXt variants.
#'
#' @section Available Models:
#' \itemize{
#'   \item `model_convnext_tiny_detection()`
#'   \item `model_convnext_small_detection()`
#'   \item `model_convnext_base_detection()`
#' }
#'
#' @inheritParams model_fasterrcnn_resnet50_fpn
#' @param pretrained_backbone Logical, if `TRUE` the ConvNeXt backbone
#'   weights are loaded from ImageNet pretraining.
#'
#' @note Currently, detection head weights are randomly initialized, so predicted
#' bounding-boxes are random. For meaningful results, you need to train the model
#' detection head on your data.
#'
#' @examples
#' \dontrun{
#' library(magrittr)
#' norm_mean <- c(0.485, 0.456, 0.406) # ImageNet normalization constants
#' norm_std  <- c(0.229, 0.224, 0.225)
#'
#' # Use a publicly available image
#' url <- paste0("https://upload.wikimedia.org/wikipedia/commons/thumb/",
#'        "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg")
#' img <- magick_loader(url) %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(520, 520))
#'
#' input <- img %>%
#'   transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)    # Add batch dimension (1, 3, H, W)
#'
#' # ConvNeXt Tiny detection
#' model <- model_convnext_tiny_detection(pretrained_backbone = TRUE)
#' model$eval()
#' # Please wait 2 mins + on CPU
#' pred <- model(batch)$detections[[1]]
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' topk <- pred$scores$topk(k = 5)[[2]]
#' boxes <- pred$boxes[topk, ]
#' labels <- imagenet_label(as.integer(pred$labels[topk]))
#'
#' # `draw_bounding_box()` may fail if bbox values are not consistent.
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(img, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#' }
#'
#' @family object_detection_model
#' @name model_convnext_detection
NULL


convnext_fpn_backbone_tiny <- function(pretrained_backbone = FALSE, ...) {
  convnext <- model_convnext_tiny_1k(pretrained = pretrained_backbone, ...)

  convnext_body <- torch::nn_module(
    initialize = function() {
      self$model <- convnext
    },
    forward = function(x) {
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

      list(c2, c3, c4, c5)
    }
  )

  backbone_module <- torch::nn_module(
    initialize = function() {
      self$body <- convnext_body()
      self$fpn <- fpn_module(
        in_channels = c(96, 192, 384, 768),
        out_channels = 256
      )()
    },
    forward = function(x) {
      c2_to_c5 <- self$body(x)
      self$fpn(c2_to_c5)
    }
  )

  backbone <- backbone_module()
  backbone$out_channels <- 256
  backbone
}


convnext_fpn_backbone_small <- function(pretrained_backbone = FALSE, ...) {
  convnext <- model_convnext_small_22k(pretrained = pretrained_backbone, ...)

  convnext_body <- torch::nn_module(
    initialize = function() {
      self$model <- convnext
    },
    forward = function(x) {
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

      list(c2, c3, c4, c5)
    }
  )

  backbone_module <- torch::nn_module(
    initialize = function() {
      self$body <- convnext_body()
      self$fpn <- fpn_module(
        in_channels = c(96, 192, 384, 768),
        out_channels = 256
      )()
    },
    forward = function(x) {
      c2_to_c5 <- self$body(x)
      self$fpn(c2_to_c5)
    }
  )

  backbone <- backbone_module()
  backbone$out_channels <- 256
  backbone
}


convnext_fpn_backbone_base <- function(pretrained_backbone = FALSE, ...) {
  convnext <- model_convnext_base_1k(pretrained = pretrained_backbone, ...)

  convnext_body <- torch::nn_module(
    initialize = function() {
      self$model <- convnext
    },
    forward = function(x) {
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

      list(c2, c3, c4, c5)
    }
  )

  backbone_module <- torch::nn_module(
    initialize = function() {
      self$body <- convnext_body()
      self$fpn <- fpn_module(
        in_channels = c(128, 256, 512, 1024),
        out_channels = 256
      )()
    },
    forward = function(x) {
      c2_to_c5 <- self$body(x)
      self$fpn(c2_to_c5)
    }
  )

  backbone <- backbone_module()
  backbone$out_channels <- 256
  backbone
}


validate_convnext_num_classes <- function(num_classes) {
  if (num_classes <= 0) {
    cli_abort("{.var num_classes} must be positive")
  }
}


#' @describeIn model_convnext_detection ConvNeXt Tiny with FPN detection head
#' @export
model_convnext_tiny_detection <- function(num_classes = 91,
                                          pretrained_backbone = FALSE,
                                          ...) {
  validate_convnext_num_classes(num_classes)

  backbone <- convnext_fpn_backbone_tiny(
    pretrained_backbone = pretrained_backbone,
    ...
  )

  model <- fasterrcnn_model(backbone, num_classes = num_classes)()
  model
}


#' @describeIn model_convnext_detection ConvNeXt Small with FPN detection head
#' @export
model_convnext_small_detection <- function(num_classes = 91,
                                           pretrained_backbone = FALSE,
                                           ...) {
  validate_convnext_num_classes(num_classes)

  backbone <- convnext_fpn_backbone_small(
    pretrained_backbone = pretrained_backbone,
    ...
  )

  model <- fasterrcnn_model(backbone, num_classes = num_classes)()
  model
}


#' @describeIn model_convnext_detection ConvNeXt Base with FPN detection head
#' @export
model_convnext_base_detection <- function(num_classes = 91,
                                          pretrained_backbone = FALSE,
                                          ...) {
  validate_convnext_num_classes(num_classes)

  backbone <- convnext_fpn_backbone_base(
    pretrained_backbone = pretrained_backbone,
    ...
  )

  model <- fasterrcnn_model(backbone, num_classes = num_classes)()
  model
}


