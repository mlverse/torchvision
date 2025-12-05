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
#' @param num_classes Number of output classes (including background).
#'   Must be strictly positive.
#' @param pretrained_backbone Logical, if `TRUE` the ConvNeXt backbone
#'   weights are loaded from ImageNet pretraining.
#' @param ... Additional arguments forwarded to the underlying ConvNeXt
#'   backbone constructors.
#'
#' @examples
#' \dontrun{
#' library(torch)
#' library(magrittr)
#'
#' # Create a random input tensor for demonstration
#' # Shape: (batch, channels, height, width)
#' batch <- torch_randn(1, 3, 224, 224)
#'
#' # Build model with pretrained backbone (detection head is random)
#' model <- model_convnext_tiny_detection(pretrained_backbone = TRUE)
#' model$eval()
#'
#' # Run inference
#' torch::with_no_grad({
#'   output <- model(batch)
#' })
#'
#' # Access detection outputs
#' pred <- output$detections
#' cat("Number of detections:", pred$boxes$size()[1], "\n")
#' cat("Box format: [x_min, y_min, x_max, y_max]\n")
#'
#' # Note: Without pretrained detection head weights, predictions are random.
#' # For meaningful results, you would need to train the model on your data
#' # or use a model with pretrained detection weights when available.
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


