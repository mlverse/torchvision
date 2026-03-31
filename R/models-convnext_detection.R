#' ConvNeXt Detection Models (Faster R-CNN style)
#'
#' @description
#' Object detection models combining a ConvNeXt backbone with a Feature Pyramid
#' Network (FPN) and the Faster R-CNN detection head. The architecture mirrors
#' [model_fasterrcnn_resnet50_fpn()], with the ResNet backbone replaced by
#' ConvNeXt variants. The design follows the paper
#' [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545).
#'
#' ## Available Models
#' - `model_convnext_tiny_detection()`
#' - `model_convnext_small_detection()`
#' - `model_convnext_base_detection()`
#'
#' ## Backbone Performance (ImageNet-1k)
#'
#' Accuracy metrics reflect backbone classification performance only.
#' Detection head weights are randomly initialized and must be fine-tuned
#' on task-specific labelled data before meaningful predictions are produced.
#'
#' ```
#' | Model                             | Top-1 Acc | Top-5 Acc | Params  | GFLOPS | File Size | Backbone Weights              | Notes                    |
#' |-----------------------------------|-----------|-----------|---------|--------|-----------|-------------------------------|--------------------------|
#' | model_convnext_tiny_detection     | 82.5%     | 96.1%     | 28.6M   | 4.46   | 109 MB    | IMAGENET1K_V1                 | Tiny backbone, FPN head  |
#' | model_convnext_small_detection    | 83.6%     | 96.7%     | 50.2M   | 8.68   | 192 MB    | IMAGENET1K_V1 (22k pretrain)  | Small backbone, FPN head |
#' | model_convnext_base_detection     | 84.1%     | 96.9%     | 88.6M   | 15.36  | 338 MB    | IMAGENET1K_V1                 | Base backbone, FPN head  |
#' ```
#'
#' ## FPN Channel Configuration
#'
#' Each ConvNeXt variant produces four feature maps (C2–C5) fed into the FPN.
#' Channel widths differ between Tiny/Small and Base:
#'
#' ```
#' | Variant | FPN in_channels          | FPN out_channels |
#' |---------|--------------------------|------------------|
#' | Tiny    | c(96, 192, 384, 768)     | 256              |
#' | Small   | c(96, 192, 384, 768)     | 256              |
#' | Base    | c(128, 256, 512, 1024)   | 256              |
#' ```
#'
#' ## Weights Selection
#' - All variants use `IMAGENET1K_V1` backbone weights by default (supervised ImageNet-1k).
#' - The Small variant backbone (`model_convnext_small_22k`) was additionally
#'   pretrained on ImageNet-22k prior to fine-tuning on ImageNet-1k.
#' - Detection head weights are **randomly initialized** — bounding-box
#'   predictions are meaningless without fine-tuning on labelled detection data.
#' - Set `pretrained_backbone = TRUE` to load ImageNet backbone weights.
#'
#' @inheritParams model_fasterrcnn_resnet50_fpn
#' @param pretrained_backbone Logical. If `TRUE`, loads ImageNet-pretrained
#'   ConvNeXt backbone weights. Default: `FALSE`.
#'
#' @note
#' Detection head weights are randomly initialized. Predicted bounding boxes
#' will be arbitrary until the detection head is trained on labelled data.
#' Only the backbone benefits from `pretrained_backbone = TRUE`.
#'
#' @examples
#' \dontrun{
#' library(magrittr)
#' norm_mean <- c(0.485, 0.456, 0.406) # ImageNet normalization constants
#' norm_std  <- c(0.229, 0.224, 0.225)
#'
#' url <- paste0("https://upload.wikimedia.org/wikipedia/commons/thumb/",
#'               "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg")
#' img <- base_loader(url) %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(520, 520))
#'
#' input <- img %>% transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)    # Add batch dimension: (1, 3, H, W)
#'
#' # ConvNeXt Tiny detection
#' model <- model_convnext_tiny_detection(pretrained_backbone = TRUE)
#' model$eval()
#' # Please wait 2 mins + on CPU
#' pred     <- model(batch)$detections[[1]]
#' num_boxes <- as.integer(pred$boxes$size()[1])
#' topk     <- pred$scores$topk(k = 5)[[2]]
#' boxes    <- pred$boxes[topk, ]
#' labels   <- imagenet_classes(as.integer(pred$labels[topk]))
#'
#' # `draw_bounding_box()` may fail if bbox values are not consistent.
#' if (num_boxes > 0) {
#'   boxed <- draw_bounding_boxes(img, boxes, labels = labels)
#'   tensor_image_browse(boxed)
#' }
#' }
#'
#' @importFrom torch nn_module
#' @family object_detection_model
#' @rdname model_convnext_detection
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
      )
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
      )
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
      )
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

  model <- fasterrcnn_model(backbone, num_classes = num_classes)
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

  model <- fasterrcnn_model(backbone, num_classes = num_classes)
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

  model <- fasterrcnn_model(backbone, num_classes = num_classes)
  model
}