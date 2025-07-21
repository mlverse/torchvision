#' Faster R-CNN with MobileNet V3 Large FPN
#'
#' Constructs a Faster R-CNN model with a MobileNet V3 Large FPN backbone.
#'
#' @inheritParams model_fasterrcnn_resnet50_fpn
#'
#' @return A `fasterrcnn_model` nn_module.
#' @export
model_fasterrcnn_mobilenet_v3_large_fpn <- function(pretrained = FALSE,
                                                    progress = TRUE,
                                                    num_classes = 91, ...) {
  backbone <- mobilenet_v3_fpn_backbone(pretrained = pretrained)

  model <- fasterrcnn_model(backbone, num_classes = num_classes)

  if (pretrained) {
    local_path <- "tools/models/fasterrcnn_mobilenet_v3_large_fpn.pth"
    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

#' Faster R-CNN with MobileNet V3 Large 320 FPN
#'
#' Constructs a Faster R-CNN model with a MobileNet V3 Large 320 FPN backbone.
#'
#' @inheritParams model_fasterrcnn_resnet50_fpn
#'
#' @return A `fasterrcnn_model` nn_module.
#' @export
model_fasterrcnn_mobilenet_v3_large_320_fpn <- function(pretrained = FALSE,
                                                        progress = TRUE,
                                                        num_classes = 91, ...) {
  backbone <- mobilenet_v3_320_fpn_backbone(pretrained = pretrained)

  model <- fasterrcnn_model(backbone, num_classes = num_classes)

  if (pretrained) {
    local_path <- "tools/models/fasterrcnn_mobilenet_v3_large_320_fpn.pth"
    state_dict <- torch::load_state_dict(local_path)
    state_dict <- state_dict[!grepl("num_batches_tracked$", names(state_dict))]
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

mobilenet_v3_fpn_backbone <- function(pretrained = TRUE) {
  mobilenet <- model_mobilenet_v3_large(pretrained = pretrained)

  backbone <- torch::nn_module(
    initialize = function() {
      self$features <- mobilenet$features
      self$fpn <- fpn_module(
        in_channels = c(24, 40, 112, 160),
        out_channels = 256
      )()
    },
    forward = function(x) {
      feats <- list()
      for (i in seq_along(self$features)) {
        x <- self$features[[i]](x)
        if (i %in% c(4, 7, 12, 16)) {
          feats[[length(feats) + 1]] <- x
        }
      }
      self$fpn(feats)
    }
  )

  backbone <- backbone()
  backbone$out_channels <- 256
  backbone
}

mobilenet_v3_320_fpn_backbone <- function(pretrained = TRUE) {
  mobilenet <- model_mobilenet_v3_large(pretrained = pretrained)

  backbone <- torch::nn_module(
    initialize = function() {
      self$features <- mobilenet$features
      self$fpn <- fpn_module(
        in_channels = c(24, 40, 112, 160),
        out_channels = 256
      )()
    },
    forward = function(x) {
      feats <- list()
      for (i in seq_along(self$features)) {
        x <- self$features[[i]](x)
        if (i %in% c(4, 7, 12, 16)) {
          feats[[length(feats) + 1]] <- x
        }
      }
      self$fpn(feats)
    }
  )

  backbone <- backbone()
  backbone$out_channels <- 256
  backbone
}
