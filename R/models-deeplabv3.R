# DeepLabV3 model family with auxiliary classifier

#' DeepLabV3 Models
#'
#' Constructs DeepLabV3 semantic segmentation models with a ResNet backbone as
#' described in \emph{Rethinking Atrous Convolution for Semantic Image
#' Segmentation}. These models employ atrous spatial pyramid pooling to capture
#' multi-scale context.
#'
#' @section Task:
#' Semantic image segmentation with 21 output classes by default (COCO).
#'
#' @section Input Format:
#' The models expect input tensors of shape \code{(batch_size, 3, H, W)}. Typical
#' training uses 520x520 images.
#'
#' @inheritParams model_resnet18
#' @param num_classes Number of output classes.
#' @param aux_loss Logical or NULL. If `TRUE`, includes an auxiliary classifier branch.
#'   If `NULL` (default), the presence of aux classifier is inferred from pretrained weights.
#' @param pretrained_backbone If `TRUE` and `pretrained = FALSE`, loads
#'   ImageNet weights for the ResNet backbone.
#' @param ... Other parameters passed to the model implementation.
#'
#' @importFrom torch nn_module
#' @family semantic_segmentation_model
#'
#' @examples
#' \dontrun{
#' library(magrittr)
#' norm_mean <- c(0.485, 0.456, 0.406) # ImageNet normalization constants, see
#' # https://pytorch.org/vision/stable/models.html
#' norm_std  <- c(0.229, 0.224, 0.225)
#' # Use a publicly available image of an animal
#' wmc <- "https://upload.wikimedia.org/wikipedia/commons/thumb/"
#' url <- "e/ea/Morsan_Normande_vache.jpg/120px-Morsan_Normande_vache.jpg"
#' img <- base_loader(paste0(wmc,url))
#'
#' input <- img %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(520, 520)) %>%
#'  transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)    # Add batch dimension (1, 3, H, W)
#'
#' # DeepLabV3 with ResNet-50
#' model <- model_deeplabv3_resnet50(pretrained = TRUE)
#' model$eval()
#' output <- model(batch)
#'
#' # visualize the result
#' # `draw_segmentation_masks()` turns the torch_float output into a boolean mask internaly:
#' segmented <- draw_segmentation_masks(input, output$out$squeeze(1))
#' tensor_image_display(segmented)
#'
#' # Show most frequent class
#' mask_id <- output$out$argmax(dim = 2)  # (1, H, W)
#' class_contingency_with_background <- mask_id$view(-1)$bincount()
#' class_contingency_with_background[1] <- 0L # we clean the counter for background class id 1
#' top_class_index <- class_contingency_with_background$argmax()$item()
#' cli::cli_inform("Majority class {.pkg ResNet-50}: {.emph {model$classes[top_class_index]}}")
#'
#' # DeepLabV3 with ResNet-101 (same steps)
#' model <- model_deeplabv3_resnet101(pretrained = TRUE)
#' model$eval()
#' output <- model(batch)
#'
#' segmented <- draw_segmentation_masks(input, output$out$squeeze(1))
#' tensor_image_display(segmented)
#'
#' mask_id <- output$out$argmax(dim = 2)
#' class_contingency_with_background <- mask_id$view(-1)$bincount()
#' class_contingency_with_background[1] <- 0L # we clean the counter for background class id 1
#' top_class_index <- class_contingency_with_background$argmax()$item()
#' cli::cli_inform("Majority class {.pkg ResNet-101}: {.emph {model$classes[top_class_index]}}")
#' }
#' @name model_deeplabv3
#' @rdname model_deeplabv3
NULL

deeplabv3_model_urls <- list(
  deeplabv3_resnet50_coco = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/deeplabv3_resnet50_coco.pth",
    "9bc0953c76291fff296211b0eba04164",
    "168 MB"
  ),
  deeplabv3_resnet101_coco = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/deeplabv3_resnet101_coco.pth",
    "bb73d96704937621d782af73125e6282",
    "172 MB"
  )
)

voc_classes <- c(
  "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
  "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse",
  "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
)

deeplabv3_meta <- list(
  classes = voc_classes,
  class_to_idx = setNames(seq_along(voc_classes) - 1, voc_classes)
)


# ASPP module matching PyTorch
aspp_module <- torch::nn_module(
  "ASPP",
  initialize = function(in_channels, out_channels, atrous_rates) {
    self$convs <- torch::nn_module_list()

    self$convs$append(torch::nn_sequential(
      torch::nn_conv2d(in_channels, out_channels, kernel_size = 1, bias = FALSE),
      torch::nn_batch_norm2d(out_channels),
      torch::nn_relu()
    ))

    for (rate in atrous_rates) {
      self$convs$append(torch::nn_sequential(
        torch::nn_conv2d(in_channels, out_channels, kernel_size = 3, padding = rate, dilation = rate, bias = FALSE),
        torch::nn_batch_norm2d(out_channels),
        torch::nn_relu()
      ))
    }

    self$convs$append(torch::nn_sequential(
      torch::nn_adaptive_avg_pool2d(output_size = c(1, 1)),
      torch::nn_conv2d(in_channels, out_channels, kernel_size = 1, bias = FALSE),
      torch::nn_batch_norm2d(out_channels),
      torch::nn_relu()
    ))

    self$project <- torch::nn_sequential(
      torch::nn_conv2d((length(atrous_rates) + 2) * out_channels, out_channels, kernel_size = 1, bias = FALSE),
      torch::nn_batch_norm2d(out_channels),
      torch::nn_relu(),
      torch::nn_dropout(0.5)
    )
  },
  forward = function(x) {
    input_size <- x$shape[3:4]  # Get height and width dimensions
    res <- list()

    for (i in 1:(length(self$convs) - 1)) {
      res[[i]] <- self$convs[[i]](x)
    }

    global_feat <- self$convs[[length(self$convs)]](x)
    target_size <- as.integer(input_size)
    global_feat <- nnf_interpolate(global_feat, size = target_size, mode = "bilinear", align_corners = FALSE)
    res[[length(res) + 1]] <- global_feat

    x <- torch_cat(res, dim = 2)
    self$project(x)
  }
)

# Main classifier head
deeplab_head <- function(in_channels, num_classes) {
  torch::nn_sequential(
    aspp_module(in_channels, 256, atrous_rates = c(12, 24, 36)),
    torch::nn_conv2d(256, 256, kernel_size = 3, padding = 1, bias = FALSE),
    torch::nn_batch_norm2d(256),
    torch::nn_relu(),
    torch::nn_conv2d(256, num_classes, kernel_size = 1)
  )
}

# Auxiliary classifier head (simpler, operates on layer3 features)
aux_classifier_head <- function(in_channels, num_classes) {
  torch::nn_sequential(
    torch::nn_conv2d(in_channels, 256, kernel_size = 3, padding = 1, bias = FALSE),
    torch::nn_batch_norm2d(256),
    torch::nn_relu(),
    torch::nn_dropout(0.1),
    torch::nn_conv2d(256, num_classes, kernel_size = 1)
  )
}

# Updated DeepLabV3 wrapper with auxiliary classifier
DeepLabV3 <- torch::nn_module(
  "DeepLabV3",
  initialize = function(backbone, classifier, aux_classifier = NULL) {
    self$backbone <- backbone
    self$classifier <- classifier
    self$aux_classifier <- aux_classifier
  },
  forward = function(x) {
    input_shape <- x$shape[3:4]  # Get height and width dimensions

    x <- self$backbone$conv1(x)
    x <- self$backbone$bn1(x)
    x <- self$backbone$relu(x)
    x <- self$backbone$maxpool(x)
    x <- self$backbone$layer1(x)
    x <- self$backbone$layer2(x)
    x <- self$backbone$layer3(x)

    aux_x <- x

    x <- self$backbone$layer4(x)

    # Main classifier
    x <- self$classifier(x)
    target_size <- as.integer(input_shape)
    x <- nnf_interpolate(x, size = target_size, mode = "bilinear", align_corners = FALSE)

    result <- list(out = x)

    # Auxiliary classifier (if present)
    if (!is.null(self$aux_classifier)) {
      aux_out <- self$aux_classifier(aux_x)
      aux_out <- nnf_interpolate(aux_out, size = target_size, mode = "bilinear", align_corners = FALSE)
      result$aux <- aux_out
    }

    result
  }
)

# Model constructor
deeplabv3_resnet_factory <- function(arch, block, layers, pretrained, progress,
                                     num_classes, aux_loss = NULL,
                                     pretrained_backbone = FALSE, ...) {
  if (pretrained && num_classes != 21) {
    cli_abort("Pretrained weights require num_classes = 21.")
  }

  if (is.null(aux_loss))
    aux_loss <- FALSE

  if (pretrained && pretrained_backbone)
    cli_warn("`pretrained_backbone` ignored when `pretrained = TRUE`." )

  backbone_arch <- sub("deeplabv3_", "", arch)

  if (pretrained_backbone && !pretrained) {
    backbone <- .resnet(backbone_arch, block, layers, pretrained = TRUE,
                        progress = progress,
                        replace_stride_with_dilation = c(FALSE, TRUE, TRUE), ...)
  } else {
    backbone <- resnet(block, layers,
                       replace_stride_with_dilation = c(FALSE, TRUE, TRUE), ...)
  }
  backbone$fc <- torch::nn_identity()
  backbone$avgpool <- torch::nn_identity()

  classifier <- deeplab_head(2048, num_classes)

  aux_classifier <- NULL
  if (aux_loss) {
    aux_classifier <- aux_classifier_head(1024, num_classes)
  }

  model <- DeepLabV3(backbone, classifier, aux_classifier)
  model$classes <- deeplabv3_meta$classes
  model$class_to_idx <- deeplabv3_meta$class_to_idx

  if (pretrained) {
    info <- deeplabv3_model_urls[[paste0(arch, "_coco")]]
    cli_inform("Downloading {.cls {arch}} pretrained weights (~{.emph {info[3]}}) ...")
    state_dict_path <- download_and_cache(info[1])
    if (tools::md5sum(state_dict_path) != info[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }
  model
}

#' @describeIn model_deeplabv3 DeepLabV3 with ResNet-50 backbone
#' @export
model_deeplabv3_resnet50 <- function(pretrained = FALSE, progress = TRUE,
                                     num_classes = 21, aux_loss = NULL,
                                     pretrained_backbone = FALSE, ...) {
  deeplabv3_resnet_factory(
    "deeplabv3_resnet50", bottleneck, c(3, 4, 6, 3),
    pretrained, progress, num_classes, aux_loss, pretrained_backbone, ...
  )
}

#' @describeIn model_deeplabv3 DeepLabV3 with ResNet-101 backbone
#' @export
model_deeplabv3_resnet101 <- function(pretrained = FALSE, progress = TRUE,
                                      num_classes = 21, aux_loss = NULL,
                                      pretrained_backbone = FALSE, ...) {
  deeplabv3_resnet_factory(
    "deeplabv3_resnet101", bottleneck, c(3, 4, 23, 3),
    pretrained, progress, num_classes, aux_loss, pretrained_backbone, ...
  )
}
