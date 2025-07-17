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
#' @param ... Other parameters passed to the model implementation.
#'
#' @family models
#'
#' @examples
#' \dontrun{
#'   model <- model_deeplabv3_resnet50(num_classes = 21)
#'   input <- torch::torch_randn(1, 3, 64, 64)
#'   out <- model(input)
#'   names(out)
#'
#'   model <- model_deeplabv3_resnet101(num_classes = 21)
#'   input <- torch::torch_randn(1, 3, 64, 64)
#'   out <- model(input)
#'   names(out)
#' }
#' @name model_deeplabv3
#' @rdname model_deeplabv3
NULL

deeplabv3_model_urls <- c(
  "deeplabv3_resnet50_coco" = "https://torch-cdn.mlverse.org/models/vision/v1/models/deeplabv3_resnet50_coco.pth",
  "deeplabv3_resnet101_coco" = "https://torch-cdn.mlverse.org/models/vision/v1/models/deeplabv3_resnet101_coco.pth"
)

# ASPP module matching PyTorch
aspp_module <- nn_module(
  "ASPP",
  initialize = function(in_channels, out_channels, atrous_rates) {
    self$convs <- nn_module_list()

    self$convs$append(nn_sequential(
      nn_conv2d(in_channels, out_channels, kernel_size = 1, bias = FALSE),
      nn_batch_norm2d(out_channels),
      nn_relu()
    ))

    for (rate in atrous_rates) {
      self$convs$append(nn_sequential(
        nn_conv2d(in_channels, out_channels, kernel_size = 3, padding = rate, dilation = rate, bias = FALSE),
        nn_batch_norm2d(out_channels),
        nn_relu()
      ))
    }

    self$convs$append(nn_sequential(
      nn_adaptive_avg_pool2d(output_size = c(1, 1)),
      nn_conv2d(in_channels, out_channels, kernel_size = 1, bias = FALSE),
      nn_batch_norm2d(out_channels),
      nn_relu()
    ))

    self$project <- nn_sequential(
      nn_conv2d((length(atrous_rates) + 2) * out_channels, out_channels, kernel_size = 1, bias = FALSE),
      nn_batch_norm2d(out_channels),
      nn_relu(),
      nn_dropout(0.5)
    )
  },
  forward = function(x) {
    # Extract spatial dimensions (height, width) from input shape [batch, channels, height, width]
    input_size <- x$shape[3:4]  # Get height and width dimensions
    res <- list()

    # Process all convolutions except the last one (global pooling)
    for (i in 1:(length(self$convs) - 1)) {
      res[[i]] <- self$convs[[i]](x)
    }

    # Handle global pooling separately
    global_feat <- self$convs[[length(self$convs)]](x)
    # Use as.integer to ensure proper format
    target_size <- as.integer(input_size)
    global_feat <- nnf_interpolate(global_feat, size = target_size, mode = "bilinear", align_corners = FALSE)
    res[[length(res) + 1]] <- global_feat

    x <- torch_cat(res, dim = 2)
    self$project(x)
  }
)

# Main classifier head
deeplab_head <- function(in_channels, num_classes) {
  nn_sequential(
    # classifier.0 = ASPP
    aspp_module(in_channels, 256, atrous_rates = c(12, 24, 36)),

    # classifier.1 = conv3x3
    nn_conv2d(256, 256, kernel_size = 3, padding = 1, bias = FALSE),

    # classifier.2 = batch norm
    nn_batch_norm2d(256),

    # classifier.3 = relu
    nn_relu(),

    # classifier.4 = final 1x1 conv
    nn_conv2d(256, num_classes, kernel_size = 1)
  )
}

# Auxiliary classifier head (simpler, operates on layer3 features)
aux_classifier_head <- function(in_channels, num_classes) {
  nn_sequential(
    # aux_classifier.0 = 3x3 conv
    nn_conv2d(in_channels, 256, kernel_size = 3, padding = 1, bias = FALSE),

    # aux_classifier.1 = batch norm
    nn_batch_norm2d(256),

    # aux_classifier.2 = relu
    nn_relu(),

    # aux_classifier.3 = dropout
    nn_dropout(0.1),

    # aux_classifier.4 = final 1x1 conv
    nn_conv2d(256, num_classes, kernel_size = 1)
  )
}

# Updated DeepLabV3 wrapper with auxiliary classifier
DeepLabV3 <- nn_module(
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

    # Store layer3 output for auxiliary classifier
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

# Updated model constructor
deeplabv3_resnet_factory <- function(arch, block, layers, pretrained, progress, num_classes, ...) {
  if (pretrained && num_classes != 21) {
    cli::cli_abort("Pretrained weights require num_classes = 21.")
  }

  backbone <- resnet(block, layers, replace_stride_with_dilation = c(FALSE, TRUE, TRUE), ...)
  backbone$fc <- nn_identity()
  backbone$avgpool <- nn_identity()

  classifier <- deeplab_head(2048, num_classes)

  # Add auxiliary classifier (operates on layer3 features, which has 1024 channels for ResNet50/101)
  aux_classifier <- aux_classifier_head(1024, num_classes)

  model <- DeepLabV3(backbone, classifier, aux_classifier)

  if (pretrained) {
    state_dict_path <- download_and_cache(deeplabv3_model_urls[[paste0(arch, "_coco")]])
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}

#' @describeIn model_deeplabv3 DeepLabV3 with ResNet-50 backbone
#' @export
model_deeplabv3_resnet50 <- function(pretrained = FALSE, progress = TRUE,
                                     num_classes = 21, ...) {
  deeplabv3_resnet_factory("deeplabv3_resnet50", bottleneck, c(3, 4, 6, 3),
                           pretrained, progress, num_classes, ...)
}

#' @describeIn model_deeplabv3 DeepLabV3 with ResNet-101 backbone
#' @export
model_deeplabv3_resnet101 <- function(pretrained = FALSE, progress = TRUE,
                                      num_classes = 21, ...) {
  deeplabv3_resnet_factory("deeplabv3_resnet101", bottleneck, c(3, 4, 23, 3),
                           pretrained, progress, num_classes, ...)
}
