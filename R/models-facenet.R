facenet_torchscript_urls <- list(
  ONet = c("https://torch-cdn.mlverse.org/models/vision/v2/models/facenet_onet.pth", "4833e21e3a610b4064b5a3d20683a55f", "2 MB"),
  PNet = c("https://torch-cdn.mlverse.org/models/vision/v2/models/facenet_pnet.pth", "7f59c98ccf07c4ed51caf68fde86373e", "30 KB"),
  RNet = c("https://torch-cdn.mlverse.org/models/vision/v2/models/facenet_rnet.pth", "c19b2f0df8f448455dd7ddbb47dcfa19", "400 KB")
)

#' @describeIn model_facenet PNet (Proposal Network) — small fully-convolutional network for candidate face box generation.
#' @export
model_facenet_pnet <- nn_module(
  classname = "PNet",
  initialize = function(pretrained=TRUE,progress=FALSE,...) {
    self$conv1 <- nn_conv2d(3, 10, kernel_size=3)
    self$prelu1 <- nn_prelu(10)
    self$pool1 <- nn_max_pool2d(2, 2, ceil_mode=TRUE)
    self$conv2 <- nn_conv2d(10, 16, kernel_size=3)
    self$prelu2 <- nn_prelu(16)
    self$conv3 <- nn_conv2d(16, 32, kernel_size=3)
    self$prelu3 <- nn_prelu(32)
    self$conv4_1 <- nn_conv2d(32, 2, kernel_size=1)
    self$softmax4_1 <- nn_softmax(dim=1)
    self$conv4_2 <- nn_conv2d(32, 4, kernel_size=1)
    
    self$training <- FALSE
    
    if (pretrained) {
      archive <- download_and_cache(facenet_torchscript_urls$PNet[1], prefix = "pnet")
      if (tools::md5sum(archive) != facenet_torchscript_urls$PNet[2]){
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")
      }
      state_dict <- load_state_dict(archive)
      self$load_state_dict(state_dict)
    }
  },
  
  forward = function(x) {
    x <- self$conv1(x)
    x <- self$prelu1(x)
    x <- self$pool1(x)
    x <- self$conv2(x)
    x <- self$prelu2(x)
    x <- self$conv3(x)
    x <- self$prelu3(x)
    a <- self$conv4_1(x)
    a <- self$softmax4_1(a)
    b <- self$conv4_2(x)
    list(boxes = b, cls = a)
  }
)

#' @describeIn model_facenet RNet (Refine Network) — medium CNN with dense layers for refining and rejecting false positives.
#' @export
model_facenet_rnet <- nn_module(
  classname = "RNet",
  initialize = function(pretrained=TRUE,progress=FALSE,...) {
    self$conv1 <- nn_conv2d(3, 28, kernel_size=3)
    self$prelu1 <- nn_prelu(28)
    self$pool1 <- nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv2 <- nn_conv2d(28, 48, kernel_size=3)
    self$prelu2 <- nn_prelu(48)
    self$pool2 <- nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv3 <- nn_conv2d(48, 64, kernel_size=2)
    self$prelu3 <- nn_prelu(64)
    self$dense4 <- nn_linear(576, 128)
    self$prelu4 <- nn_prelu(128)
    self$dense5_1 <- nn_linear(128, 2)
    self$softmax5_1 <- nn_softmax(dim=1)
    self$dense5_2 <- nn_linear(128, 4)
    
    self$training <- FALSE
    
    if (pretrained) {
      archive <- download_and_cache(facenet_torchscript_urls$RNet[1], prefix = "rnet")
      if (tools::md5sum(archive) != facenet_torchscript_urls$RNet[2]){
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")
      }
      state_dict <- load_state_dict(archive)
      self$load_state_dict(state_dict)
    }
  },
  
  forward = function(x) {
    x <- self$conv1(x)
    x <- self$prelu1(x)
    x <- self$pool1(x)
    x <- self$conv2(x)
    x <- self$prelu2(x)
    x <- self$pool2(x)
    x <- self$conv3(x)
    x <- self$prelu3(x)
    x <- x$permute(c(1,4,3,2))$contiguous()
    x <- self$dense4(x$view(c(x$size(1), -1)))
    x <- self$prelu4(x)
    a <- self$dense5_1(x)
    a <- self$softmax5_1(a)
    b <- self$dense5_2(x)
    list(boxes = b, cls = a)
  }
)

#' @describeIn model_facenet ONet (Output Network) — deeper CNN that outputs final bounding boxes and 5 facial landmark points.
#' @export
model_facenet_onet <- nn_module(
  classname = "ONet",
  initialize = function(pretrained=TRUE,progress=FALSE,...) {
    self$conv1 <- nn_conv2d(3, 32, kernel_size=3)
    self$prelu1 <- nn_prelu(32)
    self$pool1 <- nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv2 <- nn_conv2d(32, 64, kernel_size=3)
    self$prelu2 <- nn_prelu(64)
    self$pool2 <- nn_max_pool2d(3, 2, ceil_mode=TRUE)
    self$conv3 <- nn_conv2d(64, 64, kernel_size=3)
    self$prelu3 <- nn_prelu(64)
    self$pool3 <- nn_max_pool2d(2, 2, ceil_mode=TRUE)
    self$conv4 <- nn_conv2d(64, 128, kernel_size=2)
    self$prelu4 <- nn_prelu(128)
    self$dense5 <- nn_linear(1152, 256)
    self$prelu5 <- nn_prelu(256)
    self$dense6_1 <- nn_linear(256, 2)
    self$softmax6_1 <- nn_softmax(dim=1)
    self$dense6_2 <- nn_linear(256, 4)
    self$dense6_3 <- nn_linear(256, 10)
    
    self$training <- FALSE
    
    if (pretrained) {
      archive <- download_and_cache(facenet_torchscript_urls$ONet[1], prefix = "onet")
      if (tools::md5sum(archive) != facenet_torchscript_urls$ONet[2]){
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")
      }
      state_dict <- load_state_dict(archive)
      self$load_state_dict(state_dict)
    }
  },
  
  forward = function(x) {
    x <- self$conv1(x)
    x <- self$prelu1(x)
    x <- self$pool1(x)
    x <- self$conv2(x)
    x <- self$prelu2(x)
    x <- self$pool2(x)
    x <- self$conv3(x)
    x <- self$prelu3(x)
    x <- self$pool3(x)
    x <- self$conv4(x)
    x <- self$prelu4(x)
    x <- x$permute(c(1,4,3,2))$contiguous()
    x <- self$dense5(x$view(c(x$size(1), -1)))
    x <- self$prelu5(x)
    a <- self$dense6_1(x)
    a <- self$softmax6_1(a)
    b <- self$dense6_2(x)
    c <- self$dense6_3(x)
    list(boxes = b, landmarks = c, cls = a)
  }
)

#' MTCNN Face Detection Networks
#'
#' These models implement the three-stage Multi-task Cascaded Convolutional Networks (MTCNN)
#' architecture from the paper 
#' [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).
#' 
#' MTCNN detects faces and facial landmarks in an image through a coarse-to-fine pipeline:
#' - **PNet** (Proposal Network): Generates candidate face bounding boxes at multiple scales.
#' - **RNet** (Refine Network): Refines candidate boxes, rejecting false positives.
#' - **ONet** (Output Network): Produces final bounding boxes and 5-point facial landmarks.
#'
#' ## Model Variants
#' ```
#' | Model | Input Size     | Parameters | File Size | Outputs                       | Notes                             |
#' |-------|----------------|------------|-----------|-------------------------------|-----------------------------------|
#' | PNet  | ~12×12+        | ~3K        | 30 KB     | 2-class face prob + bbox reg  | Fully conv, sliding window stage  |
#' | RNet  | 24×24          | ~30K       | 400 KB    | 2-class face prob + bbox reg  | Dense layers, higher recall       |
#' | ONet  | 48×48          | ~100K      | 2 MB      | 2-class prob + bbox + 5-point | Landmark detection stage          |
#' ```
#'
#' @examples
#' \dontrun{
#' # Example usage of PNet
#' model_pnet <- model_facenet_pnet(pretrained = TRUE)
#' model_pnet$eval()
#' input_pnet <- torch_randn(1, 3, 160, 160)
#' output_pnet <- model_pnet(input_pnet)
#' output_pnet
#'
#' # Example usage of RNet
#' model_rnet <- model_facenet_rnet(pretrained = TRUE)
#' model_rnet$eval()
#' input_rnet <- torch_randn(1, 3, 24, 24)
#' output_rnet <- model_rnet(input_rnet)
#' output_rnet
#'
#' # Example usage of ONet
#' model_onet <- model_facenet_onet(pretrained = TRUE)
#' model_onet$eval()
#' input_onet <- torch_randn(1, 3, 48, 48)
#' output_onet <- model_onet(input_onet)
#' output_onet
#'
#' # Example usage of MTCNN
#' mtcnn <- model_mtcnn(pretrained = TRUE)
#' mtcnn$eval()
#' image_tensor <- torch_randn(c(1, 3, 160, 160))
#' out <- mtcnn(image_tensor)
#' out
#' }
#'
#' @inheritParams model_mobilenet_v2
#'
#' @family models
#' @rdname model_facenet
#' @name model_facenet
NULL

#' @describeIn model_facenet MTCNN (Multi-task Cascaded Convolutional Networks) — face detection and alignment using a cascade of three neural networks
#' @export
model_mtcnn <- nn_module(
  classname = "MTCNN",
  initialize = function(
    pretrained = TRUE,
    progress = TRUE,
    ...
  ) {

    
    self$pnet <- model_facenet_pnet(pretrained=pretrained,...)
    self$rnet <- model_facenet_rnet(pretrained=pretrained,...)
    self$onet <- model_facenet_onet(pretrained=pretrained,...)
  },
  
  forward = function(x) {
    pnet_out <- self$pnet(x)
    x_rnet <- nnf_interpolate(x, size = c(24, 24), mode = "bilinear", align_corners = FALSE)
    rnet_out <- self$rnet(x_rnet)
    x_onet <- nnf_interpolate(x_rnet, size = c(48, 48), mode = "bilinear", align_corners = FALSE)
    onet_out <- self$onet(x_onet)
    
    list(boxes = onet_out$boxes, landmarks = onet_out$landmarks, cls = onet_out$cls)
  }
)

fixed_image_standardization <- function(image_tensor) {
  (image_tensor - 127.5) / 128.0
}

prewhiten <- function(x) {
  mean_val <- x$mean()
  std_val <- x$std()
  std_adj <- torch_clamp(std_val, min = 1.0 / (x$numel() ^ 0.5))
  (x - mean_val) / std_adj
}
