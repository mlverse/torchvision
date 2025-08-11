facenet_torchscript_urls <- list(
  ONet = c("https://torch-cdn.mlverse.org/models/vision/v2/models/facenet_onet.pth", "4833e21e3a610b4064b5a3d20683a55f", "2 MB"),
  PNet = c("https://torch-cdn.mlverse.org/models/vision/v2/models/facenet_pnet.pth", "7f59c98ccf07c4ed51caf68fde86373e", "30 KB"),
  RNet = c("https://torch-cdn.mlverse.org/models/vision/v2/models/facenet_rnet.pth", "c19b2f0df8f448455dd7ddbb47dcfa19", "400 KB")
)

# PNet definition
pnet <- nn_module(
  classname = "PNet",
  initialize = function(pretrained=TRUE) {
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
    list(bbox_reg = b, cls = a)
  }
)

# RNet definition
rnet <- nn_module(
  classname = "RNet",
  initialize = function(pretrained=TRUE) {
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
    list(bbox_reg = b, cls = a)
  }
)

# ONet definition
onet <- nn_module(
  classname = "ONet",
  initialize = function(pretrained=TRUE) {
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
    list(bbox_reg = b, landmarks = c, cls = a)
  }
)

# MTCNN combined model wrapper
mtcnn <- nn_module(
  classname = "MTCNN",
  initialize = function(image_size=160, margin=0, min_face_size=20,
                        thresholds=c(0.6, 0.7, 0.7), factor=0.709,
                        post_process=TRUE, select_largest=TRUE, selection_method=NULL,
                        keep_all=FALSE, device=NULL,pretrained=TRUE) {
    self$image_size <- image_size
    self$margin <- margin
    self$min_face_size <- min_face_size
    self$thresholds <- thresholds
    self$factor <- factor
    self$post_process <- post_process
    self$select_largest <- select_largest
    self$keep_all <- keep_all
    self$selection_method <- selection_method
    
    self$pnet <- pnet(pretrained=pretrained)
    self$rnet <- rnet(pretrained=pretrained)
    self$onet <- onet(pretrained=pretrained)
    
    self$device <- torch_device("cpu")
    if (!is.null(device)) {
      self$device <- device
      self$to(device)
    }
    
    if (is.null(self$selection_method)) {
      self$selection_method <- ifelse(self$select_largest, "largest", "probability")
    }
  },
  
  forward = function(x) {
    x <- fixed_image_standardization(x)
    pnet_out <- self$pnet(x)
    x_rnet <- nnf_interpolate(x, size = c(24, 24), mode = "bilinear", align_corners = FALSE)
    rnet_out <- self$rnet(x_rnet)
    x_onet <- nnf_interpolate(x_rnet, size = c(48, 48), mode = "bilinear", align_corners = FALSE)
    onet_out <- self$onet(x_onet)
    
    list(bbox_reg = onet_out$bbox_reg, landmarks = onet_out$landmarks, cls = onet_out$cls)
  }
)

# Utility function for standardization (matching fixed_image_standardization)
fixed_image_standardization <- function(image_tensor) {
  (image_tensor - 127.5) / 128.0
}

# Utility function for prewhiten
prewhiten <- function(x) {
  mean_val <- x$mean()
  std_val <- x$std()
  std_adj <- torch_clamp(std_val, min = 1.0 / (x$numel() ^ 0.5))
  (x - mean_val) / std_adj
}
