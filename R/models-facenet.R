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
#' Inception-ResNet-v1 is a convolutional neural network architecture combining Inception modules 
#' with residual connections, designed for face recognition tasks. The model achieves high accuracy 
#' on standard face verification benchmarks such as LFW (Labeled Faces in the Wild).
#'
#' ## Model Variants and Performance (LFW accuracy)
#' ```
#' |    Weights     | LFW Accuracy | File Size |
#' |----------------|--------------|-----------|
#' | CASIA-Webface  | 99.05%       | 111 MB    |
#' | VGGFace2       | 99.65%       | 107 MB    |
#' ```
#'
#' - The CASIA-Webface pretrained weights provide strong baseline accuracy.
#' - The VGGFace2 pretrained weights achieve higher accuracy, benefiting from a larger, more diverse dataset.
#' 
#' @examples
#' \dontrun{
#' # Example usage of PNet
#' model_pnet <- model_facenet_pnet(pretrained = TRUE)
#' model_pnet$eval()
#' input_pnet <- torch_randn(1, 3, 224, 224)
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
#' image_tensor <- torch_randn(c(1, 3, 224, 224))
#' out <- mtcnn(image_tensor)
#' out
#'
#' # Example usage of Inception-ResNet-v1 with VGGFace2 Weights
#' model <- model_inception_resnet_v1(pretrained = "vggface2")
#' model$eval()
#' input <- torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' output
#'
#' # Example usage of Inception-ResNet-v1 with CASIA-Webface Weights
#' model <- model_inception_resnet_v1(pretrained = "casia-webface")
#' model$eval()
#' input <- torch_randn(1, 3, 224, 224)
#' output <- model(input)
#' output
#' }
#'
#' @inheritParams model_mobilenet_v2
#' @param classify Logical, whether to include the classification head. Default is FALSE.
#' @param num_classes Integer, number of output classes for classification. Default is 10.
#' @param dropout_prob Numeric, dropout probability applied before classification. Default is 0.6.
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


load_inception_weights <- function(model, name) {
  if (name == "vggface2") {
    url <- "https://torch-cdn.mlverse.org/models/vision/v2/models/vggface2.pth"
    md5 = "c446a04f0b22763858226717ba1f7410"
  } else if (name == "casia-webface") {
    url <- "https://torch-cdn.mlverse.org/models/vision/v2/models/casia-webface.pth"
    md5 = "ff4aff482f6c1941784abba5131bae20"
  }

  archive <- download_and_cache(url,prefix = name)
  if (tools::md5sum(archive) != md5){
    runtime_error("Corrupt file! Delete the file in {archive} and try again.")
  }

  state_dict <- torch::load_state_dict(archive)
  model$load_state_dict(state_dict)
  model
}

BasicConv2d <- nn_module(
  "BasicConv2d",
  initialize = function(in_channels, out_channels, kernel_size, stride, padding = 0) {
    self$conv <- nn_conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = FALSE)
    self$bn <- nn_batch_norm2d(out_channels, eps = 0.001, momentum = 0.1)
  },
  forward = function(x) {
    x %>% self$conv() %>% self$bn() %>% nnf_relu(inplace = TRUE)
  }
)

Block35 <- nn_module(
  "Block35",
  initialize = function(scale = 1.0) {
    self$scale <- scale
    self$branch0 <- BasicConv2d(256, 32, kernel_size = 1, stride = 1)
    self$branch1 <- nn_sequential(
      BasicConv2d(256, 32, kernel_size = 1, stride = 1),
      BasicConv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)
    )
    self$branch2 <- nn_sequential(
      BasicConv2d(256, 32, kernel_size = 1, stride = 1),
      BasicConv2d(32, 48, kernel_size = 3, stride = 1, padding = 1),
      BasicConv2d(48, 64, kernel_size = 3, stride = 1, padding = 1)
    )
    self$conv2d <- nn_conv2d(128, 256, kernel_size = 1, stride = 1)
  },
  forward = function(x) {
    branch0 <- self$branch0(x)
    branch1 <- self$branch1(x)
    branch2 <- self$branch2(x)
    print(branch0$shape)  
    print(branch1$shape) 
    print(branch2$shape) 
    mixed <- torch_cat(list(branch0, branch1, branch2), dim = 2)
    up <- self$conv2d(mixed)
    x + self$scale * up %>% nnf_relu(inplace = TRUE)
  }
)

Block17 <- nn_module(
  initialize = function(scale = 1.0) {
    self$scale <- scale
    self$branch0 <- BasicConv2d(896, 128, kernel_size = 1, stride = 1)
    self$branch1 <- nn_sequential(
      BasicConv2d(896, 128, kernel_size = 1, stride = 1),
      BasicConv2d(128, 128, kernel_size = c(1,7), stride = 1, padding = c(0,3)),
      BasicConv2d(128, 128, kernel_size = c(7,1), stride = 1, padding = c(3,0))
    )
    self$conv2d <- nn_conv2d(256, 896, kernel_size = 1, stride = 1)
    self$relu <- nn_relu(inplace = FALSE)
  },
  forward = function(x) {
    x0 <- self$branch0(x)
    x1 <- self$branch1(x)
    out <- torch_cat(list(x0, x1), dim = 2)
    out <- self$conv2d(out)
    out <- out * self$scale + x
    out %>% self$relu()
  }
)

Block8 <- nn_module(
  initialize = function(scale = 1.0, noReLU = FALSE) {
    self$scale <- scale
    self$noReLU <- noReLU
    self$branch0 <- BasicConv2d(1792, 192, kernel_size = 1, stride = 1)
    self$branch1 <- nn_sequential(
      BasicConv2d(1792, 192, kernel_size = 1, stride = 1),
      BasicConv2d(192, 192, kernel_size = c(1,3), stride = 1, padding = c(0,1)),
      BasicConv2d(192, 192, kernel_size = c(3,1), stride = 1, padding = c(1,0))
    )
    self$conv2d <- nn_conv2d(384, 1792, kernel_size = 1, stride = 1)
    if (!noReLU) {
      self$relu <- nn_relu(inplace = FALSE)
    }
  },
  forward = function(x) {
    x0 <- self$branch0(x)
    x1 <- self$branch1(x)
    out <- torch_cat(list(x0, x1), dim = 2)
    out <- self$conv2d(out)
    out <- out * self$scale + x
    if (!self$noReLU) {
      out <- self$relu(out)
    }
    out
  }
)

Mixed_6a <- nn_module(
  initialize = function() {
    self$branch0 <- BasicConv2d(256, 384, kernel_size = 3, stride = 2, padding = 0)
    self$branch1 <- nn_sequential(
      BasicConv2d(256, 192, kernel_size = 1, stride = 1, padding = 0),
      BasicConv2d(192, 192, kernel_size = 3, stride = 1, padding = 0),
      BasicConv2d(192, 256, kernel_size = 3, stride = 2, padding = 1)
    )
    self$branch2 <- nn_max_pool2d(kernel_size = 3, stride = 2)
  },
  forward = function(x) {
    x0 <- self$branch0(x)
    x1 <- self$branch1(x)
    x2 <- self$branch2(x)

    torch_cat(list(x0, x1, x2), dim = 2)
  }
)

Mixed_7a <- nn_module(
  initialize = function() {
    self$branch0 <- nn_sequential(
      BasicConv2d(896, 256, kernel_size = 1, stride = 1),
      BasicConv2d(256, 384, kernel_size = 3, stride = 2)
    )
    self$branch1 <- nn_sequential(
      BasicConv2d(896, 256, kernel_size = 1, stride = 1),
      BasicConv2d(256, 256, kernel_size = 3, stride = 2)
    )
    self$branch2 <- nn_sequential(
      BasicConv2d(896, 256, kernel_size = 1, stride = 1),
      BasicConv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
      BasicConv2d(256, 256, kernel_size = 3, stride = 2)
    )
    self$branch3 <- nn_max_pool2d(kernel_size = 3, stride = 2)
  },
  forward = function(x) {
    x0 <- self$branch0(x)
    x1 <- self$branch1(x)
    x2 <- self$branch2(x)
    x3 <- self$branch3(x)

    torch_cat(list(x0, x1, x2, x3), dim = 2)
  }
)

#' @describeIn model_facenet Inception-ResNet-v1 — high-accuracy face recognition model combining Inception modules with residual connections, pretrained on VGGFace2 and CASIA-Webface datasets
#' @export
model_inception_resnet_v1 <- nn_module(
  initialize = function(
    pretrained = NULL,
    classify = FALSE,
    num_classes = 10,
    dropout_prob = 0.6,
    ...
  ) {

    if (!is.null(pretrained)) {
      if (pretrained == "vggface2") {
        tmp_classes <- 8631
      } else if (pretrained == "casia-webface") {
        tmp_classes <- 10575
      } else {
        pretrained <- NULL
      }
    }
    
    self$conv2d_1a <- BasicConv2d(3, 32, kernel_size = 3, stride = 2)
    self$conv2d_2a <- BasicConv2d(32, 32, kernel_size = 3, stride = 1)
    self$conv2d_2b <- BasicConv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
    self$maxpool_3a <- nn_max_pool2d(kernel_size = 3, stride = 2)
    self$conv2d_3b <- BasicConv2d(64, 80, kernel_size = 1, stride = 1)
    self$conv2d_4a <- BasicConv2d(80, 192, kernel_size = 3, stride = 1)
    self$conv2d_4b <- BasicConv2d(192, 256, kernel_size = 3, stride = 2)
    
    self$mixed_6a <- Mixed_6a()
    self$repeat_2 <- nn_sequential(!!!lapply(1:10, function(i) Block17(0.10)))
    self$mixed_7a <- Mixed_7a()
    self$repeat_3 <- nn_sequential(!!!lapply(1:5, function(i) Block8(0.20)))
    self$block8 <- Block8(noReLU = TRUE)
    
    self$avgpool_1a <- nn_adaptive_avg_pool2d(output_size = 1)
    self$dropout <- nn_dropout(p = dropout_prob)
    self$last_linear <- nn_linear(1792, 512, bias = FALSE)
    self$last_bn <- nn_batch_norm1d(512, eps = 0.001, momentum = 0.1, affine = TRUE)
    
    self$classify <- classify
    if (!is.null(pretrained)) {
      self$logits <- nn_linear(512, tmp_classes)
      load_inception_weights(self, pretrained)
    }
    if (classify && !is.null(num_classes)) {
      self$logits <- nn_linear(512, num_classes)
    }
  },
  forward = function(x) {
    x <- self$conv2d_1a(x)
    x <- self$conv2d_2a(x)
    x <- self$conv2d_2b(x)
    x <- self$maxpool_3a(x)
    x <- self$conv2d_3b(x)
    x <- self$conv2d_4a(x)
    x <- self$conv2d_4b(x)
    x <- self$mixed_6a(x)
    x <- self$repeat_2(x)
    x <- self$mixed_7a(x)
    x <- self$repeat_3(x)
    x <- self$block8(x)
    x <- self$avgpool_1a(x)
    x <- self$dropout(x)
    x <- self$last_linear(x$view(c(x$shape[1], -1)))
    x <- self$last_bn(x)

    if (self$classify) {
      x <- self$logits(x)
    } else {
      x <- nnf_normalize(x, p = 2, dim = 2)
    }
    x
  }
)
