
resnet_model_urls <- list(
  "resnet18" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/resnet18.pth", "b36d2ea3a4dbf3fbcb27552a410054f9", "~45 MB"),
  "resnet34" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/resnet34.pth", "cfd906d8f35b0256e1066ce970f65c50","~85 MB"),
  "resnet50" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/resnet50.pth", "9ee57c8355a4ad3759c3e91b0d9f6144", "~100 MB"),
  "resnet101" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/resnet101.pth", "0898550c0f6156d18d83fc7454760827", "~170 MB"),
  "resnet152" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/resnet152.pth", "cfe6cf55d97bdd838fe7c892c30788ef", "~230 MB"),
  "resnext50_32x4d" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/resnext50_32x4d.pth", "13f1b58b0694b634e43cf55e7208abc0", "~95 MB"),
  "resnext101_32x8d" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/resnext101_32x8d.pth", "c79668937f01117ca74193a71b31557b", "~340 MB"),
  "wide_resnet50_2" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/wide_resnet50_2.pth", "b2d17fb9c5929ab96becf63b1403423d", "~130 MB"),
  "wide_resnet101_2" = c("https://torch-cdn.mlverse.org/models/vision/v2/models/wide_resnet101_2.pth","7ccd7583b94e3f5fa0f231b4c134760e", "~240 MB")
)

conv_3x3 <- function(in_planes, out_planes, stride = 1, groups = 1, dilation = 1) {
  torch::nn_conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
            padding = dilation, groups = groups, bias = FALSE,
            dilation = dilation)
}

conv_1x1 <- function(in_planes, out_planes, stride=1) {
  torch::nn_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=FALSE)
}

basic_block <- torch::nn_module(
  "basic_block",
  expansion = 1,
  initialize = function(inplanes, planes, stride=1, downsample=NULL, groups=1,
                        base_width=64, dilation=1, norm_layer=NULL) {

    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    if (groups != 1 || base_width != 64)
      value_error("basic_block only supports groups=1 and base_width=64")

    if (dilation > 1)
      not_implemented_error("Dilation > 1 not supported in basic_block")

    self$conv1 <- conv_3x3(inplanes, planes, stride)
    self$bn1 <- norm_layer(planes)
    self$relu <- torch::nn_relu(inplace = TRUE)
    self$conv2 <- conv_3x3(planes, planes)
    self$bn2 <- norm_layer(planes)
    self$downsample <- downsample
    self$stride <- stride

  },
  forward = function(x) {

    out <- self$conv1(x)
    out <- self$bn1(out)
    out <- self$relu(out)

    out <- self$conv2(out)
    out <- self$bn2(out)

    if (!is.null(self$downsample))
      identity <- self$downsample(x)
    else
      identity <- x

    out$add_(identity)
    out <- self$relu(out)

    out
  }
)

# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
# while original implementation places the stride at the first 1x1 convolution(self.conv1)
# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
# This variant is also known as ResNet V1.5 and improves accuracy according to
# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
bottleneck <- torch::nn_module(
  "bottleneck",
  expansion = 4,
  initialize = function(inplanes, planes, stride=1, downsample=NULL, groups=1,
                        base_width=64, dilation=1, norm_layer=NULL) {

    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    width <- as.integer(planes * (base_width / 64)) * groups

    self$conv1 <- conv_1x1(inplanes, width)
    self$bn1 <- norm_layer(width)

    self$conv2 <- conv_3x3(width, width, stride, groups, dilation)
    self$bn2 <- norm_layer(width)

    self$conv3 <- conv_1x1(width, planes * self$expansion)
    self$bn3 <- norm_layer(planes * self$expansion)

    self$relu <- torch::nn_relu(inplace = TRUE)
    self$downsample <- downsample
    self$stride <- stride

  },
  forward = function(x) {

    out <- self$conv1(x)
    out <- self$bn1(out)
    out <- self$relu(out)

    out <- self$conv2(out)
    out <- self$bn2(out)
    out <- self$relu(out)

    out <- self$conv3(out)
    out <- self$bn3(out)

    if (!is.null(self$downsample))
      identity <- self$downsample(x)
    else
      identity <- x

    out$add_(identity)
    out <- self$relu(out)

    out
  }
)

resnet <- torch::nn_module(
  "resnet",
  initialize = function(block, layers, num_classes=1000, zero_init_residual=FALSE,
                        groups=1, width_per_group=64, replace_stride_with_dilation=NULL,
                        norm_layer=NULL) {

    if (is.null(norm_layer))
      norm_layer <- torch::nn_batch_norm2d

    self$.norm_layer <- norm_layer

    self$inplanes <- 64
    self$dilation <- 1

    # each element in the tuple indicates if we should replace
    # the 2x2 stride with a dilated convolution instead
    if (is.null(replace_stride_with_dilation))
      replace_stride_with_dilation <- rep(FALSE, 3)

    if (length(replace_stride_with_dilation) != 3)
      value_error(
        "replace_stride_with_dilation should be NULL ",
        "or a 3-element tuple, got {length(replace_stride_with_dilation)}"
      )

    self$groups <- groups
    self$base_width <- width_per_group
    self$conv1 <- torch::nn_conv2d(3, self$inplanes, kernel_size=7, stride=2, padding=3,
                           bias=FALSE)
    self$bn1 <- norm_layer(self$inplanes)
    self$relu <- torch::nn_relu(inplace=TRUE)
    self$maxpool <- torch::nn_max_pool2d(kernel_size=3, stride=2, padding=1)
    self$layer1 <- self$.make_layer(block, 64, layers[1])
    self$layer2 <- self$.make_layer(block, 128, layers[2], stride=2,
                                   dilate=replace_stride_with_dilation[1])
    self$layer3 <- self$.make_layer(block, 256, layers[3], stride=2,
                                   dilate=replace_stride_with_dilation[2])
    self$layer4 <- self$.make_layer(block, 512, layers[4], stride=2,
                                   dilate=replace_stride_with_dilation[3])
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
    self$fc <- torch::nn_linear(512 * block$public_fields$expansion, num_classes)

    for (m in private$modules_) {
      if (inherits(m, "nn_conv2d")) {
        torch::nn_init_kaiming_normal_(m$weight, mode=  "fan_out",
                                       nonlinearity = "relu")
      } else if (inherits(m, "nn_batch_norm2d") || inherits(m, "nn_group_norm")) {
        torch::nn_init_constant_(m$weight, 1)
        torch::nn_init_constant_(m$bias, 0)
      }
    }

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if (zero_init_residual) {
      for (m in private$modules_) {

        if (inherits(m, "bottleneck"))
          torch::nn_init_constant_(m$bn3$weight, 0)
        else if (inherits(m, "basic_block"))
          torch::nn_init_constant_(m$bn2$weight, 0)

      }
    }

  },
  .make_layer = function(block, planes, blocks, stride=1, dilate=FALSE) {

    norm_layer <- self$.norm_layer
    downsample <- NULL
    previous_dilation <- self$dilation

    if (dilate) {
      self$dilation <- self$dilation*stride
      stride <- 1
    }

    if (stride != 1 || self$inplanes != planes * block$public_fields$expansion) {
      downsample <- torch::nn_sequential(
        conv_1x1(self$inplanes, planes * block$public_fields$expansion, stride),
        norm_layer(planes * block$public_fields$expansion)
      )
    }

    layers <- list()
    layers[[1]] <- block(self$inplanes, planes, stride, downsample, self$groups,
            self$base_width, previous_dilation, norm_layer)
    self$inplanes <- planes * block$public_fields$expansion

    for (i in seq(from = 2, to = blocks)) {
      layers[[i]] <- block(self$inplanes, planes, groups=self$groups,
                           base_width=self$base_width, dilation=self$dilation,
                           norm_layer=norm_layer)
    }

    do.call(torch::nn_sequential, layers)
  },
  forward = function(x) {
    x <- self$conv1(x)
    x <- self$bn1(x)
    x <- self$relu(x)
    x <- self$maxpool(x)

    x <- self$layer1(x)
    x <- self$layer2(x)
    x <- self$layer3(x)
    x <- self$layer4(x)

    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$fc(x)
    x
  }
)

.resnet <- function(arch, block, layers, pretrained, progress, ...) {
  model <- resnet(block, layers, ...)

  if (pretrained) {
    r <- resnet_model_urls[[arch]]
    cli_inform("Model weights for {.cls {arch}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}

#' ResNet implementation
#'
#' ResNet models implementation from
#'   [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) and later
#'   related papers (see Functions)
#'
#' @param pretrained (bool): If TRUE, returns a model pre-trained on ImageNet.
#' @param progress (bool): If TRUE, displays a progress bar of the download to
#'   stderr.
#' @param ... Other parameters passed to the resnet model.
#'
#' @family classification_model
#' @name model_resnet
#' @rdname model_resnet
NULL

#' @describeIn model_resnet ResNet 18-layer model
#' @export
model_resnet18 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("resnet18", basic_block, c(2,2,2,2), pretrained, progress, ...)
}

#' @describeIn model_resnet ResNet 34-layer model
#' @export
model_resnet34 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("resnet34", basic_block, c(3,4,6,3), pretrained, progress, ...)
}

#' @describeIn model_resnet ResNet 50-layer model
#' @export
model_resnet50 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("resnet50", bottleneck, c(3,4,6,3), pretrained, progress, ...)
}

#' @describeIn model_resnet ResNet 101-layer model
#' @export
model_resnet101 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("resnet101", bottleneck, c(3,4,23,3), pretrained, progress, ...)
}

#' @describeIn model_resnet ResNet 152-layer model
#' @export
model_resnet152 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("resnet152", bottleneck, c(3,8,36,3), pretrained, progress, ...)
}

#' @describeIn model_resnet ResNeXt-50 32x4d model from ["Aggregated Residual Transformation for Deep Neural Networks"](https://arxiv.org/pdf/1611.05431)
#' with 32 groups having each a width of 4.
#' @export
model_resnext50_32x4d <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("resnext50_32x4d", bottleneck, c(3,4,6,3), pretrained, progress, groups=32, width_per_group=4,...)
}

#' @describeIn model_resnet ResNeXt-101 32x8d model from ["Aggregated Residual Transformation for Deep Neural Networks"](https://arxiv.org/pdf/1611.05431)
#' with 32 groups having each a width of 8.
#' @export
model_resnext101_32x8d <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("resnext101_32x8d", bottleneck, c(3,4,23,3), pretrained, progress, groups=32, width_per_group=8,...)
}

#' @describeIn model_resnet Wide ResNet-50-2 model from ["Wide Residual Networks"](https://arxiv.org/pdf/1605.07146)
#' with width per group of 128.
#' @export
model_wide_resnet50_2 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("wide_resnet50_2", bottleneck, c(3,4,6,3), pretrained, progress, width_per_group=64*2,...)
}

#' @describeIn model_resnet Wide ResNet-101-2 model from ["Wide Residual Networks"](https://arxiv.org/pdf/1605.07146)
#' with width per group of 128.
#' @export
model_wide_resnet101_2 <- function(pretrained = FALSE, progress = TRUE, ...) {
  .resnet("wide_resnet101_2", bottleneck, c(3,4,23,3), pretrained, progress, width_per_group=64*2,...)
}
