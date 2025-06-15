
alexnet <- torch::nn_module(
  "AlexNet",
  initialize = function(num_classes = 1000) {
    self$features <- torch::nn_sequential(
      torch::nn_conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2),
      torch::nn_conv2d(64, 192, kernel_size = 5, padding = 2),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2),
      torch::nn_conv2d(192, 384, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(384, 256, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(256, 256, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2)
    )
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(6,6))
    self$classifier <- torch::nn_sequential(
      torch::nn_dropout(),
      torch::nn_linear(256 * 6 * 6, 4096),
      torch::nn_relu(inplace = TRUE),
      torch::nn_dropout(),
      torch::nn_linear(4096, 4096),
      torch::nn_relu(inplace = TRUE),
      torch::nn_linear(4096, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch::torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
  }
)

#' AlexNet Model Architecture
#'
#' AlexNet model architecture from the
#'   [One weird trick...](https://arxiv.org/abs/1404.5997) paper.
#'
#' @param pretrained (bool): If TRUE, returns a model pre-trained on ImageNet.
#' @param progress (bool): If TRUE, displays a progress bar of the download to
#'   stderr.
#' @param ... other parameters passed to the model intializer. currently only
#'   `num_classes` is used.
#'
#' @family models
#'
#' @export
model_alexnet <- function(pretrained = FALSE, progress = TRUE, ...) {

  model <- alexnet(...)

  if (pretrained) {
    state_dict_path <- download_and_cache(
      "https://torch-cdn.mlverse.org/models/vision/v1/models/alexnet.pth",
    )
    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}
