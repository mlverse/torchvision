
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
  r <- c("https://torch-cdn.mlverse.org/models/vision/v2/models/alexnet.pth", "e9fdaf62a041c79c034de8d1867e80ee", "~245 MB" )
  model <- alexnet(...)

  if (pretrained) {
    cli_inform("Model weights for {.cls {class(model)[1]}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
  }

  model
}
