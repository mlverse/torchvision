library(torch)

is_torch_tensor <- function(x) {
  inherits(x, "torch_tensor")
}

expect_no_error <- function(object, ...) {
  expect_error(object, NA, ...)
}

expect_tensor_shape <- function(object, expected) {
  expect_tensor(object)
  expect_equal(object$shape, expected)
}

expect_tensor_dtype <- function(object, expected_dtype) {
  expect_tensor(object)
  expect_true(object$dtype == expected_dtype)
}

expect_tensor <- function(object) {
  expect_true(is_torch_tensor(object))
  expect_no_error(torch::as_array(object))
}

expect_equal_to_r <- function(object, expected, ...) {
  expect_equal(torch::as_array(object), expected, ...)
}

unlink_model_file <- function() {
  cache_path <- rappdirs::user_cache_dir("torch")
  model_file <- list.files(cache_path, pattern = "*.pth", full.names = TRUE)
  unlink(model_file)
}

expect_bbox_is_xyxy <- function(object, width, height) {
  expect_tensor(object)
  N <- object$shape[1]
  expect_tensor_shape(object, c(N, 4))

  x_min <- object[, 1]
  y_min <- object[, 2]
  x_max <- object[, 3]
  y_max <- object[, 4]

  ## bbox range checks
  expect_true((x_min >= 0)$all()$item(),
              info = "All x_min values must be >= 0.")
  expect_true((y_min >= 0)$all()$item(),
              info = "All y_min values must be >= 0.")
  expect_true((x_max <= torch_tensor(width))$all()$item(),
              info = sprintf("All x_max values must be <= width (%s).", width))
  expect_true((y_max <= torch_tensor(height))$all()$item(),
              info = sprintf("All y_max values must be <= height (%s).", height))
  expect_true((x_max > torch_tensor(1))$all()$item(),
              info = "x looks like a relative delta and shall be converted back to image width.")
  expect_true((y_max > torch_tensor(1))$all()$item(),
              info = "y looks like a relative delta and shall be converted back to image height.")
  ## bbox ordering checks
  expect_true((x_min <= x_max)$all()$item(),
              info = "Each x_min must be smaller than its x_max.")
  expect_true((y_min <= y_max)$all()$item(),
              info = "Each y_min must be smaller than its y_max.")

}


# The top detection on this cat image should be the cat class (COCO id 17).
expect_coco_model_detects_cat <- function(model, min_score = 0.25) {
  input <- base_loader("assets/class/cat/cat.2.jpg") %>%
    transform_to_tensor() %>%
    transform_resize(c(640, 640)) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)) %>%
    torch::torch_unsqueeze(1)
  model$eval()
  torch::with_no_grad({
    out <- model(input)
  })
  expect_named(out, "detections")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  labels_vec <- as.integer(out$detections[[1]]$labels$cpu())
  scores_vec <- as.numeric(out$detections[[1]]$scores$cpu())
  expect_true(all(labels_vec >= 0 & labels_vec <= 90))
  top <- which.max(scores_vec)
  expect_equal(labels_vec[top], 17L)
  expect_gt(scores_vec[top], min_score)
}
