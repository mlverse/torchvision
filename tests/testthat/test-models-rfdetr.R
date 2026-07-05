context("models-rfdetr")

# Helper: run rfdetr model on cat image and validate detections
expect_rfdetr_detects_cat <- function(model, model_res, num_classes = 91L, min_score = 0.15) {
  input <- base_loader("assets/class/cat/cat.2.jpg") %>%
    transform_to_tensor() %>%
    transform_resize(c(model_res, model_res)) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)) %>%
    torch_unsqueeze(1)

  model$eval()
  torch::with_no_grad({
    out <- model(input)
  })

  expect_true("detections" %in% names(out))
  results <- out$detections

  expect_named(results[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(results[[1]]$boxes$shape[2], 4L)
  expect_bbox_is_xyxy(results[[1]]$boxes, model_res, model_res)

  labels_vec <- as.integer(results[[1]]$labels$cpu())
  scores_vec <- as.numeric(results[[1]]$scores$cpu())
  expect_true(all(labels_vec >= 0 & labels_vec < num_classes))

  top <- which.max(scores_vec)
  if (num_classes == 91L) {
    expect_true(17L %in% labels_vec, info = "label 17 (cat) must be detected")
    expect_gt(scores_vec[top], min_score)
  }
}

test_that("test for non-pretrained model_rfdetr_nano", {
  model <- model_rfdetr_nano()
  input <- torch::torch_randn(1, 3, 384, 384)
  model$eval()
  out <- model(input)
  expect_true("detections" %in% names(out))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  rm(model)
  gc()
})

test_that("test for pretrained model_rfdetr_nano", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_rfdetr_nano(pretrained = TRUE)
  expect_rfdetr_detects_cat(model, 384)

  rm(model)
  gc()
})

test_that("test for non-pretrained model_rfdetr_small", {
  model <- model_rfdetr_small()
  input <- torch::torch_randn(1, 3, 512, 512)
  model$eval()
  out <- model(input)
  expect_true("detections" %in% names(out))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  rm(model)
  gc()
})

test_that("test for pretrained model_rfdetr_small", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_rfdetr_small(pretrained = TRUE)
  expect_rfdetr_detects_cat(model, 512)

  rm(model)
  gc()
})

test_that("test for non-pretrained model_rfdetr_medium", {
  model <- model_rfdetr_medium()
  input <- torch::torch_randn(1, 3, 640, 640)
  model$eval()
  out <- model(input)
  expect_true("detections" %in% names(out))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  rm(model)
  gc()
})

test_that("test for pretrained model_rfdetr_medium", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_rfdetr_medium(pretrained = TRUE)
  expect_rfdetr_detects_cat(model, 640)

  rm(model)
  gc()
})

test_that("test for non-pretrained model_rfdetr_base", {
  model <- model_rfdetr_base()
  input <- torch::torch_randn(1, 3, 640, 640)
  model$eval()
  out <- model(input)
  expect_true("detections" %in% names(out))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  rm(model)
  gc()
})

test_that("test for pretrained model_rfdetr_base", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_rfdetr_base(pretrained = TRUE)
  expect_rfdetr_detects_cat(model, 640)

  rm(model)
  gc()
})

test_that("test for non-pretrained model_rfdetr_base_2", {
  model <- model_rfdetr_base_2()
  input <- torch::torch_randn(1, 3, 640, 640)
  model$eval()
  out <- model(input)
  expect_true("detections" %in% names(out))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  rm(model)
  gc()
})

test_that("test for pretrained model_rfdetr_base_2", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_rfdetr_base_2(pretrained = TRUE)
  expect_rfdetr_detects_cat(model, 640)

  rm(model)
  gc()
})

test_that("test for non-pretrained model_rfdetr_base_o365", {
  model <- model_rfdetr_base_o365()
  input <- torch::torch_randn(1, 3, 640, 640)
  model$eval()
  out <- model(input)
  expect_true("detections" %in% names(out))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  rm(model)
  gc()
})

test_that("test for pretrained model_rfdetr_base_o365", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_rfdetr_base_o365(pretrained = TRUE)
  expect_rfdetr_detects_cat(model, 640, num_classes = 366L)

  rm(model)
  gc()
})

test_that("test for non-pretrained model_rfdetr_large", {
  model <- model_rfdetr_large()
  input <- torch::torch_randn(1, 3, 560, 560)
  model$eval()
  out <- model(input)
  expect_true("detections" %in% names(out))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  rm(model)
  gc()
})

test_that("test for pretrained model_rfdetr_large", {

  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_rfdetr_large(pretrained = TRUE)
  expect_rfdetr_detects_cat(model, 560)

  rm(model)
  gc()
})
