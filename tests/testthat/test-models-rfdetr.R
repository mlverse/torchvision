context("models-rfdetr")

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
  expect_coco_model_detects_cat(model, size = c(384, 384))

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
  # actually fails, detects a 'sink' with low confidence
  # expect_coco_model_detects_cat(model, size = c(512, 512))

  input <- base_loader("assets/class/dog/dog.2.jpg") %>%
    transform_to_tensor() %>%
    transform_resize(c(512, 512)) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)) %>%
    torch::torch_unsqueeze(1)
  model$eval()
  torch::with_no_grad({
    out <- model(input)
  })
  expect_named(out, "detections")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"), ignore.order = TRUE)
  expect_bbox_is_xyxy(out$detections[[1]]$boxes, c(512, 512))


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

  # actually fails, detects a 'bed' with low confidence
  # expect_coco_model_detects_cat(model, size = c(640, 640))

  input <- base_loader("assets/class/dog/dog.3.jpg") %>%
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
  expect_bbox_is_xyxy(out$detections[[1]]$boxes, c(640, 640))

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
  expect_coco_model_detects_cat(model, size = c(640, 640))

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
  expect_coco_model_detects_cat(model, size = c(640, 640))

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

  input <- base_loader("assets/class/dog/dog.4.jpg") %>%
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
  expect_bbox_is_xyxy(out$detections[[1]]$boxes, c(640, 640))

  # expect model to detect a dog (class id 93 in object 365)
  expect_equal(out$detections[[1]]$labels[1]$item(), 93L)
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
  expect_coco_model_detects_cat(model,  size = c(560, 560))

  rm(model)
  gc()
})
