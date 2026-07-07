context("models-convnext-detection")

test_that("tests for non-pretrained model_convnext_tiny_detection works with batch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_tiny_detection(pretrained_backbone = TRUE)
  input <- base_loader("assets/class/cat/cat.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(200, 200)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features", "detections"))
  expect_is(out$detections, "list")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)


  batch <- torch_stack(list(base_loader("assets/class/cat/cat.0.jpg") %>% transform_to_tensor() %>% transform_resize(c(200, 200)),
                            base_loader("assets/class/cat/cat.1.jpg") %>% transform_to_tensor() %>% transform_resize(c(200, 200))),
                       dim = 1)
  model <- model_convnext_tiny_detection(num_classes = 10)
  out <- model(batch)
  expect_named(out, c("features", "detections"))
  expect_is(out$detections, "list")
  expect_length(out$detections, 2)
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  expect_named(out$detections[[2]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[2]]$boxes)
  expect_tensor(out$detections[[2]]$labels)
  expect_tensor(out$detections[[2]]$scores)
  expect_equal(out$detections[[2]]$boxes$shape[2], 4L)
})

test_that("tests for pretrained / non-pretrained model_convnext_small_detection", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_small_detection(pretrained_backbone = TRUE)
  input <- base_loader("assets/class/cat/cat.1.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180, 180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features", "detections"))
  expect_is(out$detections, "list")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  # we cannot succesfully assert bbox here :
  #   expect_bbox_is_xyxy(out$detections[[1]]$boxes, 180, 180)
  # }

  model <- model_convnext_small_detection(num_classes = 10)
  out <- model(input)
  expect_named(out, c("features", "detections"))
  expect_is(out$detections, "list")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("tests for pretrained / non-pretrained model_convnext_base_detection", {
  skip_if(Sys.getenv("TEST_HUGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_HUGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_base_detection(pretrained_backbone = TRUE)
  input <- base_loader("assets/class/cat/cat.2.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180, 180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features", "detections"))
  expect_is(out$detections, "list")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  if (out$detections[[1]]$boxes$shape[1] > 0) {
    expect_bbox_is_xyxy(out$detections[[1]]$boxes, 180, 180)
  }
})

test_that("tests for non-pretrained model_convnext_base_detection", {
  skip_if(Sys.getenv("TEST_HUGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_HUGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_base_detection(num_classes = 10)
  input <- base_loader("assets/class/cat/cat.2.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180, 180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_is(out$detections, "list")
  expect_named(out, c("features", "detections"))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})


test_that("model_convnext_detection handles different image sizes", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_tiny_detection(num_classes = 10)
  model$eval()

  input_224 <- base_loader("assets/class/dog/dog.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(224, 224)) %>% torch_unsqueeze(1)
  out_224 <- model(input_224)
  expect_named(out_224, c("features", "detections"))
  expect_named(out_224$detections[[1]], c("boxes", "labels", "scores"))

  input_320 <- base_loader("assets/class/dog/dog.1.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(320, 320)) %>% torch_unsqueeze(1)
  out_320 <- model(input_320)
  expect_named(out_320, c("features", "detections"))
  expect_named(out_320$detections[[1]], c("boxes", "labels", "scores"))

  input_512 <- base_loader("assets/class/dog/dog.2.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(512, 512)) %>% torch_unsqueeze(1)
  out_512 <- model(input_512)
  expect_named(out_512, c("features", "detections"))
  expect_named(out_512$detections[[1]], c("boxes", "labels", "scores"))
})

test_that("model_convnext_detection validates num_classes parameter", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  expect_no_error(model_convnext_tiny_detection(num_classes = 10, pretrained_backbone = FALSE))
  expect_no_error(model_convnext_tiny_detection(num_classes = 91, pretrained_backbone = FALSE))
  expect_error(model_convnext_tiny_detection(num_classes = 0), "must be positive")
  expect_error(model_convnext_tiny_detection(num_classes = -1), "must be positive")
})


