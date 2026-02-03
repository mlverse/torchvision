context("models-convnext-detection")

test_that("tests for non-pretrained model_convnext_tiny_detection", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_tiny_detection()
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

  model <- model_convnext_tiny_detection(num_classes = 10)
  out <- model(input)
  expect_named(out, c("features", "detections"))
  expect_is(out$detections, "list")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("tests for non-pretrained model_convnext_small_detection", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_small_detection()
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

test_that("tests for non-pretrained model_convnext_base_detection", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_base_detection()
  input <- base_loader("assets/class/cat/cat.2.jpg") %>%
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

  model <- model_convnext_base_detection(num_classes = 10)
  out <- model(input)
  expect_is(out$detections, "list")
  expect_named(out, c("features", "detections"))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("model_convnext_detection works with pretrained backbone", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_tiny_detection(pretrained_backbone = TRUE)
  input <- base_loader("assets/class/cat/cat.3.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180, 180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features", "detections"))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("model_convnext_detection handles different image sizes", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_tiny_detection(num_classes = 10)
  model$eval()

  input_224 <- base_loader("assets/class/dog/dog.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(224, 224)) %>% torch_unsqueeze(1)
  out_224 <- model(input_224)
  expect_named(out_224, c("features", "detections"))
  expect_named(out_224$detections, c("boxes", "labels", "scores"))

  input_320 <- base_loader("assets/class/dog/dog.1.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(320, 320)) %>% torch_unsqueeze(1)
  out_320 <- model(input_320)
  expect_named(out_320, c("features", "detections"))
  expect_named(out_320$detections, c("boxes", "labels", "scores"))

  input_512 <- base_loader("assets/class/dog/dog.2.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(512, 512)) %>% torch_unsqueeze(1)
  out_512 <- model(input_512)
  expect_named(out_512, c("features", "detections"))
  expect_named(out_512$detections, c("boxes", "labels", "scores"))
})

test_that("model_convnext_detection validates num_classes parameter", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  expect_no_error(model_convnext_tiny_detection(num_classes = 10, pretrained_backbone = FALSE))
  expect_no_error(model_convnext_tiny_detection(num_classes = 91, pretrained_backbone = FALSE))
  expect_error(model_convnext_tiny_detection(num_classes = 0), "`num_classes` must be positive")
  expect_error(model_convnext_tiny_detection(num_classes = -1), "`num_classes` must be positive")
})


test_that("model_convnext_detection output format matches faster_rcnn", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_tiny_detection(num_classes = 10)
  model$eval()
  expect_false(is.null(model$backbone))

  input <- base_loader("assets/class/dog/dog.4.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(200, 200)) %>% torch_unsqueeze(1)
  out <- model(input)

  expect_type(out$features, "list")
  expect_true(length(out$features) >= 4)

  for (i in seq_along(out$features)) {
    expect_tensor(out$features[[i]])
  }

  expect_named(out, c("features", "detections"))
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  expect_equal(out$detections[[1]]$labels$shape[1], out$detections[[1]]$scores$shape[1])
  expect_equal(out$detections[[1]]$boxes$shape[1], out$detections[[1]]$labels$shape[1])
  if (out$detections[[1]]$boxes$shape[1] > 0) {
    boxes <- as.matrix(out$detections[[1]]$boxes)

    # bbox must be positive and within (200x200)
    expect_true(all(boxes >= 0))
    expect_true(all(boxes[, c(1, 3)] <= 200))
    expect_true(all(boxes[, c(2, 4)] <= 200))

    # bbox must be coherent: x2 > x1 et y2 > y1
    # TODO may need rework
    # expect_true(all(boxes[, 3] >= boxes[, 1]))
    expect_true(all(boxes[, 4] >= boxes[, 2]))

    # scores must be within [0, 1]
    scores <- as.numeric(out$detections[[1]]$scores)
    expect_all_true(scores >= 0)
    expect_all_true(scores <= 1)
  }
})

test_that("model_convnext_detection handles batch processing", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_convnext_tiny_detection(num_classes = 10)
  model$eval()

  single <- base_loader("assets/class/dog/dog.5.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(200, 200))
  batch <- torch_stack(list(single, single), dim = 1)
  out <- model(batch)
  expect_named(out, c("features", "detections"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_tensor(out$detections[[2]]$boxes)
  expect_tensor(out$detections[[2]]$labels)
  expect_tensor(out$detections[[2]]$scores)
})
