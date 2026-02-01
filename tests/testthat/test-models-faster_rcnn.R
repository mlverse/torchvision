test_that("tests for non-pretrained model_fasterrcnn_resnet50_fpn", {

  model <- model_fasterrcnn_resnet50_fpn()
  input <- base_loader("assets/class/cat/cat.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(200,200)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)


  model <- model_fasterrcnn_resnet50_fpn(num_classes = 10)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

test_that("tests for non-pretrained model_fasterrcnn_resnet50_fpn_v2", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_resnet50_fpn_v2()
  input <- base_loader("assets/class/cat/cat.1.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)


  model <- model_fasterrcnn_resnet50_fpn_v2(num_classes = 10)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

test_that("tests for non-pretrained model_fasterrcnn_mobilenet_v3_large_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_fpn()
  input <- base_loader("assets/class/cat/cat.2.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)


  model <- model_fasterrcnn_resnet50_fpn_v2(num_classes = 10)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

test_that("tests for non-pretrained model_fasterrcnn_mobilenet_v3_large_320_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_320_fpn()
  input <- base_loader("assets/class/cat/cat.3.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)


  model <- model_fasterrcnn_resnet50_fpn_v2(num_classes = 10)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

test_that("tests for pretrained model_fasterrcnn_resnet50_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE)
  input <- base_loader("assets/class/cat/cat.4.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

test_that("tests for pretrained model_fasterrcnn_resnet50_fpn_v2", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE)
  input <- base_loader("assets/class/cat/cat.5.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

test_that("tests for pretrained model_fasterrcnn_mobilenet_v3_large_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_fpn(pretrained = TRUE)
  input <- base_loader("assets/class/dog/dog.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

test_that("tests for pretrained model_fasterrcnn_mobilenet_v3_large_320_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(pretrained = TRUE)
  input <- base_loader("assets/class/dog/dog.1.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$boxes)
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  expect_equal(out$detections$boxes$shape[2], 4L)
})

