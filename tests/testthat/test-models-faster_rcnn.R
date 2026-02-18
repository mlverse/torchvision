test_that("tests for non-pretrained model_fasterrcnn_resnet50_fpn", {

  model <- model_fasterrcnn_resnet50_fpn(score_thresh = 0.6, nms_thresh = 0.9, detections_per_img = 100)
  input <- base_loader("assets/class/cat/cat.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(200,200)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_is(out$detections, "list")
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)


  model <- model_fasterrcnn_resnet50_fpn(num_classes = 10, score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("tests for non-pretrained model_fasterrcnn_resnet50_fpn_v2", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_resnet50_fpn_v2(score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  input <- base_loader("assets/class/cat/cat.1.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)


  model <- model_fasterrcnn_resnet50_fpn_v2(num_classes = 10, score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("tests for non-pretrained model_fasterrcnn_mobilenet_v3_large_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_fpn(score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  input <- base_loader("assets/class/cat/cat.2.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)


  model <- model_fasterrcnn_resnet50_fpn_v2(num_classes = 10, score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("tests for non-pretrained model_fasterrcnn_mobilenet_v3_large_320_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  input <- base_loader("assets/class/cat/cat.3.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(num_classes = 10, score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
})

test_that("tests for pretrained model_fasterrcnn_resnet50_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE, score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 3)
  input <- base_loader("assets/class/cat/cat.4.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE, score_thresh = 0.3, nms_thresh = 0.6, detections_per_img = 100)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  if (out$detections$boxes$shape[1] > 0) {
    expert_bbox_is_xyxy(out$detections$boxes, 180, 180)
  }
})

test_that("tests for pretrained model_fasterrcnn_resnet50_fpn_v2", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  input <- base_loader("assets/class/cat/cat.5.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(180,180)) %>% torch_unsqueeze(1)
  model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE, score_thresh = 0.4, nms_thresh = 0.6, detections_per_img = 100)
  model$eval()
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections, c("boxes","labels", "scores"))
  expect_tensor(out$detections$labels)
  expect_tensor(out$detections$scores)
  if (out$detections$boxes$shape[1] > 0) {
    expert_bbox_is_xyxy(out$detections$boxes, 180, 180)
  }
})

test_that("tests for pretrained model_fasterrcnn_mobilenet_v3_large_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_fpn(pretrained = TRUE, score_thresh = 0.6, nms_thresh = 0.9, detections_per_img = 100)
  input <- base_loader("assets/class/dog/dog.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(240,240)) %>% torch_unsqueeze(1)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  if (out$detections[[1]]$boxes$shape[1] > 0) {
    boxes <- as.matrix(out$detections[[1]]$boxes)

    # bbox must be positive and within (240x240)
    expect_true(all(boxes >= 0))
    expect_true(all(boxes[, c(1, 3)] <= 240))
    expect_true(all(boxes[, c(2, 4)] <= 240))

    # bbox must be coherent: x2 > x1 et y2 > y1
    expect_true(all(boxes[, 3] >= boxes[, 1]))
    expect_true(all(boxes[, 4] >= boxes[, 2]))

    # scores must be within [0, 1]
    scores <- as.numeric(out$detections[[1]]$scores)
    expect_all_true(scores >= 0)
    expect_all_true(scores <= 1)
    }
})

test_that("tests for pretrained model_fasterrcnn_mobilenet_v3_large_320_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(pretrained = TRUE, score_thresh = 0.5, nms_thresh = 0.8, detections_per_img = 10)
  input <- base_loader("assets/class/dog/dog.1.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(360,360)) %>% torch_unsqueeze(1)
  out <- model(input)
  expect_named(out, c("features","detections"))
  expect_named(out$detections[[1]], c("boxes","labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  if (out$detections[[1]]$boxes$shape[1] > 0) {
    boxes <- as.matrix(out$detections[[1]]$boxes)

    # bbox must be positive and within (360x360)
    expect_true(all(boxes >= 0))
    expect_true(all(boxes[, c(1, 3)] <= 360))
    expect_true(all(boxes[, c(2, 4)] <= 360))

    # bbox must be coherent: x2 > x1 et y2 > y1
    expect_true(all(boxes[, 3] >= boxes[, 1]))
    expect_true(all(boxes[, 4] >= boxes[, 2]))

    # scores must be within [0, 1]
    scores <- as.numeric(out$detections[[1]]$scores)
    expect_all_true(scores >= 0)
    expect_all_true(scores <= 1)
  }
})

