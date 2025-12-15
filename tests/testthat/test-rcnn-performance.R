# Tests for Faster R-CNN configurable detection parameters

context("rcnn-performance")

test_that("model_fasterrcnn_resnet50_fpn accepts score_thresh parameter", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  result <- tryCatch({
    model <- torchvision::model_fasterrcnn_resnet50_fpn(
      pretrained = FALSE,
      num_classes = 91,
      score_thresh = 0.5
    )
    TRUE
  }, error = function(e) FALSE)
  
  expect_true(result)
})

test_that("model_fasterrcnn_resnet50_fpn accepts nms_thresh parameter", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  result <- tryCatch({
    model <- torchvision::model_fasterrcnn_resnet50_fpn(
      pretrained = FALSE,
      num_classes = 91,
      nms_thresh = 0.4
    )
    TRUE
  }, error = function(e) FALSE)
  
  expect_true(result)
})

test_that("model_fasterrcnn_resnet50_fpn accepts detections_per_img parameter", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  result <- tryCatch({
    model <- torchvision::model_fasterrcnn_resnet50_fpn(
      pretrained = FALSE,
      num_classes = 91,
      detections_per_img = 50
    )
    TRUE
  }, error = function(e) FALSE)
  
  expect_true(result)
})

test_that("model_fasterrcnn_resnet50_fpn accepts all detection parameters together", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  result <- tryCatch({
    model <- torchvision::model_fasterrcnn_resnet50_fpn(
      pretrained = FALSE,
      num_classes = 91,
      score_thresh = 0.5,
      nms_thresh = 0.4,
      detections_per_img = 50
    )
    TRUE
  }, error = function(e) FALSE)
  
  expect_true(result)
})

test_that("model_fasterrcnn_resnet50_fpn_v2 accepts detection parameters", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  result <- tryCatch({
    model <- torchvision::model_fasterrcnn_resnet50_fpn_v2(
      pretrained = FALSE,
      num_classes = 91,
      score_thresh = 0.5,
      nms_thresh = 0.4,
      detections_per_img = 50
    )
    TRUE
  }, error = function(e) FALSE)
  
  expect_true(result)
})

test_that("model_fasterrcnn_mobilenet_v3_large_fpn accepts detection parameters", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  result <- tryCatch({
    model <- torchvision::model_fasterrcnn_mobilenet_v3_large_fpn(
      pretrained = FALSE,
      num_classes = 91,
      score_thresh = 0.5,
      nms_thresh = 0.4,
      detections_per_img = 50
    )
    TRUE
  }, error = function(e) FALSE)
  
  expect_true(result)
})

test_that("model_fasterrcnn_mobilenet_v3_large_320_fpn accepts detection parameters", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  result <- tryCatch({
    model <- torchvision::model_fasterrcnn_mobilenet_v3_large_320_fpn(
      pretrained = FALSE,
      num_classes = 91,
      score_thresh = 0.5,
      nms_thresh = 0.4,
      detections_per_img = 50
    )
    TRUE
  }, error = function(e) FALSE)
  
  expect_true(result)
})
