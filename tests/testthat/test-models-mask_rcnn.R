input <- base_loader("assets/class/cat/cat.0.jpg") %>%
  transform_to_tensor() %>%
  transform_normalize(c(0.485, 0.456, 0.406), c(0.229, 0.224, 0.225)) %>%
  transform_resize(c(200,200)) %>%
  torch_unsqueeze(1)


test_that("maskrcnn_resnet50_fpn loads without pretrained weights", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_maskrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 91)
  expect_s3_class(model, "nn_module")
  expect_true(!is.null(model$backbone))
  expect_true(!is.null(model$rpn))
  expect_true(!is.null(model$roi_heads))
  expect_true(!is.null(model$mask_head))
})

test_that("maskrcnn_resnet50_fpn_v2 loads without pretrained weights", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_maskrcnn_resnet50_fpn_v2(pretrained = FALSE, num_classes = 91)
  expect_s3_class(model, "nn_module")
  expect_true(!is.null(model$backbone))
  expect_true(!is.null(model$rpn))
  expect_true(!is.null(model$roi_heads))
  expect_true(!is.null(model$mask_head))
})

test_that("maskrcnn_resnet50_fpn inference works", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  # We expect 0 detection here, we keep the compute budget for pretrained model
  model <- model_maskrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 91, score_thresh = 0.6, nms_thresh = 0.1, detections_per_img = 100)
  model$eval()

  # Test forward pass
  output <- model(input)

  # Check output structure
  expect_named(output, c("features" ,"detections"))
  expect_named(output$detections, c("boxes" ,"labels","scores","masks"))

  # Verify tensor types
  expect_tensor(output$detections$boxes)
  expect_tensor(output$detections$labels)
  expect_tensor(output$detections$scores)
  expect_tensor(output$detections$masks)

  # Verify dimensions are consistent
  n_detections <- output$detections$masks$shape[1]
  expect_tensor_shape(output$detections$masks, c(n_detections, 28, 28))
  expect_tensor_shape(output$detections$labels, n_detections)
  expect_tensor_shape(output$detections$scores, n_detections)


  # Check mask dimensions (should be N x 28 x 28)
  N <- output$detections$masks$shape[1]
  expect_tensor_shape(output$detections$masks, c(N, 28, 28))

})

test_that("maskrcnn_resnet50_fpn_v2 inference works", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")


  # We expect 0 detection here, we keep the compute budget for pretrained model
  model <- model_maskrcnn_resnet50_fpn_v2(pretrained = FALSE, num_classes = 91, score_thresh = 0.6, nms_thresh = 0.9, detections_per_img = 100)
  model$eval()

  # Test forward pass
  output <- model(input)

  # Check output structure
  expect_named(output, c("features" ,"detections"))
  expect_named(output$detections, c("boxes" ,"labels","scores","masks"))

  # Check masks are present
  expect_tensor(output$detections$masks)

  # Check mask dimensions (should be N x 28 x 28)
  n_detections <- output$detections$masks$shape[1]
  expect_tensor_shape(output$detections$masks, c(n_detections, 28, 28))
})

test_that("maskrcnn handles empty detections correctly", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")


  # Use very high score threshold to force empty detections
  model <- model_maskrcnn_resnet50_fpn(
    pretrained = FALSE,
    num_classes = 91,
    score_thresh = 0.99, nms_thresh = 0.9, detections_per_img = 100
  )
  model$eval()

  output <- model(input)

  # Should return empty tensors with correct shapes
  expect_tensor_shape(output$detections$boxes, c(0,4))
  expect_tensor_shape(output$detections$labels, 0)
  expect_tensor_shape(output$detections$scores, 0)

  # Mask shape should be (0, 28, 28)
  expect_tensor_shape(output$detections$masks, c(0,28,28))
})

test_that("maskrcnn respects detections_per_img parameter", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  max_detections <- 5
  model <- model_maskrcnn_resnet50_fpn(
    pretrained = FALSE,
    num_classes = 91,
    score_thresh = 0.4,  # Low threshold to get more detections
    nms_thresh = 0.8,
    detections_per_img = max_detections
  )
  model$eval()

  output <- model(input)

  # Number of detections should not exceed max_detections
  n_detections <- output$detections$boxes$shape[1]
  expect_lte(n_detections, max_detections)
})

test_that("maskrcnn with different num_classes works", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  # Test with custom number of classes
  model <- model_maskrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 10, score_thresh = 0.6, nms_thresh = 0.9, detections_per_img = 100)
  model$eval()

  input <- torch::torch_randn(1, 3, 800, 800)
  output <- model(input)

  # Should work without errors
  expect_tensor(output$detections$masks)
})

test_that("maskrcnn pretrained requires num_classes = 91", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  # Should error when pretrained=TRUE with num_classes != 91
  expect_error(
    model_maskrcnn_resnet50_fpn(pretrained = TRUE, num_classes = 10, score_thresh = 0.6, nms_thresh = 0.9, detections_per_img = 100),
    "Pretrained weights require num_classes = 91"
  )
})

test_that("mask_rcnn pretrained infer correctly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_maskrcnn_resnet50_fpn(pretrained = TRUE, score_thresh = 0.01, nms_thresh = 0.7, detections_per_img = 100)
  model$eval()

  output <- model(input)
  # Masks should be expanded back to image size
  if (output$detections$boxes$shape[1] > 0) {
    expert_bbox_is_xyxy(output$detections$boxes, 200, 200)
  }

})

test_that("mask_rcnn_v2 pretrained infer correctly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_maskrcnn_resnet50_fpn_v2(pretrained = TRUE, score_thresh = 0.01, nms_thresh = 0.7, detections_per_img = 100)
  model$eval()

  output <- model(input)
  # Masks should be expanded back to image size
  if (output$detections$boxes$shape[1] > 0) {
    expert_bbox_is_xyxy(output$detections$boxes, 200, 200)
  }

})

test_that("mask_head_module initializes correctly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  mask_head <- mask_head_module()
  mask_rcnn_pred <- mask_rcnn_predictor(num_classes = 91)
  expect_is(mask_head, "nn_module")

  # Test forward pass
  input <- torch::torch_randn(2, 256, 14, 14)
  output <- input %>% mask_head() %>% mask_rcnn_pred()

  # Output should be (2, 91, 28, 28)
  expect_tensor_shape(output, c(2, 91, 28, 28))
})

test_that("mask_head_module_v2 initializes correctly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  mask_head <- mask_head_module_v2(num_classes = 91)
  expect_s3_class(mask_head, "nn_module")

  # Test forward pass
  input <- torch::torch_randn(2, 256, 14, 14)
  output <- mask_head(input)

  # Output should be (2, 91, 28, 28)
  expect_tensor_shape(output, c(2, 91, 28, 28))
})

test_that("roi_align_masks produces correct output shape", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  # Create dummy feature map (1, 256, 100, 100)
  feature_map <- torch::torch_randn(1, 256, 100, 100)

  # Create dummy proposals (5 boxes)
  proposals <- torch::torch_tensor(rbind(
    c(10, 10, 50, 50),
    c(20, 20, 60, 60),
    c(30, 30, 70, 70),
    c(40, 40, 80, 80),
    c(50, 50, 90, 90)
  ))

  # Apply ROI align
  output <- roi_align_masks(feature_map, proposals, output_size = c(14L, 14L))

  # Output should be (5, 256, 14, 14)
  expect_tensor_shape(output, c(5, 256, 14, 14))
})
