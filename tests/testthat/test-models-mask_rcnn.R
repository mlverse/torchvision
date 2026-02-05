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

test_that("maskrcnn_resnet50_fpn forward pass works", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  model <- model_maskrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 91)
  model$eval()
  
  # Create dummy input (1, 3, 800, 800)
  input <- torch::torch_randn(1, 3, 800, 800)
  
  # Test forward pass
  output <- model(input)
  
  # Check output structure
  expect_true("features" %in% names(output))
  expect_true("detections" %in% names(output))
  expect_true("boxes" %in% names(output$detections))
  expect_true("labels" %in% names(output$detections))
  expect_true("scores" %in% names(output$detections))
  expect_true("masks" %in% names(output$detections))
  
  # Check masks are present
  expect_s3_class(output$detections$masks, "torch_tensor")
  
  # Check mask dimensions (should be N x 28 x 28)
  mask_shape <- output$detections$masks$shape
  expect_equal(length(mask_shape), 3)
  if (mask_shape[1] > 0) {
    expect_equal(as.integer(mask_shape[2]), 28)
    expect_equal(as.integer(mask_shape[3]), 28)
  }
})

test_that("maskrcnn_resnet50_fpn_v2 forward pass works", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  model <- model_maskrcnn_resnet50_fpn_v2(pretrained = FALSE, num_classes = 91)
  model$eval()
  
  # Create dummy input (1, 3, 800, 800)
  input <- torch::torch_randn(1, 3, 800, 800)
  
  # Test forward pass
  output <- model(input)
  
  # Check output structure
  expect_true("features" %in% names(output))
  expect_true("detections" %in% names(output))
  expect_true("boxes" %in% names(output$detections))
  expect_true("labels" %in% names(output$detections))
  expect_true("scores" %in% names(output$detections))
  expect_true("masks" %in% names(output$detections))
  
  # Check masks are present
  expect_s3_class(output$detections$masks, "torch_tensor")
  
  # Check mask dimensions (should be N x 28 x 28)
  mask_shape <- output$detections$masks$shape
  expect_equal(length(mask_shape), 3)
  if (mask_shape[1] > 0) {
    expect_equal(as.integer(mask_shape[2]), 28)
    expect_equal(as.integer(mask_shape[3]), 28)
  }
})

test_that("maskrcnn output format matches expected structure", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  model <- model_maskrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 91)
  model$eval()
  
  input <- torch::torch_randn(1, 3, 800, 800)
  output <- model(input)
  
  # Verify all required fields are present
  expect_true(all(c("boxes", "labels", "scores", "masks") %in% names(output$detections)))
  
  # Verify tensor types
  expect_s3_class(output$detections$boxes, "torch_tensor")
  expect_s3_class(output$detections$labels, "torch_tensor")
  expect_s3_class(output$detections$scores, "torch_tensor")
  expect_s3_class(output$detections$masks, "torch_tensor")
  
  # Verify dimensions are consistent
  n_detections <- output$detections$boxes$shape[1]
  expect_equal(output$detections$labels$shape[1], n_detections)
  expect_equal(output$detections$scores$shape[1], n_detections)
  expect_equal(output$detections$masks$shape[1], n_detections)
})

test_that("maskrcnn handles empty detections correctly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  # Use very high score threshold to force empty detections
  model <- model_maskrcnn_resnet50_fpn(
    pretrained = FALSE, 
    num_classes = 91,
    score_thresh = 0.99
  )
  model$eval()
  
  input <- torch::torch_randn(1, 3, 800, 800)
  output <- model(input)
  
  # Should return empty tensors with correct shapes
  expect_equal(as.integer(output$detections$boxes$shape[1]), 0)
  expect_equal(as.integer(output$detections$labels$shape[1]), 0)
  expect_equal(as.integer(output$detections$scores$shape[1]), 0)
  expect_equal(as.integer(output$detections$masks$shape[1]), 0)
  
  # Mask shape should be (0, 28, 28)
  expect_equal(length(output$detections$masks$shape), 3)
  expect_equal(as.integer(output$detections$masks$shape[2]), 28)
  expect_equal(as.integer(output$detections$masks$shape[3]), 28)
})

test_that("maskrcnn respects detections_per_img parameter", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  max_detections <- 5
  model <- model_maskrcnn_resnet50_fpn(
    pretrained = FALSE,
    num_classes = 91,
    score_thresh = 0.01,  # Low threshold to get more detections
    detections_per_img = max_detections
  )
  model$eval()
  
  input <- torch::torch_randn(1, 3, 800, 800)
  output <- model(input)
  
  # Number of detections should not exceed max_detections
  n_detections <- as.integer(output$detections$boxes$shape[1])
  expect_lte(n_detections, max_detections)
})

test_that("maskrcnn mask values are in valid range", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  model <- model_maskrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 91)
  model$eval()
  
  input <- torch::torch_randn(1, 3, 800, 800)
  output <- model(input)
  
  # Masks should be probabilities (0 to 1) after sigmoid
  if (output$detections$masks$shape[1] > 0) {
    mask_min <- torch::torch_min(output$detections$masks)$item()
    mask_max <- torch::torch_max(output$detections$masks)$item()
    
    expect_gte(mask_min, 0)
    expect_lte(mask_max, 1)
  }
})

test_that("maskrcnn with different num_classes works", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  # Test with custom number of classes
  model <- model_maskrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 10)
  model$eval()
  
  input <- torch::torch_randn(1, 3, 800, 800)
  output <- model(input)
  
  # Should work without errors
  expect_s3_class(output$detections$masks, "torch_tensor")
})

test_that("maskrcnn pretrained requires num_classes = 91", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  # Should error when pretrained=TRUE with num_classes != 91
  expect_error(
    model_maskrcnn_resnet50_fpn(pretrained = TRUE, num_classes = 10),
    "Pretrained weights require num_classes = 91"
  )
})

test_that("mask_head_module initializes correctly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  mask_head <- mask_head_module(num_classes = 91)()
  expect_s3_class(mask_head, "nn_module")
  
  # Test forward pass
  input <- torch::torch_randn(2, 256, 14, 14)
  output <- mask_head(input)
  
  # Output should be (2, 91, 28, 28)
  expect_equal(as.integer(output$shape[1]), 2)
  expect_equal(as.integer(output$shape[2]), 91)
  expect_equal(as.integer(output$shape[3]), 28)
  expect_equal(as.integer(output$shape[4]), 28)
})

test_that("mask_head_module_v2 initializes correctly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  
  mask_head <- mask_head_module_v2(num_classes = 91)()
  expect_s3_class(mask_head, "nn_module")
  
  # Test forward pass
  input <- torch::torch_randn(2, 256, 14, 14)
  output <- mask_head(input)
  
  # Output should be (2, 91, 28, 28)
  expect_equal(as.integer(output$shape[1]), 2)
  expect_equal(as.integer(output$shape[2]), 91)
  expect_equal(as.integer(output$shape[3]), 28)
  expect_equal(as.integer(output$shape[4]), 28)
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
  expect_equal(as.integer(output$shape[1]), 5)
  expect_equal(as.integer(output$shape[2]), 256)
  expect_equal(as.integer(output$shape[3]), 14)
  expect_equal(as.integer(output$shape[4]), 14)
})
