# Test file for issue #266: RCNN performance issues
# These tests verify the performance fixes are implemented correctly
# Following TDD approach: Tests now PASS after fixes implemented

context("rcnn-performance")

# =============================================================================
# Issue #266: RCNN object detection head performance improvements
# https://github.com/mlverse/torchvision/issues/266
#
# Fixes implemented:
# 1. Configurable score_thresh parameter (default: 0.05)
# 2. Configurable nms_thresh parameter (default: 0.5)
# 3. Configurable detections_per_img limit (default: 100)
# 4. NMS applied in both RPN and postprocessing
# 5. Fixed NMS tensor indexing crash on Windows
# =============================================================================

# Helper function to find source file
find_source_file <- function() {
  candidates <- c(
    "../../R/models-faster_rcnn.R",  # From tests/testthat/
    "R/models-faster_rcnn.R",         # From package root
    system.file("R/models-faster_rcnn.R", package = "torchvision")
  )
  for (f in candidates) {
    if (file.exists(f) && nchar(f) > 0) return(f)
  }
  return(NULL)
}

test_that("Issue #266: score_thresh is now a configurable parameter", {
  skip_on_cran()
  
  source_file <- find_source_file()
  if (is.null(source_file)) skip("Cannot find models-faster_rcnn.R source file")
  
  source_code <- paste(readLines(source_file), collapse = "\n")
  
  # Check that score_thresh is now a parameter (fix for issue #266)
  has_score_thresh_param <- grepl("score_thresh\\s*=", source_code)
  
  expect_true(
    has_score_thresh_param,
    info = paste(
      "Issue #266 FIX: score_thresh should be a configurable parameter.",
      "This allows users to filter low-confidence detections."
    )
  )
})

test_that("Issue #266: model should accept score_thresh parameter", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  source_file <- find_source_file()
  if (is.null(source_file)) skip("Cannot find models-faster_rcnn.R source file")
  
  source_code <- paste(readLines(source_file), collapse = "\n")
  
  # Check if score_thresh is a parameter in model functions
  has_score_thresh_param <- grepl("score_thresh\\s*=", source_code)
  
  expect_true(
    has_score_thresh_param,
    info = paste(
      "Issue #266: Model should accept score_thresh parameter.",
      "This allows users to filter low-confidence detections early.",
      "PyTorch uses score_thresh=0.05 as default but it should be configurable."
    )
  )
})

test_that("Issue #266: model should accept nms_thresh parameter", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  source_file <- find_source_file()
  if (is.null(source_file)) skip("Cannot find models-faster_rcnn.R source file")
  
  source_code <- paste(readLines(source_file), collapse = "\n")
  
  # Check if nms_thresh is a parameter
  has_nms_thresh_param <- grepl("nms_thresh\\s*=", source_code)
  
  expect_true(
    has_nms_thresh_param,
    info = paste(
      "Issue #266: Model should accept nms_thresh parameter.",
      "NMS threshold controls overlap removal between detections.",
      "Currently hardcoded at 0.7 in RPN, should be configurable."
    )
  )
})

test_that("Issue #266: model should accept detections_per_img parameter", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  source_file <- find_source_file()
  if (is.null(source_file)) skip("Cannot find models-faster_rcnn.R source file")
  
  source_code <- paste(readLines(source_file), collapse = "\n")
  
  # Check if max detections limit parameter exists
  has_max_detections <- grepl(
    "detections_per_img|max_detections|max_det",
    source_code,
    ignore.case = TRUE
  )
  
  expect_true(
    has_max_detections,
    info = paste(
      "Issue #266: Model should limit maximum detections per image.",
      "Without a limit, can return 1000+ detections causing slowdowns.",
      "PyTorch default is detections_per_img=100."
    )
  )
})

test_that("Issue #266: postprocess should apply NMS to final detections", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  source_file <- find_source_file()
  if (is.null(source_file)) skip("Cannot find models-faster_rcnn.R source file")
  
  source_code <- paste(readLines(source_file), collapse = "\n")
  
  # Check if NMS is applied in postprocessing (after ROI heads)
  # Currently NMS is only in RPN (line 162), not in final detection postprocessing
  
  # Count NMS usage - should appear multiple times (RPN + postprocess)
  nms_matches <- gregexpr("\\bnms\\(", source_code)[[1]]
  nms_count <- ifelse(nms_matches[1] == -1, 0, length(nms_matches))
  
  # Should have at least 2 NMS calls: one in RPN, one in postprocess
  expect_true(
    nms_count >= 2,
    info = paste(
      "Issue #266: NMS should be applied in postprocessing.",
      "Currently NMS is only used in RPN, not after ROI heads.",
      "This causes overlapping detections in final output.",
      sprintf("Found %d NMS call(s), expected >= 2.", nms_count)
    )
  )
})

test_that("Issue #266: postprocess should filter by score using configurable threshold", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  source_file <- find_source_file()
  if (is.null(source_file)) skip("Cannot find models-faster_rcnn.R source file")
  
  source_code <- paste(readLines(source_file), collapse = "\n")
  
  # Should use score_thresh variable (self$score_thresh), not hardcoded value
  uses_score_thresh_variable <- grepl(
    "final_scores\\s*>\\s*self\\$score_thresh|scores\\s*>\\s*self\\$score_thresh|scores\\s*>\\s*score_thresh",
    source_code
  )
  
  expect_true(
    uses_score_thresh_variable,
    info = paste(
      "Issue #266 FIX: Postprocess should use configurable score_thresh.",
      "Should use: final_scores > self$score_thresh"
    )
  )
})

# =============================================================================
# Integration test: Verify model accepts new parameters
# This test will fail until the fix is implemented
# =============================================================================

test_that("Issue #266: model constructor should accept performance parameters", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not_installed("torchvision")
  
  # Try creating model with performance parameters
  # This should work after fix is implemented
  
  result <- tryCatch({
    # These parameters should be accepted by the model
    model <- torchvision::model_fasterrcnn_resnet50_fpn(
      pretrained = FALSE,
      num_classes = 91,
      score_thresh = 0.5,       # Custom confidence threshold
      nms_thresh = 0.5,         # Custom NMS threshold  
      detections_per_img = 100  # Limit detections
    )
    TRUE
  }, error = function(e) {
    message("Error creating model: ", e$message)
    FALSE
  })
  
  expect_true(
    result,
    info = paste(
      "Issue #266: Model should accept score_thresh, nms_thresh, detections_per_img.",
      "These parameters control detection filtering and improve performance."
    )
  )
})

# =============================================================================
# Performance baseline test (informational)
# =============================================================================

test_that("Issue #266: document current detection count baseline", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip("Skipping performance baseline - requires model inference")
  
  # This test documents the current behavior
  # With threshold=0.05, expect many more detections than with threshold=0.5
  
  torch::torch_manual_seed(42)
  
  model <- model_fasterrcnn_resnet50_fpn(pretrained = FALSE, num_classes = 91)
  model$eval()
  
  # Create dummy input
  input <- torch::torch_randn(c(1, 3, 224, 224))
  
  torch::with_no_grad({
    output <- model(input)
  })
  
  # Document current detection count
  n_detections <- output$detections$boxes$shape[1]
  
  # With proper thresholding, should have reasonable number of detections
  message(sprintf("Current detection count with threshold=0.05: %d", n_detections))
  
  # Expect reasonable number (< 100 for a random image)
  expect_lte(
    n_detections,
    100,
    info = "With proper filtering, should have <= 100 detections per image"
  )
})
