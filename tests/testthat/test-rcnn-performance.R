# Test file for issue #266: RCNN performance issues
# These tests document the current issues and will FAIL until fixes are implemented
# Following TDD approach: Write failing tests first, then implement fixes

context("rcnn-performance")

# =============================================================================
# Issue #266: RCNN object detection head performance problems
# https://github.com/mlverse/torchvision/issues/266
#
# Problems identified:
# 1. Hardcoded confidence threshold (0.05) - too low, causes too many proposals
# 2. Missing configurable NMS threshold parameter
# 3. Missing max detections per image limit
# 4. Using R-based NMS instead of GPU-accelerated torchvisionlib::ops_nms
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

test_that("Issue #266: confidence threshold should be configurable, not hardcoded at 0.05", {
  skip_on_cran()
  
  source_file <- find_source_file()
  if (is.null(source_file)) skip("Cannot find models-faster_rcnn.R source file")
  
  source_code <- paste(readLines(source_file), collapse = "\n")
  
  # Check for hardcoded 0.05 threshold (the bug)
  # Pattern matches: > 0.05 or >= 0.05 with various spacing
  has_hardcoded_threshold <- grepl(">\\s*0\\.05", source_code)
  
  # This test documents the bug - should FAIL until fixed
  # Once fixed, threshold should be a parameter, not hardcoded
  expect_false(
    has_hardcoded_threshold,
    info = paste(
      "Issue #266: Confidence threshold is hardcoded at 0.05.",
      "This causes too many low-confidence proposals, hurting performance.",
      "Threshold should be a configurable parameter (default: 0.5 recommended)."
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
  
  # Should use score_thresh variable, not hardcoded value
  uses_score_thresh_variable <- grepl(
    "final_scores\\s*>\\s*score_thresh|scores\\s*>\\s*score_thresh",
    source_code
  )
  
  expect_true(
    uses_score_thresh_variable,
    info = paste(
      "Issue #266: Postprocess should use configurable score_thresh.",
      "Currently uses hardcoded '> 0.05' which is too permissive.",
      "Should use: final_scores > score_thresh"
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
  
  # Try creating model with performance parameters
  # This should work after fix is implemented
  
  result <- tryCatch({
    # These parameters should be accepted by the model
    # Currently they will cause an error (unused argument)
    model <- model_fasterrcnn_resnet50_fpn(
      pretrained = FALSE,
      num_classes = 91,
      score_thresh = 0.5,       # Custom confidence threshold
      nms_thresh = 0.5,         # Custom NMS threshold  
      detections_per_img = 100  # Limit detections
    )
    TRUE
  }, error = function(e) {
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
