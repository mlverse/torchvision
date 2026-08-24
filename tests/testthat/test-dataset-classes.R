context("dataset-classes")

test_that("a task class is inserted right before dataset", {
  detection <- torch::dataset(
    name = "mock_detection_dataset",
    initialize = function() as_object_detection_dataset(self),
    .getitem = function(index) index,
    .length = function() 1L
  )

  expect_equal(
    class(detection()),
    c("mock_detection_dataset", "object_detection_dataset", "dataset", "R6")
  )
})

test_that("a task class replaces the one inherited from the parent dataset", {
  detection <- torch::dataset(
    name = "mock_detection_dataset",
    initialize = function() as_object_detection_dataset(self),
    .getitem = function(index) index,
    .length = function() 1L
  )
  segmentation <- torch::dataset(
    name = "mock_segmentation_dataset",
    inherit = detection,
    initialize = function() {
      super$initialize()
      as_segmentation_dataset(self)
    }
  )

  cls <- class(segmentation())
  expect_true("segmentation_dataset" %in% cls)
  expect_false("object_detection_dataset" %in% cls)
  expect_equal(cls[[1]], "mock_segmentation_dataset")
})

test_that("as_object_detection_target rejects a target without boxes", {
  expect_error(as_object_detection_target(list(masks = 1L)),
               "not an object detection target")
})

test_that("coercing an already classed target is a no-op", {
  target <- make_detection_target(matrix(c(10, 20, 50, 60), ncol = 4))
  expect_identical(class(as_object_detection_target(target)), class(target))

  segmentation <- make_segmentation_item()$y
  expect_identical(class(as_segmentation_target(segmentation)), class(segmentation))
})
