test_that("coco_dataset initializes correctly", {
  skip("Not implemented yet")
  d <- coco_dataset(tempdir(), split = "train", year = "2017", download = FALSE)
  expect_true(inherits(d, "coco"))
})
