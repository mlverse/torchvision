context("dataset-fer")

test_that("tests for the FER-2013 dataset", {

  t <- tempfile()

  expect_error(
    ds <- fer_dataset(root = t, train = TRUE),
    class = "runtime_error"
  )

  ds <- fer_dataset(root = t, train = TRUE, download = TRUE)
  expect_equal(length(ds), 28709)
  first_item <- ds[1]
  expect_equal(dim(first_item[[1]]), c(1, 48, 48))
  expect_named(first_item, c("x", "y"))
  expect_equal(first_item[[2]], "Angry")

  ds <- fer_dataset(root = t, train = FALSE, download = TRUE)
  first_item <- ds[1]
  expect_equal(dim(first_item[[1]]), c(1, 48, 48))
  expect_equal(length(first_item[[2]]), 1)
  expect_named(first_item, c("x", "y"))

  ds2 <- fer_dataset(root = t, train = TRUE)
  dl <- torch::dataloader(ds2, batch_size = 32)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor_shape(batch$x, c(32, 1, 48, 48))
  expect_equal(length(batch$y), 32)
  expect_equal(as.character(batch$y[1]), "Angry")

})
