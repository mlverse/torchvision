context("dataset-oxfordiiitpet")

test_that("tests for the Oxford-IIIT Pet dataset", {

  oxfordiiitpet <- oxfordiiitpet_dataset(target_type = "category", train = TRUE, download = TRUE)
  expect_equal(length(oxfordiiitpet), 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(target_type = "binary-category", train = TRUE)
  expect_equal(length(oxfordiiitpet), 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_type(first_item$y, "double")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(target_type = "segmentation", train = TRUE)
  expect_equal(length(oxfordiiitpet), 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_true(inherits(first_item$y, "torch_tensor"))

  oxfordiiitpet <- oxfordiiitpet_dataset(target_type = "category", train = FALSE)
  expect_equal(length(oxfordiiitpet), 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(target_type = "binary-category", train = FALSE)
  expect_equal(length(oxfordiiitpet), 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_type(first_item$y, "double")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(target_type = "segmentation", train = FALSE)
  expect_equal(length(oxfordiiitpet), 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_true(inherits(first_item$x, "torch_tensor"))
  expect_true(inherits(first_item$y, "torch_tensor"))

  collate_fn_variable_size <- function(batch) {
    x <- lapply(batch, function(item) item$x)
    y <- lapply(batch, function(item) item$y)
    if (inherits(y[[1]], "torch_tensor") && length(dim(y[[1]])) == 0) {
      y <- torch_stack(y)
    }
    list(x = x, y = y)
  }
  ds <- oxfordiiitpet_dataset(target_type = "category", train = TRUE)
  dl <- dataloader(ds, batch_size = 4, shuffle = FALSE, collate_fn = collate_fn_variable_size)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_length(batch$x, 4)
  expect_true(all(sapply(batch$x, function(t) inherits(t, "torch_tensor"))))
  expect_length(batch$y, 4)
  expect_equal(dim(batch$x[[1]])[1], 3)
  expect_equal(dim(batch$x[[1]])[2], 500)
  expect_equal(dim(batch$x[[1]])[3], 394)
  expect_equal(batch$y[[1]], 1)
  expect_equal(batch$y[[2]], 1)

})