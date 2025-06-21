context("dataset-fer")

t <- withr::local_tempdir()

test_that("tests for the FER-2013 dataset", {

  expect_error(
    ds <- fer_dataset(root = tempfile()),
    class = "runtime_error"
  )

  ds <- fer_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(ds, 28709)
  first_item <- ds[1]
  expect_type(first_item$x,"integer")
  expect_length(first_item$x ,2304)
  expect_named(first_item, c("x", "y"))
  expect_equal(first_item$y, "Angry")

  ds <- fer_dataset(root = t, train = FALSE)
  expect_length(ds, 7178)
  first_item <- ds[1]
  expect_type(first_item$x,"integer")
  expect_length(first_item$x ,2304)
  expect_named(first_item, c("x", "y"))
  expect_equal(first_item$y, "Angry")

  ds2 <- fer_dataset(root = t, train = TRUE)
  dl <- torch::dataloader(ds2, batch_size = 32)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x , 73728)
  expect_tensor_shape(batch$x, c(32, 48, 48))
  expect_tensor_dtype(batch$x , torch_long())
  expect_length(batch$y, 32)
  expect_equal(as.character(batch$y[1]), "Angry")
  expect_equal(as.character(batch$y[3]), "Fear")
  expect_equal(as.character(batch$y[32]), "Neutral")

})
