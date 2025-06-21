context("dataset-oxfordiiitpet")

test_that("tests for the Oxford-IIIT Pet dataset", {

  t = tempdir()

  expect_error(
    oxfordiiitpet <- oxfordiiitpet_dataset(root = tempfile()),
    class = "runtime_error"
  )

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "category", train = TRUE, download = TRUE)
  expect_length(oxfordiiitpet, 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,591000)
  expect_type(first_item$x, "integer")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "binary-category", train = TRUE)
  expect_length(oxfordiiitpet, 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,591000)
  expect_type(first_item$x, "integer")
  expect_type(first_item$y, "double")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "segmentation", train = TRUE)
  expect_length(oxfordiiitpet, 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,591000)
  expect_type(first_item$x, "integer")
  expect_type(first_item$y, "integer")
  expect_length(first_item$y,197000)

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "category", train = FALSE)
  expect_length(oxfordiiitpet, 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,202500)
  expect_type(first_item$x, "integer")
  expect_type(first_item$y, "integer")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "binary-category", train = FALSE)
  expect_length(oxfordiiitpet, 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,202500)
  expect_type(first_item$x, "integer")
  expect_type(first_item$y, "double")
  expect_equal(first_item$y, 1)

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, target_type = "segmentation", train = FALSE)
  expect_length(oxfordiiitpet, 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,202500)
  expect_type(first_item$x, "integer")
  expect_type(first_item$y, "integer")
  expect_length(first_item$y,67500)

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- sapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, transform = transform_to_tensor)
  dl <- dataloader(oxfordiiitpet, batch_size = 4, collate_fn = resize_collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"integer")
  expect_length(batch$y, 4)
  expect_equal(batch$y[1],1)
  expect_equal(batch$y[2],1)
  expect_equal(batch$y[3],1)
  expect_equal(batch$y[4],1)

  unlink(file.path(t, "oxfordiiitpet"), recursive = TRUE)

})