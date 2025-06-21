context("dataset-flickr")

test_that("tests for the flickr8k dataset", {
  skip_on_cran()

  t <- tempdir()

  expect_error(
    flickr8k <- flickr8k_dataset(root = tempfile()),
    class = "runtime_error"
  )

  flickr8k <- flickr8k_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(flickr8k, 6000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,598500)
  expect_equal((first_item$y[[1]]), "A black dog is running after a white dog in the snow .")
  expect_equal((first_item$y[[2]]), "Black dog chasing brown dog through snow")
  expect_equal((first_item$y[[3]]), "Two dogs chase each other across the snowy ground .")
  expect_equal((first_item$y[[4]]), "Two dogs play together in the snow .")
  expect_equal((first_item$y[[5]]), "Two dogs running through a low lying body of water .")

  flickr8k <- flickr8k_dataset(root = t, train = FALSE)
  expect_length(flickr8k, 1000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,502500)
  expect_equal((first_item$y[[1]]), "The dogs are in the snow in front of a fence .")
  expect_equal((first_item$y[[2]]), "The dogs play on the snow .")
  expect_equal((first_item$y[[3]]), "Two brown dogs playfully fight in the snow .")
  expect_equal((first_item$y[[4]]), "Two brown dogs wrestle in the snow .")
  expect_equal((first_item$y[[5]]), "Two dogs playing in the snow .")

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- sapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  flickr8k <- flickr8k_dataset(root = t, transform = transform_to_tensor)
  dl <- dataloader(flickr8k, batch_size = 4, collate_fn = resize_collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"character")
  expect_length(batch$y, 20)
  expect_equal(batch$y[1],"A black dog is running after a white dog in the snow .")
  expect_equal(batch$y[2],"Black dog chasing brown dog through snow")
  expect_equal(batch$y[3],"Two dogs chase each other across the snowy ground .")
  expect_equal(batch$y[4],"Two dogs play together in the snow .")

  unlink(file.path(t, "flickr8k"), recursive = TRUE)

})

test_that("tests for the flickr30k dataset", {
  skip_on_cran()

  if (Sys.info()[["sysname"]] == "Linux") {
    skip("Skipping on Ubuntu/Linux due to disk constraints.")
  }

  t <- tempdir()

  expect_error(
    flickr30k <- flickr30k_dataset(root = tempfile()),
    class = "runtime_error"
  )

  flickr30k <- flickr30k_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(flickr30k, 29000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,499500)
  expect_equal((first_item$y[[1]]), "Two young guys with shaggy hair look at their hands while hanging out in the yard.")
  expect_equal((first_item$y[[2]]), "Two young, White males are outside near many bushes.")
  expect_equal((first_item$y[[3]]), "Two men in green shirts are standing in a yard.")
  expect_equal((first_item$y[[4]]), "A man in a blue shirt standing in a garden.")
  expect_equal((first_item$y[[5]]), "Two friends enjoy time spent together.")

  flickr30k <- flickr30k_dataset(root = t, train = FALSE)
  expect_length(flickr30k, 1000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "integer")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,691500)
  expect_equal((first_item$y[[1]]), "The man with pierced ears is wearing glasses and an orange hat.")
  expect_equal((first_item$y[[2]]), "A man with glasses is wearing a beer can crocheted hat.")
  expect_equal((first_item$y[[3]]), "A man with gauges and glasses is wearing a Blitz hat.")
  expect_equal((first_item$y[[4]]), "A man in an orange hat starring at something.")
  expect_equal((first_item$y[[5]]), "A man wears an orange hat and glasses.")

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- sapply(batch, function(sample) sample$y)
    list(x = xs, y = ys)
  }
  flickr30k <- flickr30k_dataset(root = t, transform = transform_to_tensor)
  dl <- dataloader(flickr30k, batch_size = 4, collate_fn = resize_collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"character")
  expect_length(batch$y, 20)
  expect_equal(batch$y[1],"Two young guys with shaggy hair look at their hands while hanging out in the yard.")
  expect_equal(batch$y[2],"Two young, White males are outside near many bushes.")
  expect_equal(batch$y[3],"Two men in green shirts are standing in a yard.")
  expect_equal(batch$y[4],"A man in a blue shirt standing in a garden.")

  unlink(file.path(t, "flickr30k"), recursive = TRUE)
  
})