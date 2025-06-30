context("dataset-fgvc")

test_that("tests for the FGVC-Aircraft dataset", {
  t <- tempdir()

  skip_if(Sys.getenv("TORCHVISION_ALLOW_LARGE_TESTS") != "1",
        "Skipping test: set TORCHVISION_ALLOW_LARGE_TESTS=1 to enable tests requiring large downloads.")

  expect_error(
    fgvc_aircraft_dataset(root = tempfile(), split = "train", annotation_level = "variant", download = FALSE),
    class = "runtime_error"
  )

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(item) {
      torchvision::transform_resize(item$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- torch::torch_tensor(sapply(batch, function(item) item$y), dtype = torch::torch_long())
    list(x = xs, y = ys)
  }
  fgvc <- fgvc_aircraft_dataset(root = t, transform = transform_to_tensor, download = TRUE)
  dl <- torch::dataloader(dataset = fgvc,batch_size = 2,collate_fn = resize_collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_tensor_shape(batch$x,c(2,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_shape(batch$y,2)
  expect_tensor_dtype(batch$y,torch_long())
  expect_equal_to_r(batch$y[1],86)
  expect_equal_to_r(batch$y[2],42)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "variant")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 1854900)
  expect_type(item$y, "integer")
  expect_equal(item$y, 56)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "variant")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 2155620)
  expect_type(item$y, "integer")
  expect_equal(item$y, 3)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "family")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 1854900)
  expect_type(item$y, "integer")
  expect_equal(item$y, 32)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "family")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 2155620)
  expect_type(item$y, "integer")
  expect_equal(item$y, 16)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "manufacturer")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 1854900)
  expect_type(item$y, "integer")
  expect_equal(item$y, 13)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "manufacturer")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 2155620)
  expect_type(item$y, "integer")
  expect_equal(item$y, 5)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "all")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 2155620)
  expect_type(item$y, "integer")
  expect_equal(fgvc$classes$manufacturer[item$y[1]],"Boeing")
  expect_equal(fgvc$classes$family[item$y[2]],"Boeing 737")
  expect_equal(fgvc$classes$variant[item$y[3]],"737-200")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "all")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_type(item$x, "double")
  expect_length(item$x, 1854900)
  expect_type(item$y, "integer")
  expect_equal(fgvc$classes$manufacturer[item$y[1]],"Douglas Aircraft Company")
  expect_equal(fgvc$classes$family[item$y[2]],"DC-8")
  expect_equal(fgvc$classes$variant[item$y[3]],"DC-8")

    resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(item) {
      torchvision::transform_resize(item$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- torch::torch_tensor(sapply(batch, function(item) item$y), dtype = torch::torch_long())
    list(x = xs, y = ys)
  }
  fgvc <- fgvc_aircraft_dataset(root = t, annotation_level = "all", transform = transform_to_tensor)
  dl <- torch::dataloader(dataset = fgvc,batch_size = 2,collate_fn = resize_collate_fn)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_tensor_shape(batch$x,c(2,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_shape(batch$y,c(3,2))
  expect_tensor_dtype(batch$y,torch_long())
  expect_equal_to_r(batch$y[1],c(22,5))
  expect_equal_to_r(batch$y[2],c(58,14))
  expect_equal_to_r(batch$y[3],c(86,42))

  unlink(file.path(t, "fgvc-aircraft-2013b"), recursive = TRUE)
})