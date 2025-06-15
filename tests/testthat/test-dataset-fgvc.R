context("dataset-fgvc")

test_that("tests for the FGVC-Aircraft dataset", {
  t <- tempfile()

  expect_error(
    fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = FALSE),
    class = "runtime_error"
  )

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = TRUE)
  expect_length(fgvc, 3334)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 1)

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(item) {
      torchvision::transform_resize(item$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- torch::torch_tensor(sapply(batch, function(item) item$y), dtype = torch::torch_long())
    list(x = xs, y = ys)
  }
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
  expect_equal_to_r(batch$y[1],1)
  expect_equal_to_r(batch$y[2],1)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "variant")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,802,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 1)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "variant")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 1)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "variant")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,882,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 1)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "family")
  expect_length(fgvc, 3334)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 13)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "family")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,802,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 13)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "family")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 13)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "family")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,882,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 13)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "manufacturer")
  expect_length(fgvc, 3334)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 5)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "manufacturer")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,802,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 5)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "manufacturer")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 5)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "manufacturer")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,882,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 5)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "all")
  expect_length(fgvc, 3334)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "list")
  expect_equal(fgvc$classes$manufacturer[item$y$manufacturer],"Boeing")
  expect_equal(fgvc$classes$family[item$y$family],"Boeing 707")
  expect_equal(fgvc$classes$variant[item$y$variant],"707-320")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "all")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,802,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "list")
  expect_equal(fgvc$classes$manufacturer[item$y$manufacturer],"Boeing")
  expect_equal(fgvc$classes$family[item$y$family],"Boeing 707")
  expect_equal(fgvc$classes$variant[item$y$variant],"707-320")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "all")
  expect_length(fgvc, 3333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,882,1200))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "list")
  expect_equal(fgvc$classes$manufacturer[item$y$manufacturer],"Boeing")
  expect_equal(fgvc$classes$family[item$y$family],"Boeing 707")
  expect_equal(fgvc$classes$variant[item$y$variant],"707-320")

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "all")
  expect_length(fgvc, 6667)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_tensor(item$x)
  expect_tensor_shape(item$x,c(3,695,1024))
  expect_tensor_dtype(item$x,torch_float())
  expect_type(item$y, "list")
  expect_equal(fgvc$classes$manufacturer[item$y$manufacturer],"Boeing")
  expect_equal(fgvc$classes$family[item$y$family],"Boeing 707")
  expect_equal(fgvc$classes$variant[item$y$variant],"707-320")
})