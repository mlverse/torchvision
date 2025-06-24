context("dataset-caltech")

t <- withr::local_tempdir()

test_that("Caltech101 dataset works correctly", {

  expect_error(
    caltech101_detection_dataset(root = tempfile(), download = FALSE),
    class = "rlang_error"
  )

  ds <- caltech101_detection_dataset(root = t, download = TRUE)
  expect_equal(length(ds), 8677)
  first_item <- ds[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_length(first_item$x, 234000)
  expect_type(first_item$y,"list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_float())
  expect_type(first_item$y$labels, "integer")
  expect_equal(first_item$y$labels,1)
  expect_tensor(first_item$y$contour)
  expect_tensor_shape(first_item$y$contour,c(1,20,2))
  expect_tensor_dtype(first_item$y$contour,torch_float())
})

test_that("Caltech101 dataset works correctly (dataloader)", {

  resize_contour <- function(contour, target_n = 15) {
    n <- contour$size(2)
    idx <- torch_linspace(1, n, target_n)
    idx_floor <- idx$floor()$to(dtype = torch_long())
    contour_interp <- contour[ , idx_floor, ]
    contour_interp
  }
  caltech101 <- caltech101_detection_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    },
    target_transform = function(y) {
      y$contour <- resize_contour(y$contour)
      y
    }
  )
  dl <- dataloader(caltech101, batch_size = 4)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_length(batch, 2)
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"list")
  expect_tensor(batch$y$labels)
  expect_tensor_dtype(batch$y$labels, torch_long())
  expect_tensor_shape(batch$y$labels, 4)
  expect_equal_to_r(batch$y$labels[1],1)
  expect_tensor(batch$y$boxes)
  expect_tensor_dtype(batch$y$boxes, torch_float())
  expect_tensor_shape(batch$y$boxes, c(4,1,4))
  expect_tensor(batch$y$contour)
  expect_tensor_dtype(batch$y$contour, torch_float())
  expect_tensor_shape(batch$y$contour, c(4,1,15,2))

})

test_that("Caltech256 dataset works correctly", {

  expect_error(
    caltech256_dataset(root = tempfile(), download = FALSE),
    class = "rlang_error"
  )

  caltech256 <- caltech256_dataset(root = t, download = TRUE)
  expect_length(caltech256, 30607)
  first_item <- caltech256[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,416166)
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("Caltech256 dataset works correctly (dataloader)", {

  caltech256 <- caltech256_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(caltech256, batch_size = 4)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_length(batch, 2)
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_dtype(batch$y, torch_long())
  expect_tensor_shape(batch$y, 4)
  expect_tensor(batch$y[1])
  expect_tensor_dtype(batch$y[1],torch_long())
  expect_equal_to_r(batch$y[1],1)
  expect_equal_to_r(batch$y[2],1)
  expect_equal_to_r(batch$y[3],1)
  expect_equal_to_r(batch$y[4],1)
  
})