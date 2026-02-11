context("dataset-oxfordiiitpet")

t <- withr::local_tempdir()

test_that("tests for the Oxford-IIIT Pet Segmentation dataset for train split with target type category", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_segmentation_dataset(root = t, target_type = "category", train = TRUE, download = TRUE,
                                                      target_transform = target_transform_trimap_masks)
  expect_length(oxfordiiitpet, 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,591000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(3,500,394))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$label, "integer")
  expect_equal(first_item$y$label, 1)
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Oxford-IIIT Pet Segmentation dataset for train split with target type binary category", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_segmentation_dataset(root = t, target_type = "binary-category", train = TRUE,
                                                      target_transform = target_transform_trimap_masks)
  expect_length(oxfordiiitpet, 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,591000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(3,500,394))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$label,"integer")
  expect_equal(first_item$y$label, 1)
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Oxford-IIIT Pet Segmentation dataset for test split with target type category", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_segmentation_dataset(root = t, target_type = "category", train = FALSE,
                                                      target_transform = target_transform_trimap_masks)
  expect_length(oxfordiiitpet, 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,202500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(3,225,300))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$label,"integer")
  expect_equal(first_item$y$label, 1)
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Oxford-IIIT Pet Segmentation dataset for test split with target type binary category", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_segmentation_dataset(root = t, target_type = "binary-category", train = FALSE,
                                                      target_transform = target_transform_trimap_masks)
  expect_length(oxfordiiitpet, 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,202500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(3,225,300))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$label,"integer")
  expect_equal(first_item$y$label, 1)
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Oxford-IIIT Pet Segmentation dataset for dataloader", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_segmentation_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    },
    target_transform = function(y) {
       y <- target_transform_trimap_masks(y)
       y$masks <- y$masks %>% transform_resize(c(224, 224))
      y
    }
  )
  expect_s3_class(oxfordiiitpet[1], "image_with_segmentation_mask")
  dl <- dataloader(oxfordiiitpet, batch_size = 4)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"list")
  expect_length(batch$y, 2)
  expect_type(batch$y[1], "list")
  expect_tensor(batch$y$mask)
  expect_tensor_shape(batch$y$mask, c(4,3,224,224))
  expect_tensor_dtype(batch$y$mask, torch_bool())
  expect_tensor(batch$y$label)
  expect_tensor_shape(batch$y$label, 4)
  expect_tensor_dtype(batch$y$label, torch_long())
  expect_equal_to_r(batch$y$label[1], 1)
  expect_equal_to_r(batch$y$label[2], 1)
  expect_equal_to_r(batch$y$label[3], 1)
  expect_equal_to_r(batch$y$label[4], 1)
})

test_that("tests for the Oxford-IIIT Pet dataset for train split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(oxfordiiitpet, 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,591000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Oxford-IIIT Pet dataset for test split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_dataset(root = t, train = FALSE)
  expect_length(oxfordiiitpet, 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,202500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Oxford-IIIT Pet dataset for dataloader", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(oxfordiiitpet, batch_size = 4)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_dtype(batch$y,torch_long())
  expect_tensor_shape(batch$y,4)
  expect_equal_to_r(batch$y[1], 1)
  expect_equal_to_r(batch$y[2], 1)
  expect_equal_to_r(batch$y[3], 1)
  expect_equal_to_r(batch$y[4], 1)
})

test_that("tests for the Oxford-IIIT Pet binary dataset for train split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_binary_dataset(root = t, train = TRUE)
  expect_length(oxfordiiitpet, 3680)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,591000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Oxford-IIIT Pet binary dataset for test split", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_binary_dataset(root = t, train = FALSE)
  expect_length(oxfordiiitpet, 3669)
  first_item <- oxfordiiitpet[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,202500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"integer")
  expect_equal(first_item$y, 1)
})

test_that("tests for the Oxford-IIIT Pet binary dataset for dataloader", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  oxfordiiitpet <- oxfordiiitpet_binary_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    }
  )
  dl <- dataloader(oxfordiiitpet, batch_size = 4)
  batch <- dataloader_next(dataloader_make_iter(dl))
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_tensor(batch$y)
  expect_tensor_dtype(batch$y,torch_long())
  expect_tensor_shape(batch$y,4)
  expect_equal_to_r(batch$y[1], 1)
  expect_equal_to_r(batch$y[2], 1)
  expect_equal_to_r(batch$y[3], 1)
  expect_equal_to_r(batch$y[4], 1)
})
