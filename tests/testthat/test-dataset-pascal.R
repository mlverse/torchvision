context('dataset-pascal')

t = withr::local_tempdir()

test_that("tests for the Pascal VOC Segmentation dataset for train split for year 2007", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2007', split = 'train', download = TRUE)
  expect_length(pascal, 209)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for test split for year 2007", {
  
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  
  pascal <- pascal_segmentation_dataset(root = t, year = '2007', split = 'test', download = TRUE)
  expect_length(pascal, 210)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,375,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 2)
  expect_equal(first_item$y$labels, c(1, 4))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for trainval split for year 2007", {
  
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  
  pascal <- pascal_segmentation_dataset(root = t, year = '2007', split = 'trainval', download = TRUE)
  expect_length(pascal, 422)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for val split for year 2007", {
  
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  
  pascal <- pascal_segmentation_dataset(root = t, year = '2007', split = 'val', download = TRUE)
  expect_length(pascal, 213)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,375,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 2)
  expect_equal(first_item$y$labels, c(1, 21))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for train split for year 2008", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2008', split = 'train', download = TRUE)
  expect_length(pascal, 511)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for trainval split for year 2008", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2008', split = 'trainval', download = TRUE)
  expect_length(pascal, 1023)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for val split for year 2008", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2008', split = 'val', download = TRUE)
  expect_length(pascal, 512)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,549000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,366,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 2)
  expect_equal(first_item$y$labels, c(1, 2))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for train split for year 2009", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2009', split = 'train', download = TRUE)
  expect_length(pascal, 749)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for trainval split for year 2009", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2009', split = 'trainval', download = TRUE)
  expect_length(pascal, 1499)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for val split for year 2009", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2009', split = 'val', download = TRUE)
  expect_length(pascal, 750)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,549000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,366,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 2)
  expect_equal(first_item$y$labels, c(1, 2))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for train split for year 2010", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2010', split = 'train', download = TRUE)
  expect_length(pascal, 964)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for trainval split for year 2010", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2010', split = 'trainval', download = TRUE)
  expect_length(pascal, 1928)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for val split for year 2010", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2010', split = 'val', download = TRUE)
  expect_length(pascal, 964)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,549000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,366,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 2)
  expect_equal(first_item$y$labels, c(1, 2))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for train split for year 2011", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2011', split = 'train', download = TRUE)
  expect_length(pascal, 1112)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for trainval split for year 2011", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2011', split = 'trainval', download = TRUE)
  expect_length(pascal, 2223)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for val split for year 2011", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2011', split = 'val', download = TRUE)
  expect_length(pascal, 1111)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,549000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,366,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 2)
  expect_equal(first_item$y$labels, c(1, 2))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for train split for year 2012", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2012', split = 'train', download = TRUE)
  expect_length(pascal, 1464)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for trainval split for year 2012", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2012', split = 'trainval', download = TRUE)
  expect_length(pascal, 2913)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,421500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,281,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 3)
  expect_equal(first_item$y$labels, c(1, 2, 16))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC Segmentation dataset for val split for year 2012", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_segmentation_dataset(root = t, year = '2012', split = 'val', download = TRUE)
  expect_length(pascal, 1449)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,549000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$mask)
  expect_tensor_shape(first_item$y$mask,c(1,366,500))
  expect_tensor_dtype(first_item$y$mask,torch_bool())
  expect_type(first_item$y$labels, "double")
  expect_length(first_item$y$labels, 2)
  expect_equal(first_item$y$labels, c(1, 2))
  expect_s3_class(first_item, "image_with_segmentation_mask")
})

test_that("tests for the Pascal VOC detection dataset for train split for year 2007", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2007', split = 'train', download = TRUE)
  expect_length(pascal, 2501)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,499500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for test split for year 2007", {
  
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  
  pascal <- pascal_detection_dataset(root = t, year = '2007', split = 'test', download = TRUE)
  expect_length(pascal, 4952)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,529500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(2,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 2)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for trainval split for year 2007", {
  
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  
  pascal <- pascal_detection_dataset(root = t, year = '2007', split = 'trainval', download = TRUE)
  expect_length(pascal, 5011)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(5,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 5)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for val split for year 2007", {
  
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  
  pascal <- pascal_detection_dataset(root = t, year = '2007', split = 'val', download = TRUE)
  expect_length(pascal, 2510)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(5,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 5)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for train split for year 2008", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2008', split = 'train', download = TRUE)
  expect_length(pascal, 2111)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,663000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(2,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 2)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for trainval split for year 2008", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2008', split = 'trainval', download = TRUE)
  expect_length(pascal, 4332)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for val split for year 2008", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2008', split = 'val', download = TRUE)
  expect_length(pascal, 2221)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for train split for year 2009", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2009', split = 'train', download = TRUE)
  expect_length(pascal, 3473)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,663000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(2,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 2)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for trainval split for year 2009", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2009', split = 'trainval', download = TRUE)
  expect_length(pascal, 7054)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for val split for year 2009", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2009', split = 'val', download = TRUE)
  expect_length(pascal, 3581)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for train split for year 2010", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2010', split = 'train', download = TRUE)
  expect_length(pascal, 4998)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,663000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(2,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 2)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for trainval split for year 2010", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2010', split = 'trainval', download = TRUE)
  expect_length(pascal, 10103)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for val split for year 2010", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2010', split = 'val', download = TRUE)
  expect_length(pascal, 5105)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for train split for year 2011", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2011', split = 'train', download = TRUE)
  expect_length(pascal, 5717)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,663000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(2,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 2)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for trainval split for year 2011", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2011', split = 'trainval', download = TRUE)
  expect_length(pascal, 11540)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for val split for year 2011", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2011', split = 'val', download = TRUE)
  expect_length(pascal, 5823)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for train split for year 2012", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2012', split = 'train', download = TRUE)
  expect_length(pascal, 5717)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,663000)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(2,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 2)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for trainval split for year 2012", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2012', split = 'trainval', download = TRUE)
  expect_length(pascal, 11540)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})

test_that("tests for the Pascal VOC detection dataset for val split for year 2012", {

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
        "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  pascal <- pascal_detection_dataset(root = t, year = '2012', split = 'val', download = TRUE)
  expect_length(pascal, 5823)
  first_item <- pascal[1]
  expect_named(first_item, c("x", "y"))
  expect_length(first_item$x,562500)
  expect_type(first_item$x, "double")
  expect_type(first_item$y, "list")
  expect_tensor(first_item$y$boxes)
  expect_tensor_shape(first_item$y$boxes,c(1,4))
  expect_tensor_dtype(first_item$y$boxes,torch_int64())
  expect_type(first_item$y$labels,"character")
  expect_length(first_item$y$labels, 1)
  expect_s3_class(first_item, "image_with_bounding_box")
})