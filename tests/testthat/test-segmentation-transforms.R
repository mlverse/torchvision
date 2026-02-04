context("segmentation-transforms")

# Helper to create mock COCO target
mock_coco_target <- function(h = 100, w = 100) {
  list(
    boxes = torch::torch_tensor(matrix(c(10, 20, 50, 60), nrow = 1)),
    labels = "cat",
    area = torch::torch_tensor(c(800)),
    iscrowd = torch::torch_tensor(FALSE),
    segmentation = list(list(c(10, 20, 50, 20, 50, 60, 10, 60))),
    image_height = h,
    image_width = w
  )
}

# Helper to create mock trimap
mock_trimap <- function(h = 3, w = 3) {
  torch::torch_tensor(
    array(rep(c(1, 2, 3), length.out = h * w), dim = c(h, w)),
    dtype = torch::torch_int32()
  )
}

# COCO mask transformation tests

test_that("target_transform_coco_masks() converts polygons to masks", {
  skip_if_not_installed("torch")
  skip_if_not_installed("magick")
  y <- mock_coco_target()
  result <- target_transform_coco_masks(y)
  expect_true("masks" %in% names(result))
  expect_tensor(result$masks)
  expect_tensor_dtype(result$masks, torch::torch_bool())
  expect_equal(result$masks$ndim, 3)
  expect_true("segmentation" %in% names(result))
})

test_that("target_transform_coco_masks() handles empty segmentation", {
  skip_if_not_installed("torch")
  y <- list(
    boxes = torch::torch_zeros(c(0, 4)),
    labels = character(),
    segmentation = list(),
    image_height = 100,
    image_width = 100
  )
  result <- target_transform_coco_masks(y)
  expect_tensor(result$masks)
  expect_equal(result$masks$shape[1], 0)
})

test_that("target_transform_coco_masks() handles multiple objects", {
  skip_if_not_installed("torch")
  skip_if_not_installed("magick")
  y <- list(
    boxes = torch::torch_tensor(matrix(c(10,20,50,60, 60,10,90,40), nrow = 2, byrow = TRUE)),
    labels = c("cat", "dog"),
    segmentation = list(
      list(c(10, 20, 50, 20, 50, 60, 10, 60)),
      list(c(60, 10, 90, 10, 90, 40, 60, 40))
    ),
    image_height = 100,
    image_width = 100
  )
  result <- target_transform_coco_masks(y)
  expect_equal(result$masks$shape, c(2, 100, 100))
})

# Trimap mask transformation tests

test_that("target_transform_trimap_masks() converts trimap to masks", {
  skip_if_not_installed("torch")
  y <- list(trimap = mock_trimap(), label = 1L)
  result <- target_transform_trimap_masks(y)
  expect_true("masks" %in% names(result))
  expect_tensor(result$masks)
  expect_tensor_dtype(result$masks, torch::torch_bool())
  expect_equal(result$masks$shape[1], 3)
  expect_equal(result$label, 1L)
})

test_that("target_transform_trimap_masks() creates mutually exclusive masks", {
  skip_if_not_installed("torch")
  y <- list(trimap = mock_trimap(), label = 1L)
  result <- target_transform_trimap_masks(y)
  mask_sum <- result$masks$sum(dim = 1)
  expect_true(all(as.array(mask_sum) == 1))
})

# Dataset behavior tests (skipped unless TEST_LARGE_DATASETS=1)

test_that("coco_segmentation_dataset with target_transform produces masks", {
  skip_on_cran()
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Set TEST_LARGE_DATASETS=1 to enable")
  tmp <- withr::local_tempdir()
  ds <- coco_segmentation_dataset(root = tmp, train = FALSE, year = "2017",
                               download = TRUE, target_transform = target_transform_coco_masks)
  y <- ds[1]$y
  expect_true("masks" %in% names(y))
  expect_tensor(y$masks)
  expect_equal(y$masks$ndim, 3)
  expect_tensor_dtype(y$masks, torch::torch_bool())
})

# Integration with draw_segmentation_masks()

test_that("transformed masks work with draw_segmentation_masks()", {
  skip_if_not_installed("torch")
  skip_if_not_installed("magick")
  image <- torch::torch_randint(100, 200, c(3, 50, 50), dtype = torch::torch_uint8())
  y <- list(trimap = mock_trimap(50, 50), label = 1L)
  transformed_y <- target_transform_trimap_masks(y)
  item <- list(x = image, y = transformed_y)
  class(item) <- c("image_with_segmentation_mask", class(item))
  result <- draw_segmentation_masks(item)
  expect_tensor(result)
  expect_equal(result$shape[1], 3)
})

# Existing helper tests

test_that("coco_polygon_to_mask handles single polygon", {
  skip_if_not_installed("torch")
  skip_if_not_installed("magick")
  polygon <- list(c(10, 10, 50, 10, 50, 50, 10, 50))
  mask <- coco_polygon_to_mask(polygon, 100, 100)
  expect_tensor(mask)
  expect_tensor_dtype(mask, torch::torch_bool())
  expect_tensor_shape(mask, c(100, 100))
  expect_true(as.array(mask[30, 30]))
})

test_that("coco_polygon_to_mask handles empty polygon", {
  skip_if_not_installed("torch")
  skip_if_not_installed("magick")
  mask <- coco_polygon_to_mask(list(), 100, 100)
  expect_tensor(mask)
  expect_equal_to_r(mask$sum(), 0)
})
