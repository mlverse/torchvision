test_that("cityscapes_dataset validates split parameter", {
  skip_on_cran()
  
  expect_error(
    cityscapes_dataset(
      root = tempdir(),
      split = "invalid"
    ),
    "split must be one of"
  )
  
  expect_error(
    cityscapes_dataset(
      root = tempdir(),
      split = "training"
    ),
    "split must be one of"
  )
})

test_that("cityscapes_dataset validates mode parameter", {
  skip_on_cran()
  
  expect_error(
    cityscapes_dataset(
      root = tempdir(),
      mode = "invalid"
    ),
    "mode must be either"
  )
  
  expect_error(
    cityscapes_dataset(
      root = tempdir(),
      mode = "medium"
    ),
    "mode must be either"
  )
})

test_that("cityscapes_dataset validates target_type parameter", {
  skip_on_cran()
  
  expect_error(
    cityscapes_dataset(
      root = tempdir(),
      target_type = "invalid"
    ),
    "target_type must be one or more of"
  )
  
  expect_error(
    cityscapes_dataset(
      root = tempdir(),
      target_type = c("instance", "invalid", "semantic")
    ),
    "Got invalid types"
  )
})

test_that("cityscapes_dataset requires dataset to exist", {
  skip_on_cran()
  
  temp_root <- file.path(tempdir(), "nonexistent_cityscapes")
  
  expect_error(
    cityscapes_dataset(
      root = temp_root,
      split = "train"
    ),
    "Cityscapes dataset not found"
  )
})

test_that("cityscapes_dataset structure is correct with fine mode", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "instance"
  )
  
  expect_s3_class(ds, "cityscapes")
  expect_s3_class(ds, "dataset")
  expect_true(ds$.length() > 0)
  expect_equal(length(ds$classes), 19)
})

test_that("cityscapes_dataset loads images correctly", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "instance",
    transform = transform_to_tensor
  )
  
  item <- ds[1]
  
  expect_type(item, "list")
  expect_true("x" %in% names(item))
  expect_true("y" %in% names(item))
  expect_true("instance" %in% names(item$y))
  
  # Check image tensor shape (should be 3, H, W after transform_to_tensor)
  expect_s3_class(item$x, "torch_tensor")
  expect_equal(item$x$dim(), 3)
  expect_equal(as.integer(item$x$shape[1]), 3)
  
  # Check mask shape (should be H, W)
  expect_true(is.matrix(item$y$instance) || is.array(item$y$instance))
})

test_that("cityscapes_dataset supports semantic target type", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "semantic",
    transform = transform_to_tensor
  )
  
  item <- ds[1]
  expect_true("semantic" %in% names(item$y))
  expect_true(is.matrix(item$y$semantic) || is.array(item$y$semantic))
})

test_that("cityscapes_dataset supports multiple target types", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = c("instance", "semantic"),
    transform = transform_to_tensor
  )
  
  item <- ds[1]
  expect_true("instance" %in% names(item$y))
  expect_true("semantic" %in% names(item$y))
  expect_type(item$y, "list")
  expect_equal(length(item$y), 2)
})

test_that("cityscapes_dataset supports color target type", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "color"
  )
  
  item <- ds[1]
  expect_true("color" %in% names(item$y))
  
  # Color should be (H, W, 3)
  if (!is.null(item$y$color)) {
    expect_equal(length(dim(item$y$color)), 3)
    expect_equal(dim(item$y$color)[3], 3)
  }
})

test_that("cityscapes_dataset supports polygon target type", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "polygon"
  )
  
  item <- ds[1]
  expect_true("polygon" %in% names(item$y))
  
  # Polygon data should be a list structure
  if (!is.null(item$y$polygon)) {
    expect_type(item$y$polygon, "list")
  }
})

test_that("cityscapes_dataset works with coarse mode", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes/gtCoarse"), "Cityscapes coarse annotations not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "train",
    mode = "coarse",
    target_type = "instance",
    transform = transform_to_tensor
  )
  
  expect_s3_class(ds, "cityscapes")
  expect_true(ds$.length() > 0)
  
  item <- ds[1]
  expect_true("instance" %in% names(item$y))
})

test_that("cityscapes_dataset works with train split", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "train",
    mode = "fine",
    target_type = "instance"
  )
  
  expect_s3_class(ds, "cityscapes")
  # Train split should have ~2975 images
  expect_true(ds$.length() > 2000)
})

test_that("cityscapes_dataset works with val split", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "instance"
  )
  
  expect_s3_class(ds, "cityscapes")
  # Val split should have 500 images
  expect_true(ds$.length() > 400 && ds$.length() < 600)
})

test_that("cityscapes_dataset works with test split", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "test",
    mode = "fine",
    target_type = "instance"
  )
  
  expect_s3_class(ds, "cityscapes")
  # Test split should have ~1525 images
  expect_true(ds$.length() > 1400)
})

test_that("cityscapes_dataset output is compatible with draw_segmentation_masks", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "instance",
    transform = transform_to_tensor
  )
  
  item <- ds[1]
  
  # Create boolean mask from instance IDs
  mask_tensor <- torch::torch_tensor(item$y$instance > 0)$unsqueeze(1)
  
  # This should not error
  expect_no_error({
    overlay <- draw_segmentation_masks(
      item$x,
      mask_tensor,
      alpha = 0.5
    )
  })
})

test_that("cityscapes_dataset works with dataloader", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "instance",
    transform = transform_to_tensor
  )
  
  dl <- torch::dataloader(ds, batch_size = 2, shuffle = FALSE)
  
  expect_s3_class(dl, "dataloader")
  
  # Get first batch
  batch <- dl$.iter()$.next()
  
  expect_type(batch, "list")
  expect_true("x" %in% names(batch))
  expect_true("y" %in% names(batch))
})

test_that("cityscapes_dataset has correct class names", {
  skip_on_cran()
  
  # Check class names are defined
  ds_classes <- c(
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
  )
  
  expect_equal(length(ds_classes), 19)
  
  # These are the standard Cityscapes evaluation classes
  expect_true("person" %in% ds_classes)
  expect_true("car" %in% ds_classes)
  expect_true("road" %in% ds_classes)
})

test_that("cityscapes_dataset returns image_with_segmentation_mask class", {
  skip_on_cran()
  skip_if_not(dir.exists("~/datasets/cityscapes"), "Cityscapes dataset not available")
  skip_if_not(torch::torch_is_installed())
  
  ds <- cityscapes_dataset(
    root = "~/datasets/cityscapes",
    split = "val",
    mode = "fine",
    target_type = "instance"
  )
  
  item <- ds[1]
  
  expect_s3_class(item, "image_with_segmentation_mask")
  expect_s3_class(item, "list")
})
