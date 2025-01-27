test_that("eurosat_dataset API handles splits correctly via API", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  temp_root <- "./data/eurosat_test"
  
  if (!dir.exists(temp_root)) {
    dir.create(temp_root, recursive = TRUE)
  }
  
  ds <- eurosat_dataset(root = temp_root, split = "train", download = TRUE)
  
  expect_true(!is.null(ds), "Dataset should load successfully")
  expect_true(length(ds) > 0, "Dataset should have a non-zero length")
  
  expect_true(dir.exists(temp_root), "Temporary directory should exist")
  
  extracted_dir <- list.dirs(temp_root, recursive = TRUE, full.names = TRUE)
  extracted_dir <- extracted_dir[grepl("/2750$", extracted_dir)]
  expect_true(length(extracted_dir) > 0, "Extracted data folder should exist")
  
  image_files <- list.files(extracted_dir, pattern = "\\.jpg$", recursive = TRUE, full.names = TRUE)
  expect_true(length(image_files) > 0, "Image files should be present in the extracted directory")
  
  train_ds <- eurosat_dataset(root = temp_root, split = "train", download = TRUE)
  expect_equal(length(train_ds), 16200, tolerance = 0, label = "Train dataset should have exactly 16200 samples")
  
  val_ds <- eurosat_dataset(root = temp_root, split = "val", download = TRUE)
  expect_equal(length(val_ds), 5400, tolerance = 0, label = "Validation dataset should have exactly 5400 samples")
  
  test_ds <- eurosat_dataset(root = temp_root, split = "test", download = TRUE)
  expect_equal(length(test_ds), 5400, tolerance = 0, label = "Test dataset should have exactly 5400 samples")
  
  sample <- train_ds[1]
  
  expect_true(!is.null(sample$x), "Image should not be null")
  expect_equal(dim(sample$x), c(64, 64, 3), label = "Image tensor should have shape (64, 64, 3)")
  expect_equal(typeof(sample$x), "double", label = "Image tensor should have data type 'double'")
  
  expect_true(!is.null(sample$y), "Label should not be null")
  expect_equal(typeof(sample$y), "character", label = "Label should have data type 'character'")
})