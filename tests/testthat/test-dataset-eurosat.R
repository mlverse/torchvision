test_that("eurosat_dataset API handles splits correctly via API", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  # Define the root directory for testing
  temp_root <- "./data/eurosat_test"
  
  # Ensure the root directory exists
  if (!dir.exists(temp_root)) {
    dir.create(temp_root, recursive = TRUE)
  }
  
  # Initialize dataset with root and default arguments
  ds <- eurosat_dataset(root = temp_root, split = "train", download = TRUE)
  expect_true(!is.null(ds), "Dataset should load successfully")
  expect_true(length(ds) > 0, "Dataset should have a non-zero length")
  
  # Check the root directory exists
  expect_true(dir.exists(temp_root), "Temporary directory should exist")
  
  # Check the extracted directory structure
  extracted_dir <- list.dirs(temp_root, recursive = TRUE, full.names = TRUE)
  extracted_dir <- extracted_dir[grepl("/2750$", extracted_dir)]
  expect_true(length(extracted_dir) > 0, "Extracted data folder should exist")
  
  # Check image files in the extracted directory
  image_files <- list.files(extracted_dir, pattern = "\\.jpg$", recursive = TRUE, full.names = TRUE)
  expect_true(length(image_files) > 0, "Image files should be present in the extracted directory")
  
  # Test train split
  train_ds <- eurosat_dataset(root = temp_root, split = "train", download = TRUE)
  expect_equal(length(train_ds), 16200, tolerance = 0, label = "Train dataset should have exactly 16200 samples")
  
  # Test validation split
  val_ds <- eurosat_dataset(root = temp_root, split = "val", download = TRUE)
  expect_equal(length(val_ds), 5400, tolerance = 0, label = "Validation dataset should have exactly 5400 samples")
  
  # Test test split
  test_ds <- eurosat_dataset(root = temp_root, split = "test", download = TRUE)
  expect_equal(length(test_ds), 5400, tolerance = 0, label = "Test dataset should have exactly 5400 samples")
  
  # Test data content
  sample <- train_ds[1]
  expect_true(!is.null(sample$x), "Image should not be null")
  expect_true(!is.null(sample$y), "Label should not be null")
})
