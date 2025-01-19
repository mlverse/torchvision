test_that("eurosat_dataset handles dataset correctly with secure download", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  ds <- eurosat_dataset(download = TRUE)
  expect_true(!is.null(ds), info = "Dataset should load successfully")
  expect_true(length(ds) > 0, info = "Dataset should have a non-zero length")
  
  temp_root <- ds$root
  expect_true(dir.exists(temp_root), info = "Temporary directory should exist")
  
  extracted_dir <- list.dirs(temp_root, recursive = TRUE, full.names = TRUE)
  extracted_dir <- extracted_dir[grepl("/2750$", extracted_dir)]
  expect_true(length(extracted_dir) > 0, info = "Extracted data folder should exist")
  
  image_files <- list.files(extracted_dir, pattern = "\\.jpg$", recursive = TRUE, full.names = TRUE)
  expect_true(length(image_files) > 0, info = "Image files should be present in the extracted directory")
})

