test_that("eurosat_dataset handles temporary directories correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  ds <- eurosat_dataset(download = TRUE)
  
  expect_true(!is.null(ds), info = "Dataset should load successfully")
  expect_true(length(ds) > 0, info = "Dataset should have a non-zero length")
  
  temp_root <- ds$root
  expect_true(dir.exists(temp_root), info = "Temporary directory should exist")
  
  files <- list.files(temp_root, recursive = TRUE, full.names = TRUE)
  expect_true(length(files) > 0, info = "Files should be downloaded in the temporary directory")
  
  extracted_dir <- file.path(temp_root, "2750")
  expect_true(dir.exists(extracted_dir), info = "Extracted data folder should exist")
  
  image_files <- list.files(extracted_dir, pattern = "\\.jpg$", recursive = TRUE, full.names = TRUE)
  expect_true(length(image_files) > 0, info = "Image files should be present in the extracted directory")
})

