test_that("eurosat_dataset handles temporary directories correctly", {
  skip_on_cran()
  skip_if_not_installed("torch")
  
  ds <- eurosat_dataset(download = TRUE)
  expect_true(!is.null(ds), "Dataset should load successfully")
  expect_true(length(ds) > 0, "Dataset should have a non-zero length")
  
  temp_root <- ds$root
  expect_true(dir.exists(temp_root), "Temporary directory should exist")
  files <- list.files(temp_root, recursive = TRUE, full.names = TRUE)
  expect_true(length(files) > 0, "Files should be downloaded in the temporary directory")
})
