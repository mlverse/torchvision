test_that("RF100 catalog data exists and has correct structure", {
  skip_on_cran()

  # Load catalog
  catalog <- get_rf100_catalog()

  # Should be a data frame
  expect_s3_class(catalog, "data.frame")

  # Should have expected columns
  expected_cols <- c("collection", "dataset", "description", "task", "nlevels",
                     "num_images", "image_width", "image_height",
                     "train_size_mb", "test_size_mb", "valid_size_mb", "total_size_mb",
                     "has_train", "has_test", "has_valid",
                     "function_name", "roboflow_url")
  expect_length(setdiff(expected_cols, names(catalog)), 0)

  # Should have datasets (allow for future additions)
  expect_gte(nrow(catalog), 34)

  # All datasets should have descriptions
  expect_true(all(nchar(catalog$description) > 10))

  # All datasets should have positive sizes
  expect_true(all(catalog$total_size_mb > 0))

  # All datasets should have positive image counts
  expect_true(all(catalog$num_images > 0))

  # All datasets should have positive dimensions
  expect_true(all(catalog$image_width > 0))
  expect_true(all(catalog$image_height > 0))

  # Check collections are valid
  valid_collections <- c("biology", "medical", "infrared", "damage", "underwater", "document")
  expect_length(setdiff(catalog$collection, valid_collections), 0)

  # All datasets should have all three splits
  expect_true(all(catalog$has_train))
  expect_true(all(catalog$has_test))
  expect_true(all(catalog$has_valid))
})

test_that("search_rf100 works with keyword search", {
  skip_on_cran()

  # Search for cell-related datasets
  cell_results <- search_rf100("cell")
  expect_s3_class(cell_results, "data.frame")
  expect_gt(nrow(cell_results), 0)

  # All results should contain "cell" in name or description
  has_cell <- sapply(seq_len(nrow(cell_results)), function(i) {
    grepl("cell", cell_results$dataset[i], ignore.case = TRUE) ||
    grepl("cell", cell_results$description[i], ignore.case = TRUE)
  })
  expect_true(all(has_cell))

  # Search for solar datasets
  solar_results <- search_rf100("solar")
  expect_s3_class(solar_results, "data.frame")
  expect_gt(nrow(solar_results), 0)

  # Search for non-existent keyword
  expect_message(
    result <- search_rf100("nonexistent_keyword_xyz"),
    "No datasets found"
  )
  expect_null(result)
})

test_that("search_rf100 works with collection filter", {
  skip_on_cran()

  # Search biology collection
  biology_results <- search_rf100(collection = "biology")
  expect_s3_class(biology_results, "data.frame")
  expect_gte(nrow(biology_results), 10)
  expect_true(all(biology_results$collection == "biology"))

  # Search medical collection
  medical_results <- search_rf100(collection = "medical")
  expect_s3_class(medical_results, "data.frame")
  expect_gte(nrow(medical_results), 8)
  expect_true(all(medical_results$collection == "medical"))

  # Invalid collection should error
  expect_error(
    search_rf100(collection = "invalid_collection"),
    "Invalid collection"
  )
})

test_that("search_rf100 works with combined keyword and collection", {
  skip_on_cran()

  # Search for "cell" in biology collection
  results <- search_rf100("cell", collection = "biology")
  expect_s3_class(results, "data.frame")
  expect_true(all(results$collection == "biology"))

  # All results should contain "cell"
  has_cell <- sapply(seq_len(nrow(results)), function(i) {
    grepl("cell", results$dataset[i], ignore.case = TRUE) ||
    grepl("cell", results$description[i], ignore.case = TRUE)
  })
  expect_true(all(has_cell))
})

test_that("search_rf100 with no arguments returns all datasets", {
  skip_on_cran()

  all_results <- search_rf100()
  expect_s3_class(all_results, "data.frame")
  expect_gte(nrow(all_results), 37)
})

test_that("list_rf100_datasets works correctly", {
  skip_on_cran()

  # List biology datasets
  biology_datasets <- list_rf100_datasets("biology")
  expect_type(biology_datasets, "character")
  expect_gte(length(biology_datasets), 10)
  expect_true("blood_cell" %in% biology_datasets)

  # List medical datasets
  medical_datasets <- list_rf100_datasets("medical")
  expect_type(medical_datasets, "character")
  expect_length(medical_datasets, 8)

  # Invalid collection should error
  expect_error(
    list_rf100_datasets("invalid_collection"),
    "Invalid collection"
  )
})

test_that("catalog has correct dataset counts per collection", {
  skip_on_cran()

  catalog <- get_rf100_catalog()

  # Check counts
  counts <- table(catalog$collection)
  expect_gte(as.numeric(counts["biology"]), 9)
  expect_gte(as.numeric(counts["medical"]), 8)
  expect_gte(as.numeric(counts["infrared"]), 4)
  expect_gte(as.numeric(counts["damage"]), 3)
  expect_gte(as.numeric(counts["underwater"]), 4)
  expect_gte(as.numeric(counts["document"]), 8)
})

test_that("catalog includes known datasets", {
  skip_on_cran()

  catalog <- get_rf100_catalog()

  # Check for some known datasets
  expect_true("blood_cell" %in% catalog$dataset)
  expect_true("stomata_cell" %in% catalog$dataset)
  expect_true("solar_panel" %in% catalog$dataset)
  expect_true("brain_tumor" %in% catalog$dataset)
  expect_true("coral" %in% catalog$dataset)
})

test_that("CSV catalog file exists and is readable", {
  skip_on_cran()

  csv_path <- system.file("extdata", "rf100_catalog.csv", package = "torchvision")

  # File should exist
  expect_true(file.exists(csv_path))

  # Should be readable as CSV
  csv_data <- read.csv(csv_path, stringsAsFactors = FALSE)
  expect_s3_class(csv_data, "data.frame")
  expect_gte(nrow(csv_data), 37)
})

