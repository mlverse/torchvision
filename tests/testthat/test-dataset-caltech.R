test_that("caltech101", {
  skip_if_not_installed("R.matlab")
  t <- tempfile()
<<<<<<< HEAD

=======
  
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
  expect_error(
    caltech101_dataset(t),
    "Dataset not found"
  )
<<<<<<< HEAD

  # Test download
  ds <- caltech101_dataset(t, download = TRUE)
  expect_equal(length(ds), 8677)

=======
  
  # Test download
  ds <- caltech101_dataset(t, download = TRUE)
  expect_equal(length(ds), 8677)
  
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
  # Check sample item
  el <- ds[1]
  expect_tensor(el$x)
  expect_equal(dim(el$x)[1], 3)  # CHW format
  expect_equal(length(el$y), 1)  # Default target_type="category"
<<<<<<< HEAD

=======
  
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
  # Test annotations
  ds_anno <- caltech101_dataset(t, target_type = "annotation")
  el_anno <- ds_anno[1]
  expect_true(is.matrix(el_anno$y[[1]]))
<<<<<<< HEAD

=======
  
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
  # Test combined targets
  ds_both <- caltech101_dataset(t, target_type = c("category", "annotation"))
  el_both <- ds_both[1]
  expect_length(el_both$y, 2)
  expect_type(el_both$y[[1]], "integer")
  expect_true(is.matrix(el_both$y[[2]]))
<<<<<<< HEAD
})
=======
})
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
