test_that("image_folder dataset", {

  ds <- image_folder_dataset(
    root = "assets/class",
    transform = . %>% transform_to_tensor %>%
      transform_resize(c(32,32))
  )
  expect_length(ds[1], 2)

  dl <- torch::dataloader(ds, batch_size = 2, drop_last = TRUE)
  coro::loop(for(batch in dl) {
    expect_tensor_shape(batch[[1]], c(2, 3, 32, 32))
    expect_tensor_shape(batch[[2]], 2)
    expect_tensor_shape(batch$x, c(2, 3, 32, 32))
    expect_tensor_shape(batch$y, 2)
  })

  expect_length(ds, 15)

})

test_that("default_loader works as expected", {
  # rvb jpeg
  cat1 <- base_loader("assets/class/cat/cat.1.jpg")
  expect_equal(dim(cat1)[3], 3L)
  # rvb png
  horse1 <- base_loader("assets/class/horse/horse-1.png")
  expect_equal(dim(horse1)[3], 3L)
  # rvb tiff
  horse2 <- base_loader("assets/class/horse/horse-2.tif")
  expect_equal(dim(horse2)[3], 3L)
  # grayscale jpeg
  dog5 <- base_loader("assets/class/dog/dog.5.jpg")
  expect_equal(dim(dog5)[3], 3L)
})
