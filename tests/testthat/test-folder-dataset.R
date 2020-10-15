test_that("image_folder dataset", {

  ds <- image_folder_dataset(
    root = "assets/class",
    transform = . %>% transform_to_tensor %>%
      transform_resize(c(32,32))
  )
  expect_length(ds[1], 2)

  dl <- torch::dataloader(ds, batch_size = 2, drop_last = TRUE)
  for(batch in torch::enumerate(dl)) {
    expect_tensor_shape(batch[[1]], c(2, 3, 32, 32))
    expect_tensor_shape(batch[[2]], 2)
    expect_tensor_shape(batch$x, c(2, 3, 32, 32))
    expect_tensor_shape(batch$y, 2)
  }

  expect_length(ds, 12)

})
