test_that("inception_v3 pretrained", {
  model <- model_inception_v3(pretrained = TRUE)
  model$eval()
  x <- model(torch_ones(2, 3, 299, 299))
  # the value has been copied from running the same model on pytorch.
  expect_equal(as.numeric(x[1,1]), 0.18005196750164032, tol = 5e-6)
  expect_tensor_shape(x, c(2, 1000))
})

test_that("produces correctly classification", {
  model <- model_inception_v3(pretrained = TRUE)
  model$eval()

  ds <- torchvision::image_folder_dataset(
    testthat::test_path("assets/class"),
    transform = function(x) {
      x <- transform_to_tensor(x)
      x <- transform_resize(x, size = c(299, 299))
      x <- torchvision::transform_normalize(
        x,
        mean = c(0.485, 0.456, 0.406),
        std = c(0.229, 0.224, 0.225)
      )
      x
    })
  dl <- dataloader(ds, batch_size = 5)
  batch <- dataloader_next(dataloader_make_iter(dl))

  pred <- model(batch[[1]])
  pred <- pred$argmax(dim=-1)
  expect_equal(
    imagenet_classes()[as.integer(pred)],
    c("Pembroke, Pembroke Welsh corgi", "tabby, tabby cat", "tabby, tabby cat",
      "Egyptian cat", "Egyptian cat")
  )

})
