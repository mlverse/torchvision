context("models-facenet")

test_that("tests for pretrained model_mtcnn", {
  model <- model_mtcnn(pretrained = TRUE)
  input <- torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out$boxes, c(1, 4))
  expect_tensor_shape(out$landmarks, c(1, 10))
  expect_tensor_shape(out$cls, c(1, 2))

  rm(model)
  gc()
})

test_that("tests for non-pretrained model_mtcnn", {
  model <- model_mtcnn(pretrained = FALSE)
  input <- torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_tensor_shape(out$boxes, c(1, 4))
  expect_tensor_shape(out$landmarks, c(1, 10))
  expect_tensor_shape(out$cls, c(1, 2))

  rm(model)
  gc()
})