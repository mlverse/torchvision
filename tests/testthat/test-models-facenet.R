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

test_that("tests for pretrained model_facenet_pnet", {

  modelpnet = model_facenet_pnet(pretrained = TRUE)
  modelpnet$eval()
  input = torch_randn(1,3,160,160)
  out = modelpnet(input)
  expect_tensor_shape(out$boxes, c(1,4,75,75))
  expect_tensor_shape(out$cls, c(1,2,75,75))

  rm(modelpnet)
  gc()
})

test_that("tests for non-pretrained model_facenet_pnet", {

  modelpnet = model_facenet_pnet(pretrained = FALSE)
  modelpnet$eval()
  input = torch_randn(1,3,160,160)
  out = modelpnet(input)
  expect_tensor_shape(out$boxes, c(1,4,75,75))
  expect_tensor_shape(out$cls, c(1,2,75,75))

  rm(modelpnet)
  gc()
})

test_that("tests for pretrained model_facenet_rnet", {

  modelrnet = model_facenet_rnet(pretrained = TRUE)
  modelrnet$eval()
  input = torch_randn(1,3,24,24)
  out = modelrnet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))

  rm(modelrnet)
  gc()
})

test_that("tests for non-pretrained model_facenet_rnet", {

  modelrnet = model_facenet_rnet(pretrained = FALSE)
  modelrnet$eval()
  input = torch_randn(1,3,24,24)
  out = modelrnet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))

  rm(modelrnet)
  gc()
})

test_that("tests for pretrained model_facenet_onet", {

  modelonet = model_facenet_onet(pretrained = TRUE)
  modelonet$eval()
  input = torch_randn(1,3,48,48)
  out = modelonet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))
  expect_tensor_shape(out$landmarks, c(1,10))

  rm(modelonet)
  gc()
})

test_that("tests for non-pretrained model_facenet_onet", {

  modelonet = model_facenet_onet(pretrained = FALSE)
  modelonet$eval()
  input = torch_randn(1,3,48,48)
  out = modelonet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))
  expect_tensor_shape(out$landmarks, c(1,10))

  rm(modelonet)
  gc()
})