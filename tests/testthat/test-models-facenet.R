context("models-facenet")

test_that("tests for pretrained model_mtcnn", {
  model <- model_mtcnn(pretrained = TRUE)
  input <- torch_randn(1, 3, 192, 192)
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

test_that("tests for pretrained model_facenet_inception_resnet_v1 with vgg2face weights", {

  model_vgg = model_facenet_inception_resnet_v1(pretrained = 'vggface2')
  model_vgg$eval()
  input = torch_randn(1,3,224,224)
  out = model_vgg(input)
  expect_tensor_shape(out, c(1,512))

  rm(model_vgg)
  gc()
})

test_that("tests for pretrained model_facenet_inception_resnet_v1 with casia-webface weights", {

  model_casia = model_facenet_inception_resnet_v1(pretrained = 'casia-webface')
  model_casia$eval()
  input = torch_randn(1,3,320,260)
  out = model_casia(input)
  expect_tensor_shape(out, c(1,512))

  rm(model_casia)
  gc()
})

test_that("tests for non-pretrained model_facenet_inception_resnet_v1", {

  model = model_facenet_inception_resnet_v1(pretrained = NULL)
  model$eval()
  input = torch_randn(1,3,224,224)
  out = model(input)
  expect_tensor_shape(out, c(1,512))

  rm(model)
  gc()
})

test_that("tests for model_facenet_inception_resnet_v1 with classify=TRUE and default num_classes", {
  model = model_facenet_inception_resnet_v1(pretrained = NULL, classify = TRUE)
  model$eval()
  input = torch_randn(1,3,224,224)
  out = model(input)
  expect_tensor_shape(out, c(1,10))  # Default num_classes is 10

  rm(model)
  gc()
})

test_that("tests for model_facenet_inception_resnet_v1 with classify=TRUE and custom num_classes", {
  model = model_facenet_inception_resnet_v1(pretrained = NULL, classify = TRUE, num_classes = 100)
  model$eval()
  input = torch_randn(1,3,224,224)
  out = model(input)
  expect_tensor_shape(out, c(1,100))  # Custom num_classes is 100

  rm(model)
  gc()
})

test_that("tests for model_facenet_inception_resnet_v1 with batch size", {
  model = model_facenet_inception_resnet_v1(pretrained = NULL)
  model$eval()
  input = torch_randn(4,3,224,224)  # Batch size of 4
  out = model(input)
  expect_tensor_shape(out, c(4,512))

  rm(model)
  gc()
})

test_that("error test for model_mtcnn with error input size", {
  model <- model_mtcnn(pretrained = FALSE)
  input <- torch_randn(1, 3, 10, 10)
  model$eval()

  expect_error(
    model(input),
    regexp = "size|dimension|shape"
  )

  rm(model)
  gc()
})
