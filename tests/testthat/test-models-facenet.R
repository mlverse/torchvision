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

  modelpnet <- model_facenet_pnet(pretrained = TRUE)
  modelpnet$eval()
  input <- torch_randn(1,3,160,160)
  out <- modelpnet(input)
  expect_tensor_shape(out$boxes, c(1,4,75,75))
  expect_tensor_shape(out$cls, c(1,2,75,75))

  rm(modelpnet)
  gc()
})

test_that("tests for non-pretrained model_facenet_pnet", {

  modelpnet <- model_facenet_pnet(pretrained = FALSE)
  modelpnet$eval()
  input <- torch_randn(1,3,160,160)
  out <- modelpnet(input)
  expect_tensor_shape(out$boxes, c(1,4,75,75))
  expect_tensor_shape(out$cls, c(1,2,75,75))

  rm(modelpnet)
  gc()
})

test_that("tests for pretrained model_facenet_rnet", {

  modelrnet <- model_facenet_rnet(pretrained = TRUE)
  modelrnet$eval()
  input <- torch_randn(1,3,24,24)
  out <- modelrnet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))

  rm(modelrnet)
  gc()
})

test_that("tests for non-pretrained model_facenet_rnet", {

  modelrnet <- model_facenet_rnet(pretrained = FALSE)
  modelrnet$eval()
  input <- torch_randn(1,3,24,24)
  out <- modelrnet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))

  rm(modelrnet)
  gc()
})

test_that("tests for pretrained model_facenet_onet", {

  modelonet <- model_facenet_onet(pretrained = TRUE)
  modelonet$eval()
  input <- torch_randn(1,3,48,48)
  out <- modelonet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))
  expect_tensor_shape(out$landmarks, c(1,10))

  rm(modelonet)
  gc()
})

test_that("tests for non-pretrained model_facenet_onet", {

  modelonet <- model_facenet_onet(pretrained = FALSE)
  modelonet$eval()
  input <- torch_randn(1,3,48,48)
  out <- modelonet(input)
  expect_tensor_shape(out$boxes, c(1,4))
  expect_tensor_shape(out$cls, c(1,2))
  expect_tensor_shape(out$landmarks, c(1,10))

  rm(modelonet)
  gc()
})

test_that("tests for pretrained model_facenet_inception_resnet_v1 with vgg2face weights", {

  model_vgg <- model_facenet_inception_resnet_v1(pretrained = 'vggface2')
  model_vgg$eval()
  input <- torch_randn(1,3,224,224)
  out <- model_vgg(input)
  expect_tensor_shape(out, c(1,512))

  rm(model_vgg)
  gc()
})

test_that("tests for pretrained model_facenet_inception_resnet_v1 with casia-webface weights", {

  model_casia <- model_facenet_inception_resnet_v1(pretrained = 'casia-webface')
  model_casia$eval()
  input <- torch_randn(1,3,320,260)
  out <- model_casia(input)
  expect_tensor_shape(out, c(1,512))

  rm(model_casia)
  gc()
})

test_that("tests for non-pretrained model_facenet_inception_resnet_v1", {

  model <- model_facenet_inception_resnet_v1(pretrained = NULL)
  model$eval()
  input <- torch_randn(1,3,224,224)
  out <- model(input)
  expect_tensor_shape(out, c(1,512))

  rm(model)
  gc()
})

test_that("model_facenet_inception_resnet_v1 with classify=TRUE requires num_classes without pretrained weights", {
  # matches facenet-pytorch: num_classes has no default when no pretrained weights are used
  expect_error(model_facenet_inception_resnet_v1(pretrained = NULL, classify = TRUE), "num_classes")
})

test_that("model_facenet_inception_resnet_v1 rejects unknown pretrained values", {
  expect_error(model_facenet_inception_resnet_v1(pretrained = "imagenet"), "pretrained")
})

test_that("tests for model_facenet_inception_resnet_v1 with classify=TRUE and custom num_classes", {
  model <- model_facenet_inception_resnet_v1(pretrained = NULL, classify = TRUE, num_classes = 100)
  model$eval()
  input <- torch_randn(1,3,224,224)
  out <- model(input)
  expect_tensor_shape(out, c(1,100))  # Custom num_classes is 100

  rm(model)
  gc()
})

test_that("tests for model_facenet_inception_resnet_v1 with batch size", {
  model <- model_facenet_inception_resnet_v1(pretrained = NULL)
  model$eval()
  input <- torch_randn(4,3,224,224)  # Batch size of 4
  out <- model(input)
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

test_that("P/R/ONet softmax is taken over the class dimension, not the batch", {
  torch_manual_seed(1)
  pnet <- model_facenet_pnet(pretrained = FALSE)
  rnet <- model_facenet_rnet(pretrained = FALSE)
  onet <- model_facenet_onet(pretrained = FALSE)
  with_no_grad({
    p <- pnet(torch_randn(2, 3, 24, 24))$cls
    r <- rnet(torch_randn(3, 3, 24, 24))$cls
    o <- onet(torch_randn(3, 3, 48, 48))$cls
    o1 <- onet(torch_randn(1, 3, 48, 48))$cls
  })
  expect_equal(as_array(p$sum(dim = 2)), array(1, c(2, 7, 7)), tolerance = 1e-5)
  expect_equal(as.numeric(as_array(r$sum(dim = 2))), rep(1, 3), tolerance = 1e-5)
  expect_equal(as.numeric(as_array(o$sum(dim = 2))), rep(1, 3), tolerance = 1e-5)
  # a single image must not always get probability 1
  expect_false(all(as_array(o1) == 1))
})

test_that("model_facenet_inception_resnet_v1 has the repeat_1 stage of facenet-pytorch", {
  model <- model_facenet_inception_resnet_v1(pretrained = NULL)
  keys <- names(model$state_dict())
  expect_length(keys, 714)
  expect_equal(sum(grepl("^repeat_1\\.", keys)), 190)
  expect_equal(sum(sapply(model$parameters, function(p) p$numel())), 23482624)
})
