test_that("model_fcn_resnet50 returns correct output shape", {
  model <- model_fcn_resnet50(pretrained = FALSE, num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet50 works with custom num_classes", {
  model <- model_fcn_resnet50(pretrained = FALSE, num_classes = 3)
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(2, 3, 224, 224))
})

test_that("model_fcn_resnet50 with aux classifier returns aux output", {
  model <- model_fcn_resnet50(pretrained = FALSE, num_classes = 21, aux_loss = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
  expect_tensor_shape(output$aux, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet50 loads pretrained weights", {
  model <- model_fcn_resnet50(pretrained = TRUE, num_classes = 12)
  expect_true(inherits(model, "fcn"))
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(2, 12, 224, 224))
  expect_tensor_shape(output$aux, c(2, 12, 224, 224))
})

test_that("model_fcn_resnet50 can segment a cat", {
  voc_idx <- setNames(seq_along(torchvision:::voc_segmentation_classes), torchvision:::voc_segmentation_classes)

  model <- model_fcn_resnet50(pretrained = TRUE)
  input <- torch::torch_tensor(jpeg::readJPEG("assets/class/cat/cat.1.jpg"))$permute(c(3,1,2))
  output <- model(input$unsqueeze(1))
  mask_id <- output$out$argmax(dim = 2)
  mask_table <- factor(mask_id |> torch::as_array(), levels = voc_idx, labels = names(voc_idx)) |> table()

  expect_gt(mask_table[["cat"]], 0)
  expect_gt(mask_table[["cat"]], mask_table[["dog"]])
  expect_gt(mask_table[["cat"]], mask_table[["person"]])

  expect_gt(mask_table[["background"]], 0)

})

test_that("model_fcn_resnet50 can segment a cat", {
  voc_idx <- setNames(seq_along(torchvision:::voc_segmentation_classes), torchvision:::voc_segmentation_classes)

  model <- model_fcn_resnet50(pretrained = TRUE)
  img <- torch::torch_tensor(jpeg::readJPEG("assets/class/cat/cat.1.jpg"))$permute(c(3,1,2))
  norm_mean <- c(0.485, 0.456, 0.406) #ImageNet normalization constants
  norm_std  <- c(0.229, 0.224, 0.225)
  input <- img %>% transform_resize(c(520, 520)) %>% transform_normalize(input, norm_mean, norm_std)

  output <- model(input$unsqueeze(1))
  mask_id <- output$out$argmax(dim = 2)
  mask_table <- factor(mask_id %>% torch::as_array(), levels = voc_idx, labels = names(voc_idx)) %>% table()

  expect_gt(mask_table[["cat"]], 0)
  expect_gt(mask_table[["cat"]], mask_table[["dog"]])
  expect_gt(mask_table[["cat"]], mask_table[["person"]])

  expect_gt(mask_table[["background"]], 0)

})


test_that("model_fcn_resnet101 returns correct output shape", {
  model <- model_fcn_resnet101(pretrained = FALSE, num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet101 works with custom num_classes", {
  model <- model_fcn_resnet101(pretrained = FALSE, num_classes = 3)
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(2, 3, 224, 224))
})

test_that("model_fcn_resnet101 with aux classifier returns aux output", {
  model <- model_fcn_resnet101(pretrained = FALSE, num_classes = 21, aux_loss = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
  expect_tensor_shape(output$aux, c(1, 21, 224, 224))
})

test_that("model_fcn_resnet101 loads pretrained weights", {
  model <- model_fcn_resnet101(pretrained = TRUE)
  expect_true(inherits(model, "fcn"))
})

t
