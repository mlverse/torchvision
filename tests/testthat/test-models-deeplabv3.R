test_that("deeplabv3_resnet50 works with default aux_loss=NULL", {
  model <- model_deeplabv3_resnet50(num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))

  withr::with_options(list(timeout = 360), {
    model <- model_deeplabv3_resnet50(pretrained = TRUE)
  })
  out <- model(input)
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
})

test_that("deeplabv3_resnet101 works with default aux_loss=NULL", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  model <- model_deeplabv3_resnet101(num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_true(is.list(out))
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))

  withr::with_options(list(timeout = 360), {
    model <- model_deeplabv3_resnet101(pretrained = TRUE)
  })
  out <- model(input)
  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
})

test_that("deeplabv3_resnet50 works with aux_loss = TRUE", {
  model <- model_deeplabv3_resnet50(aux_loss = TRUE, num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
  expect_tensor_shape(out$aux, c(1, 21, 32, 32))
})

test_that("deeplabv3_resnet50 works with aux_loss = FALSE", {
  model <- model_deeplabv3_resnet50(aux_loss = FALSE, num_classes = 21)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_named(out, "out")
  expect_tensor_shape(out$out, c(1, 21, 32, 32))
})

test_that("custom num_classes works with aux_loss = TRUE", {
  model <- model_deeplabv3_resnet50(num_classes = 3, aux_loss = TRUE)
  input <- torch::torch_randn(1, 3, 32, 32)
  out <- model(input)

  expect_named(out, c("out", "aux"))
  expect_tensor_shape(out$out, c(1, 3, 32, 32))
  expect_tensor_shape(out$aux, c(1, 3, 32, 32))
})

test_that("pretrained requires num_classes to be \'21\'", {
  expect_error(
    model_deeplabv3_resnet50(pretrained = TRUE, num_classes = 3),
    "Pretrained weights on COCO require"
  )
})

test_that("model_deeplabv3_resnet50 detects cat in Wikipedia image", {
  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  voc_classes <- c(
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
  )

  img_url <- "https://upload.wikimedia.org/wikipedia/commons/3/36/United_Airlines_Boeing_777-200_Meulemans.jpg"
  img <- magick::image_read(img_url)

  norm_mean <- c(0.485, 0.456, 0.406)
  norm_std <- c(0.229, 0.224, 0.225)

  input <- transform_to_tensor(img)
  input <- transform_resize(input, c(520, 520))
  input <- transform_normalize(input, mean = norm_mean, std = norm_std)
  input <- input$unsqueeze(1)

  model <- model_deeplabv3_resnet50(pretrained = TRUE)
  model$eval()

  output <- model(input)
  mask <- output$out$argmax(dim = 2)  # shape (1, H, W)

  label_array <- mask %>% torch::as_array()  # convert to R array
  label_table <- table(factor(label_array, levels = 0:20, labels = voc_classes))

  expect_gt(label_table[["aeroplane"]], 0)
  expect_gt(label_table[["aeroplane"]], label_table[["dog"]])
  expect_gt(label_table[["aeroplane"]], label_table[["person"]])
})

