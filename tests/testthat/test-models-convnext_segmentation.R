# Tests for ConvNeXt Segmentation Models
# These tests cover FCN and UPerNet heads with ConvNeXt backbone

# FCN Head Tests

test_that("model_convnext_tiny_fcn returns correct output shape", {
  model <- model_convnext_tiny_fcn(num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_true(is.list(output))
  expect_named(output, "out")
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_small_fcn returns correct output shape", {
  model <- model_convnext_small_fcn(num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_true(is.list(output))
  expect_named(output, "out")
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_base_fcn returns correct output shape", {
  model <- model_convnext_base_fcn(num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_true(is.list(output))
  expect_named(output, "out")
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_tiny_fcn works with custom num_classes", {
  model <- model_convnext_tiny_fcn(num_classes = 3)
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(2, 3, 224, 224))
})

test_that("model_convnext_tiny_fcn with aux_loss returns aux output", {
  model <- model_convnext_tiny_fcn(num_classes = 21, aux_loss = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
  expect_tensor_shape(output$aux, c(1, 21, 224, 224))
})

test_that("model_convnext_tiny_fcn works with different input sizes", {
  model <- model_convnext_tiny_fcn(num_classes = 21)
  model$eval()

  # Test with 256x256
  input <- torch::torch_randn(1, 3, 256, 256)
  output <- model(input)
  expect_tensor_shape(output$out, c(1, 21, 256, 256))

  # Test with 512x512
  input <- torch::torch_randn(1, 3, 512, 512)
  output <- model(input)
  expect_tensor_shape(output$out, c(1, 21, 512, 512))
})

test_that("model_convnext_tiny_fcn validates num_classes", {
  expect_error(
    model_convnext_tiny_fcn(num_classes = 0),
    "num_classes"
  )
  expect_error(
    model_convnext_tiny_fcn(num_classes = -1),
    "num_classes"
  )
})

# UPerNet Head Tests

test_that("model_convnext_tiny_upernet returns correct output shape", {
  model <- model_convnext_tiny_upernet(num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_true(is.list(output))
  expect_named(output, "out")
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_small_upernet returns correct output shape", {
  model <- model_convnext_small_upernet(num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_true(is.list(output))
  expect_named(output, "out")
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_base_upernet returns correct output shape", {
  model <- model_convnext_base_upernet(num_classes = 21)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_true(is.list(output))
  expect_named(output, "out")
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_tiny_upernet works with custom num_classes", {
  model <- model_convnext_tiny_upernet(num_classes = 3)
  model$eval()

  input <- torch::torch_randn(2, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(2, 3, 224, 224))
})

test_that("model_convnext_tiny_upernet with aux_loss returns aux output", {
  model <- model_convnext_tiny_upernet(num_classes = 21, aux_loss = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_named(output, c("out", "aux"))
  expect_tensor_shape(output$out, c(1, 21, 224, 224))
  expect_tensor_shape(output$aux, c(1, 21, 224, 224))
})

test_that("model_convnext_tiny_upernet works with different input sizes", {
  model <- model_convnext_tiny_upernet(num_classes = 21)
  model$eval()

  # Test with 256x256
  input <- torch::torch_randn(1, 3, 256, 256)
  output <- model(input)
  expect_tensor_shape(output$out, c(1, 21, 256, 256))

  # Test with 512x512
  input <- torch::torch_randn(1, 3, 512, 512)
  output <- model(input)
  expect_tensor_shape(output$out, c(1, 21, 512, 512))
})

test_that("model_convnext_tiny_upernet validates num_classes", {
  expect_error(
    model_convnext_tiny_upernet(num_classes = 0),
    "num_classes"
  )
  expect_error(
    model_convnext_tiny_upernet(num_classes = -1),
    "num_classes"
  )
  expect_error(
    model_convnext_tiny_upernet(num_classes = 21, pretrained = TRUE),
    "num_classes"
  )
})

# Pretrained Backbone Tests

test_that("model_convnext_tiny_fcn works with pretrained backbone", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_convnext_tiny_fcn(num_classes = 21, pretrained_backbone = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_tiny_upernet works with pretrained backbone", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_convnext_tiny_upernet(num_classes = 21, pretrained_backbone = TRUE)
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("model_convnext_tiny_upernet works with pretrained model", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  expect_no_error(
    model <- model_convnext_tiny_upernet(num_classes = 150, pretrained = TRUE)
  )
  model$eval()

  input <- torch::torch_randn(1, 3, 512, 512)
  output <- model(input)

  expect_tensor_shape(output$out, c(1, 150, 512, 512))
})

test_that("model_convnext_small_upernet works with pretrained model", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  expect_no_error(
    model <- model_convnext_small_upernet(num_classes = 150, pretrained = TRUE)
  )
  model$eval()

  input <- torch::torch_randn(1, 3, 512, 512)
  output <- model(input)

  expect_tensor_shape(output$out, c(1, 150, 512, 512))
})

test_that("model_convnext_base_upernet works with pretrained model", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = "0") != "1",
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  expect_no_error(
    model <- model_convnext_base_upernet(num_classes = 150, pretrained = TRUE)
  )
  model$eval()

  input <- torch::torch_randn(1, 3, 512, 512)
  output <- model(input)

  expect_tensor_shape(output$out, c(1, 150, 512, 512))
})

# Pool Scales Tests (UPerNet specific)

test_that("model_convnext_tiny_upernet works with custom pool_scales", {
  model <- model_convnext_tiny_upernet(num_classes = 21, pool_scales = c(1, 2, 3))
  model$eval()

  input <- torch::torch_randn(1, 3, 224, 224)
  output <- model(input)

  expect_tensor_shape(output$out, c(1, 21, 224, 224))
})

test_that("UPerNet checkpoint keys (incl. backbone.norm0-3) all map onto the model", {
  model <- model_convnext_tiny_upernet(num_classes = 150, aux_loss = TRUE)
  ckpt_keys <- c(
    "backbone.downsample_layers.0.0.weight", "backbone.stages.0.0.depthwise_conv.weight",
    "backbone.stages.0.0.pointwise_conv1.weight", "backbone.stages.0.0.gamma",
    paste0("backbone.norm", 0:3, ".weight"), paste0("backbone.norm", 0:3, ".bias"),
    "decode_head.psp_modules.0.1.conv.weight", "decode_head.psp_modules.3.1.bn.running_var",
    "decode_head.bottleneck.conv.weight", "decode_head.lateral_convs.2.bn.weight",
    "decode_head.fpn_convs.0.conv.weight", "decode_head.fpn_bottleneck.bn.bias",
    "decode_head.conv_seg.weight", "auxiliary_head.convs.0.conv.weight",
    "auxiliary_head.convs.0.bn.running_mean", "auxiliary_head.conv_seg.bias"
  )
  fake <- setNames(as.list(seq_along(ckpt_keys)), ckpt_keys)
  renamed <- names(torchvision:::.rename_convnext_state_dict(fake))
  expect_true(all(renamed %in% names(model$state_dict())))
})
