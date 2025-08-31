test_that("tests for non-pretrained model_fasterrcnn_resnet50_fpn", {
  model <- model_fasterrcnn_resnet50_fpn()
  input <- torch::torch_randn(1, 3, 224, 224)
  model$eval()
  out <- model(input)
  expect_s3_class(out, "list")
  expect_names(out, c(""))

  model <- model_fasterrcnn_resnet50_fpn(num_classes = 10)
  input <- torch::torch_randn(1, 3, 224, 224)
  out <- model(input)
  expect_s3_class(out, "list")
})

test_that("tests for pretrained model_fasterrcnn_resnet50_fpn", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")

  model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE)
  input <- torch::torch_randn(1, 3, 448, 448)
  out <- model(input)
  expect_tensor_shape(out, c(1, 1000))
})

ds <- coco_detection_dataset(train = FALSE, year = "2017", download = TRUE)
sample <- ds[1]
image <- (sample$x * 255)$to(dtype = torch::torch_uint8())
# ResNet-50 FPN
model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE)
model$eval()
pred <- model(list(sample$x))$detections
num_boxes <- as.integer(pred$boxes$size()[1])
keep <- seq_len(min(5, num_boxes))
boxes <- pred$boxes[keep, ]$view(c(-1, 4))
labels <- ds$category_names[as.character(as.integer(pred$labels[keep]))]
if (num_boxes > 0) {
 boxed <- draw_bounding_boxes(image, boxes, labels = labels)
 tensor_image_browse(boxed)
}
# ResNet-50 FPN V2
model <- model_fasterrcnn_resnet50_fpn_v2(pretrained = TRUE)
model$eval()
pred <- model(list(sample$x))$detections
num_boxes <- as.integer(pred$boxes$size()[1])
keep <- seq_len(min(5, num_boxes))
boxes <- pred$boxes[keep, ]$view(c(-1, 4))
labels <- ds$category_names[as.character(as.integer(pred$labels[keep]))]
if (num_boxes > 0) {
 boxed <- draw_bounding_boxes(image, boxes, labels = labels)
 tensor_image_browse(boxed)
}
# MobileNet V3 Large FPN
model <- model_fasterrcnn_mobilenet_v3_large_fpn(pretrained = TRUE)
model$eval()
pred <- model(list(sample$x))$detections
num_boxes <- as.integer(pred$boxes$size()[1])
keep <- seq_len(min(5, num_boxes))
boxes <- pred$boxes[keep, ]$view(c(-1, 4))
labels <- ds$category_names[as.character(as.integer(pred$labels[keep]))]
if (num_boxes > 0) {
 boxed <- draw_bounding_boxes(image, boxes, labels = labels)
 tensor_image_browse(boxed)
}
# MobileNet V3 Large 320 FPN
model <- model_fasterrcnn_mobilenet_v3_large_320_fpn(pretrained = TRUE)
model$eval()
pred <- model(list(sample$x))$detections
num_boxes <- as.integer(pred$boxes$size()[1])
keep <- seq_len(min(5, num_boxes))
boxes <- pred$boxes[keep, ]$view(c(-1, 4))
labels <- ds$category_names[as.character(as.integer(pred$labels[keep]))]
if (num_boxes > 0) {
 boxed <- draw_bounding_boxes(image, boxes, labels = labels)
 tensor_image_browse(boxed)
}
