# Overview ---------------------------------------------------------
#
# model_fasterrcnn_resnet50_fpn() implements the Faster R-CNN
# object detection architecture with a ResNet-50 backbone and
# Feature Pyramid Network (FPN), combining region proposal,
# classification, and bounding box regression in a single pipeline.


# Loading Images ---------------------------------------------------
library(torchvision)
library(torch)

url1 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
url2 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"

dog1 <- magick_loader(url1) %>% transform_to_tensor()
dog2 <- magick_loader(url2) %>% transform_to_tensor()


# Visualizing a grid of images -------------------------------------


dogs <- torch_stack(list(dog1, dog2))
grid <- vision_make_grid(dogs, scale = TRUE, num_rows = 2)
tensor_image_browse(grid)


# Preprocessing the data -------------------------------------


norm_mean <- c(0.485, 0.456, 0.406)
norm_std  <- c(0.229, 0.224, 0.225)

dog1_prep <- dog1 %>%
  transform_resize(c(520,520)) %>%
  transform_normalize(mean = norm_mean, std = norm_std)
dog2_prep <- dog2 %>%
  transform_resize(c(520,520)) %>%
  transform_normalize(mean = norm_mean, std = norm_std)

# make batch (2,3,520,520)
dog_batch <- torch_stack(list(dog1_prep, dog2_prep))


# Loading Model -------------------------------------

# pretrained = TRUE loads weights trained on the COCO dataset
# (Common Objects in Context benchmark for object detection)
# score_thresh filters detections with confidence below the threshold
model <- model_fasterrcnn_resnet50_fpn(pretrained = TRUE, score_thresh = 0.5)
model$eval()

# run model
output <- model(dog_batch)


# Understanding the Output Structure -------------------------------

# In eval() mode, output contains:
names(output)  # typically "detections" in eval() mode

# output$detections: list with one element per image in batch

detections <- output$detections
length(detections)  # equals the batch size (2 images)

# Each detection contains boxes, labels, and scores
det1 <- detections[[1]]
names(det1)  # "boxes", "labels", "scores"

# boxes: (N, 4) tensor with [x1, y1, x2, y2] pixel coordinates
# representing top-left and bottom-right corners
det1$boxes$shape

# labels: (N) tensor with COCO class IDs
det1$labels$shape

# scores: (N) tensor with confidence values [0, 1]
det1$scores$shape

# Note: In train() mode, output contains losses instead:
# model$train()
# output <- model(images, targets)
# output$losses  # list(loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg)


# Visualizing the Output ------------------------------


num_boxes1 <- as.integer(det1$boxes$size()[1])
if (num_boxes1 > 0) {
  keep <- seq_len(min(5, num_boxes1))
  boxes1 <- det1$boxes[keep, ]
  # coco_label() converts COCO class IDs to readable class names
  labels1 <- coco_label(as.integer(det1$labels[keep]))
  scores1 <- as.numeric(det1$scores[keep])
  labels_with_scores1 <- paste0(labels1, " (", sprintf("%.2f", scores1), ")")
  
  detected1 <- draw_bounding_boxes(
    dog1 %>% transform_resize(c(520,520)),
    boxes = boxes1,
    labels = labels_with_scores1,
    width = 3
  )
  
  tensor_image_browse(detected1)
}

det2 <- detections[[2]]
num_boxes2 <- as.integer(det2$boxes$size()[1])
if (num_boxes2 > 0) {
  keep <- seq_len(min(5, num_boxes2))
  boxes2 <- det2$boxes[keep, ]
  labels2 <- coco_label(as.integer(det2$labels[keep]))
  scores2 <- as.numeric(det2$scores[keep])
  labels_with_scores2 <- paste0(labels2, " (", sprintf("%.2f", scores2), ")")
  
  detected2 <- draw_bounding_boxes(
    dog2 %>% transform_resize(c(520,520)),
    boxes = boxes2,
    labels = labels_with_scores2,
    width = 3
  )
  
  tensor_image_browse(detected2)
}


# Summary ----------------------------------------------------------
#
# This article demonstrated:
# - Loading and preprocessing images for object detection
# - Running Faster R-CNN inference with pretrained weights
# - Understanding detection output structure (boxes, labels, scores)
# - Visualizing bounding boxes with confidence scores and class labels
#
# Faster R-CNN provides high-quality instance-level object detection
# suitable for both inference and training workflows.
