# Loading Images ---------------------------------------------------
library(torchvision)
library(torch)

url1 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
url2 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"

dog1 <- base_loader(url1) %>% transform_to_tensor()
dog2 <- base_loader(url2) %>% transform_to_tensor()


# Visualizing a grid of images -------------------------------------


dogs <- torch_stack(list(dog1, dog2))
grid <- vision_make_grid(dogs, scale = TRUE, num_rows = 2)
tensor_image_browse(grid)


# Preprocessing the data -------------------------------------------


norm_mean <- c(0.485, 0.456, 0.406)
norm_std  <- c(0.229, 0.224, 0.225)

dog1_prep <- dog1 %>%
  transform_resize(c(800, 800)) %>%
  transform_normalize(mean = norm_mean, std = norm_std) %>%
  torch_tensor(dtype = torch_float32())
dog2_prep <- dog2 %>%
  transform_resize(c(800, 800)) %>%
  transform_normalize(mean = norm_mean, std = norm_std) %>%
  torch_tensor(dtype = torch_float32())

# make batch (2,3,800,800)
dog_batch <- torch_stack(list(dog1_prep, dog2_prep))


# Loading Model ----------------------------------------------------


model <- model_fasterrcnn_resnet50_fpn(
  pretrained = TRUE,
  score_thresh = 0.5,
  nms_thresh = 0.8,
  detections_per_img = 2
)
model$eval()

# run model
output <- model(dog_batch)


# Processing the Output --------------------------------------------

pred1 <- output$detections[[1]]
pred2 <- output$detections[[2]]

pred1$boxes
pred1$labels
pred1$scores


# Visualizing the Output -------------------------------------------


boxed1 <- draw_bounding_boxes(
  dog1 %>% transform_resize(c(800, 800)),
  boxes = pred1$boxes,
  labels = coco_label(as.integer(pred1$labels))
)

boxed2 <- draw_bounding_boxes(
  dog2 %>% transform_resize(c(800, 800)),
  boxes = pred2$boxes,
  labels = coco_label(as.integer(pred2$labels))
)

tensor_image_browse(boxed1)
tensor_image_browse(boxed2)