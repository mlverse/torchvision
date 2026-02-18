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


model <- model_fcn_resnet50(pretrained = TRUE)
model$eval()

# run model
output <- model(dog_batch)


# Processing the Output ------------------------------

mask <- output$out
mask$shape
mask$dtype

# Visualizing the Output ------------------------------


segmented1 <- draw_segmentation_masks(
  dog1 %>% transform_resize(c(520,520)),
  masks = mask[1,, ],
  alpha = 0.5
)

segmented2 <- draw_segmentation_masks(
  dog2 %>% transform_resize(c(520,520)),
  masks = mask[2,, ],
  alpha = 0.5
)

tensor_image_browse(segmented1)
tensor_image_browse(segmented2)
