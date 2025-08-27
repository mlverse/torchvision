# Loading Images ---------------------------------------------------
library(torchvision)
library(torch)

read_to_tensor <- function(url) {
  arr <- magick_loader(url)
  torch_tensor(arr, dtype = torch_float())$permute(c(3,1,2))
}

url1 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
url2 <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"

dog1 <- read_to_tensor(url1)
dog2 <- read_to_tensor(url2)


# Visualizing a grid of images -------------------------------------


dogs <- torch_stack(list(dog1, dog2))
grid <- vision_make_grid(dogs, scale = TRUE, num_rows = 2)
grid_arr <- as.array(grid$permute(c(2,3,1)))
plot(c(0, dim(grid_arr)[2]), c(0, dim(grid_arr)[1]), type = "n", ann = FALSE, axes = FALSE, asp = 1)
rasterImage(grid_arr, 0, 0, w, h)


# Preprocessing the data -------------------------------------


norm_mean <- c(0.485, 0.456, 0.406)
norm_std  <- c(0.229, 0.224, 0.225)

preprocess <- function(img) {
  resized <- nnf_interpolate(
    img$unsqueeze(1), size = c(520, 520), mode = "bilinear", align_corners = FALSE
  )$squeeze(1)
  normed <- (resized - torch_tensor(norm_mean)$unsqueeze(2)$unsqueeze(3)) /
              torch_tensor(norm_std)$unsqueeze(2)$unsqueeze(3)
  list(resized = resized, normed = normed)
}

dog1_prep <- preprocess(dog1)
dog2_prep <- preprocess(dog2)

# make batch (2,3,520,520)
input <- torch_stack(list(dog1_prep$normed, dog2_prep$normed))


# Loading Model -------------------------------------


model <- model_fcn_resnet50(pretrained = TRUE)
model$eval()

# run model
output <- model(input)


# Processing the Output ------------------------------


# argmax over classes -> (1,H,W)
mask_id <- output$out$argmax(dim = 2)

make_masks <- function(mask_id_img, num_classes = 21) {
  torch_stack(lapply(0:(num_classes-1), function(cls) mask_id_img$eq(cls)), dim = 1)
}
mask_bool1 <- make_masks(mask_id[1,..])
mask_bool2 <- make_masks(mask_id[2,..])


# Visualizing the Output ------------------------------


segmented1 <- draw_segmentation_masks(
  (dog1_prep$resized * 255)$to(torch_uint8()),
  masks = mask_bool1,
  alpha = 0.6
)

segmented2 <- draw_segmentation_masks(
  (dog2_prep$resized * 255)$to(torch_uint8()),
  masks = mask_bool2,
  alpha = 0.6
)

tensor_image_browse(segmented1)
tensor_image_browse(segmented2)
