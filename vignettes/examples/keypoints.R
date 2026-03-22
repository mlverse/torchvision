# Load image using base_loader ---------------------------------------------------
url <- "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg"
image <- base_loader(url) %>% transform_to_tensor()

# 1 triangle, 3 keypoints each (x, y) --------------------------------------------
keypoints <- torch_tensor(rbind(
  c(263, 225),
  c(238, 220),
  c(242, 247)
), dtype = torch_float())$reshape(c(1L, 3L, 2L))

# connectivity to form a triangle ------------------------------------------------
connectivity <- list(
  c(1L, 2L),
  c(1L, 3L),
  c(2L, 3L)
)

# Visualize keypoints ------------------------------------------------------------
triangle_img <- draw_keypoints(
  image,
  keypoints,
  connectivity = connectivity,
  colors       = c("red", "blue"),
  radius       = 4L,
  width        = 2L
)

tensor_image_browse(triangle_img)