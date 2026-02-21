library(torchvision)
library(torch)

image <- torch_randint(180, 240, c(3, 400, 400))$to(torch_uint8())

# 13-point human skeleton
kpts <- torch_tensor(array(c(
  200, 80,   # head
  200, 120,  # neck
  160, 160,  # l shoulder
  240, 160,  # r shoulder
  140, 220,  # l elbow
  260, 220,  # r elbow
  130, 270,  # l wrist
  270, 270,  # r wrist
  200, 240,  # torso
  180, 300,  # l hip
  220, 300,  # r hip
  170, 360,  # l knee
  230, 360   # r knee
), dim = c(1, 13, 2)))

skeleton <- matrix(c(
  1,2, 2,3, 2,4, 3,5, 4,6, 5,7, 6,8, 2,9, 9,10, 9,11, 10,12, 11,13
), ncol = 2, byrow = TRUE)

result <- draw_keypoints(image, kpts, connectivity = skeleton, 
                         colors = "cyan", radius = 4, width = 2)

cat("Result shape:", paste(result$shape, collapse = "x"), "\n")
# tensor_image_browse(result)
