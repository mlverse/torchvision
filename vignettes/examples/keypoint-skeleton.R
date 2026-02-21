library(torchvision)
library(torch)

# Create sample image
image <- torch_randint(180, 240, c(3, 400, 400))$to(torch_uint8())

# Define 13-point human pose
keypoints <- torch_tensor(array(c(
  200, 80,   # 1. head
  200, 120,  # 2. neck
  160, 160,  # 3. left shoulder
  240, 160,  # 4. right shoulder
  140, 220,  # 5. left elbow
  260, 220,  # 6. right elbow
  130, 270,  # 7. left wrist
  270, 270,  # 8. right wrist
  200, 240,  # 9. torso
  180, 300,  # 10. left hip
  220, 300,  # 11. right hip
  170, 360,  # 12. left knee
  230, 360   # 13. right knee
), dim = c(1, 13, 2)))

# Define skeleton connections
skeleton <- matrix(c(
  1,2,  2,3,  2,4,  3,5,  4,6,  5,7,  6,8,  
  2,9,  9,10, 9,11, 10,12, 11,13
), ncol = 2, byrow = TRUE)

# Draw keypoints with skeleton
result <- draw_keypoints(image, keypoints, connectivity = skeleton, 
                         colors = "cyan", radius = 4, width = 2)

cat("Result shape:", paste(result$shape, collapse = "x"), "\n")
tensor_image_browse(result)
