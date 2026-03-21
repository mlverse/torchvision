library(torchvision)
library(torch)

# Create sample image
image <- torch_zeros(c(3, 400, 400), dtype = torch_uint8())

# Define 14-point human pose (tree structure)
# 1 head, 2 neck
# 3 left shoulder, 4 right shoulder
# 5 left elbow, 6 right elbow
# 7 left wrist, 8 right wrist
# 9 left hip, 10 right hip
# 11 left knee, 12 right knee
# 13 left ankle, 14 right ankle
keypoints_xy <- matrix(c(
  200, 35,   # 1. head
  200, 80,   # 2. neck
  145, 110,  # 3. left shoulder
  255, 110,  # 4. right shoulder
  120, 175,  # 5. left elbow
  280, 175,  # 6. right elbow
  95, 240,   # 7. left wrist
  305, 240,  # 8. right wrist
  165, 220,  # 9. left hip
  235, 220,  # 10. right hip
  158, 300,  # 11. left knee
  242, 300,  # 12. right knee
  152, 380,  # 13. left ankle
  248, 380   # 14. right ankle
), ncol = 2, byrow = TRUE)

# Build (N, K, 2) with explicit channel ordering: all x first, then all y.
keypoints <- torch_tensor(array(
  c(keypoints_xy[, 1], keypoints_xy[, 2]),
  dim = c(1, nrow(keypoints_xy), 2)
))

# Define skeleton connections
skeleton <- matrix(c(
  1,2,
  2,3,  3,5,  5,7,
  2,4,  4,6,  6,8,
  2,9,  9,11, 11,13,
  2,10, 10,12, 12,14
), ncol = 2, byrow = TRUE)

# Draw keypoints with skeleton
result <- draw_keypoints(image, keypoints, connectivity = skeleton, 
                         colors = "red", radius = 5, width = 3)

cat("Result shape:", paste(result$shape, collapse = "x"), "\n")
tensor_image_browse(result)
