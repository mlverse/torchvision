# Run from the PR checkout after creating a separate main checkout:
# git worktree add --detach /tmp/torchvision-crop-main-414 origin/main
library(torch)
pkgload::load_all("/tmp/torchvision-crop-main-414", quiet = TRUE)

item <- list(x = torch_zeros(3, 100, 100), y = list())
item$x[, 11:20, 11:20] <- 1
item$y$boxes <- torch_tensor(rbind(c(10, 10, 20, 20)))
class(item) <- "image_with_bounding_box"

before <- draw_bounding_boxes(item, colors = "red")
main <- item_transform_center_crop(item, 120)
after_main <- draw_bounding_boxes(main, colors = "red")

pkgload::load_all(".", quiet = TRUE)
pr <- item_transform_center_crop(item, 120)
after_pr <- draw_bounding_boxes(pr, colors = "red")

lapply(list(before, after_main, after_pr), tensor_image_browse)
