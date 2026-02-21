#' @importFrom magrittr %>%
NULL

# ============================================================================ #
#' A simplified version of torchvision.utils.make_grid
#'
#' Arranges a batch B of (image) tensors in a grid, with optional padding between images.
#' Expects a 4d mini-batch tensor of shape (B x C x H x W).
#'
#' @param tensor tensor of shape (B x C x H x W) to arrange in grid.
#' @param scale whether to normalize (min-max-scale) the input tensor.
#' @param num_rows number of rows making up the grid (default 8).
#' @param padding amount of padding between batch images (default 2).
#' @param pad_value pixel value to use for padding.
#'
#' @return a 3d torch_tensor of shape \eqn{\approx(C, num_rows * H, num_cols * W)}.
#' @family image display
#' @export
vision_make_grid <- function(tensor,
                             scale = TRUE,
                             num_rows = 8,
                             padding = 2,
                             pad_value = 0) {
  tensor <- tensor$clone()
  if (scale) {
    for (i in seq_len(tensor$size(1))) {
      img <- tensor[i, , , ]
      min_val <- img$min()$item()
      max_val <- img$max()$item()
      img$clamp_(min = min_val, max = max_val)
      img$add_(-min_val)$div_(max_val - min_val + 1e-5)
    }
  }

  nmaps <- tensor$size(1)
  xmaps <- min(num_rows, nmaps)
  ymaps <- ceiling(nmaps / xmaps)
  height <- floor(tensor$size(3) + padding)
  width <- floor(tensor$size(4) + padding)
  num_channels <- tensor$size(2)
  grid <- tensor$new_full(c(num_channels, height * ymaps + padding, width * xmaps + padding),
                          pad_value)
  k <- 0

  for (y in 0:(ymaps - 1)) {
    for (x in 0:(xmaps - 1)) {
      if (k >= nmaps) break
      y_start <- y * height + padding + 1
      x_start <- x * width + padding + 1
      grid$narrow(dim = 2, start = y_start, length = height - padding)$
        narrow(dim = 3, start = x_start, length = width - padding)$
        copy_(tensor[k + 1, , ,])
      k <- k + 1
    }
  }
  grid
}

# ============================================================================ #
#' Draw bounding boxes on image
#' @param x Tensor (C x H x W) uint8 or float ([0,1])
#' @param boxes Tensor (N x 4) in c(xmin, ymin, xmax, ymax)
#' @param labels character vector of box labels
#' @param colors character vector or single color
#' @param fill whether to fill the box
#' @param width width of the box border
#' @param font font family vector
#' @param font_size font size in points
#' @return torch_tensor (C x H x W) with boxes drawn
#' @family image display
#' @export
draw_bounding_boxes <- function(x, ...) {
  UseMethod("draw_bounding_boxes")
}

draw_bounding_boxes.default <- function(x, ...) {
  cli_abort("The provided x class {.class {class(x)}} is not supported")
}

draw_bounding_boxes.torch_tensor <- function(x,
                                             boxes,
                                             labels = NULL,
                                             colors = NULL,
                                             fill = FALSE,
                                             width = 1,
                                             font = c("serif", "plain"),
                                             font_size = 10, ...) {
  rlang::check_installed("magick")

  if (x$ndim == 4 && x$size(1) == 1) x <- x$squeeze(1)
  if (x$ndim != 3) value_error("Pass an individual image as `x`, not a batch")
  if (!x$size(1) %in% c(1, 3)) value_error("Only grayscale and RGB images are supported")

  img_to_draw <- if (x$dtype == torch::torch_uint8()) {
    x$div(255)$permute(c(2, 3, 1))$to(device = "cpu") %>% as.array()
  } else if (x$dtype == torch::torch_float()) {
    x$permute(c(2, 3, 1))$to(device = "cpu") %>% as.array()
  } else type_error("`x` should be torch_uint8 or torch_float")

  if ((boxes[, 1] >= boxes[, 3])$any() %>% as.logical() ||
      (boxes[, 2] >= boxes[, 4])$any() %>% as.logical()) {
    value_error("Boxes must be in c(xmin, ymin, xmax, ymax) format")
  }

  num_boxes <- boxes$shape[1]
  if (num_boxes == 0) {
    cli_warn("boxes doesn't contain any box. No box was drawn")
    return(x)
  }

  if (!is.null(labels) && (num_boxes %% length(labels) != 0)) {
    cli_abort("Number of labels cannot be broadcasted on boxes")
  }

  if (is.null(colors)) colors <- grDevices::hcl.colors(n = num_boxes)
  if (num_boxes %% length(colors) != 0) value_error("colors vector cannot be broadcasted on boxes")
  fill_col <- ifelse(fill, colors, NA)
  if (is.null(font)) font <- c("serif", "plain")
  if (x$size(1) == 1) x <- x$repeat_interleave(3, dim = 1)

  img_bb <- boxes$to(torch::torch_int64()) %>% as.array()
  H <- nrow(img_to_draw)
  img_bb[, c(2,4)] <- H - img_bb[, c(4,2)]

  tmp <- tempfile(fileext = ".png")
  png::writePNG(img_to_draw, tmp)
  draw <- magick::image_read(tmp) %>% magick::image_draw()
  unlink(tmp)

  graphics::rect(img_bb[,1], img_bb[,2], img_bb[,3], img_bb[,4],
                 col = fill_col, border = colors, lwd = width)

  if (!is.null(labels)) {
    graphics::text(img_bb[,1] + 2*width + font_size,
                   img_bb[,2] + 2*width,
                   labels = labels, col = colors,
                   vfont = font, cex = font_size / 10)
  }

  grDevices::dev.off()
  draw_tt <- draw %>% magick::image_data(channels="rgb") %>% as.integer() %>%
    torch::torch_tensor(dtype=torch::torch_uint8())
  draw_tt$permute(c(3,1,2))
}

draw_bounding_boxes.image_with_bounding_box <- function(x, ...) {
  draw_bounding_boxes(x$x, boxes = x$y$boxes, labels = x$y$labels, ...)
}

# ============================================================================ #
#' Convert COCO polygon to mask tensor
#' @keywords internal
coco_polygon_to_mask <- function(segmentation, height, width) {
  rlang::check_installed("magick")
  if (length(segmentation) == 0) {
    return(torch::torch_tensor(matrix(FALSE, nrow=height, ncol=width), dtype=torch::torch_bool()))
  }

  mask_img <- magick::image_blank(width=width, height=height, color="black")
  mask_img <- magick::image_draw(mask_img)

  for (poly in segmentation) {
    flat <- unlist(poly)
    if (length(flat) %% 2 != 0) flat <- flat[-length(flat)]
    if (length(flat) >= 6) {
      coords <- matrix(flat, ncol=2, byrow=TRUE)
      coords[,2] <- height - coords[,2]
      graphics::polygon(coords[,1], coords[,2], col="white", border=NA)
    }
  }
  grDevices::dev.off()
  gray <- magick::image_data(mask_img, channels="gray")
  mask_matrix <- if (length(dim(gray)) == 3) {
    t(as.matrix(gray[1,,]))
  } else if (length(dim(gray)) == 2) {
    t(as.matrix(gray))
  } else {
    matrix(as.vector(gray), nrow=height, ncol=width, byrow=TRUE)
  }
  torch::torch_tensor(mask_matrix > 0, dtype=torch::torch_bool())
}

# ============================================================================ #
#' Draw segmentation masks
draw_segmentation_masks <- function(x, ...) UseMethod("draw_segmentation_masks")
draw_segmentation_masks.default <- function(x, ...) {
  type_error("The provided object of class {.cls {class(x)}} is not supported by draw_segmentation_masks.")
}

draw_segmentation_masks.torch_tensor <- function(x, masks, alpha=0.8, colors=NULL, ...) {
  rlang::check_installed("magick")
  out_dtype <- torch::torch_uint8()
  if (x$ndim != 3) value_error("Pass individual `image`, not batches")
  if (!x$size(1) %in% c(1,3)) value_error("Only grayscale and RGB images are supported")

  img_to_draw <- if (x$dtype == out_dtype) {
    x$detach()$clone()
  } else if (x$dtype == torch::torch_float()) {
    x$detach()$clone()$mul(255)$to(dtype=out_dtype)
  } else type_error("`x` should be torch_uint8 or torch_float")

  if (masks$ndim == 2) masks <- masks$unsqueeze(1)
  if (masks$ndim != 3) value_error("`masks` must be shape (H, W) or (num_masks, H, W)")
  if (!(masks$dtype == torch::torch_bool() || masks$dtype == torch::torch_float())) {
    type_error("`masks` should be torch_bool() or torch_float()")
  }
  if (any(masks$shape[-2:-1] != img_to_draw$shape[-2:-1])) {
    value_error("`masks` and `image` must have same height and width")
  }
  if (masks$dtype == torch::torch_float()) masks <- masks$sigmoid()$gt(0.5)

  num_masks <- masks$size(1)
  if (num_masks == 0) {
    cli_warn("masks doesn't contain any mask. No mask was drawn")
    return(x)
  }

  if (is.null(colors)) colors <- grDevices::hcl.colors(n=num_masks)
  if (num_masks %% length(colors) != 0) cli_abort("colors vector cannot be broadcasted on masks")

  color_tt <- grDevices::col2rgb(colors) %>% t() %>% torch::torch_tensor(dtype=out_dtype)
  out <- img_to_draw$clone()
  for (i in seq_len(num_masks)) {
    mask <- masks[i,,]$unsqueeze(1)
    color <- color_tt[i,]$view(c(3,1,1))
    out <- torch::torch_where(mask, out*(1-alpha) + color*alpha, out)
  }
  out$to(out_dtype)
}

draw_segmentation_masks.image_with_segmentation_mask <- function(x, alpha=0.5, colors=NULL, ...) {
  draw_segmentation_masks(x$x, masks=x$y$masks, alpha=alpha, colors=colors, ...)
}

# ============================================================================ #
#' Draw Keypoints
#'
#' Draws keypoints and optional skeleton connections on an image tensor.
#'
#' @param image Tensor of shape (3 x H x W) and dtype `uint8` or `float`.
#' @param keypoints Tensor of shape (num_instances, K, 2) containing K keypoint 
#'   coordinates (x, y) for each of the N instances.
#' @param connectivity Optional matrix with 2 columns defining skeleton edges. 
#'   Each row specifies a pair of keypoint indices to connect with a line.
#' @param colors Character vector of colors for each instance. By default, 
#'   viridis colors are used.
#' @param radius Numeric value controlling the size of keypoint circles.
#' @param width Numeric value controlling the width of skeleton lines.
#'
#' @return Image tensor of dtype uint8 with keypoints and skeleton drawn.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' \dontrun{
#' # Create sample image
#' img <- torch_randint(190, 255, c(3, 400, 300))$to(torch_uint8())
#' 
#' # Define 5 keypoints for one pose
#' kpts <- torch_tensor(array(c(150,50, 120,100, 180,100, 130,200, 170,200), 
#'                            dim=c(1,5,2)))
#' 
#' # Define skeleton connections
#' skeleton <- matrix(c(1,2, 1,3, 2,4, 3,5), ncol=2, byrow=TRUE)
#' 
#' # Draw keypoints with skeleton
#' result <- draw_keypoints(img, kpts, connectivity=skeleton)
#' tensor_image_browse(result)
#' }
#' }
#' @family image display
#' @export
draw_keypoints <- function(image, keypoints, connectivity=NULL, colors=NULL, radius=2, width=3) {
  rlang::check_installed("magick")
  if (!inherits(image, "torch_tensor")) type_error("`image` should be torch_tensor")
  if (image$ndim != 3) value_error("Pass individual `image`, not batches")
  if (!image$size(1) %in% c(1,3)) value_error("Only grayscale and RGB images supported")

  img_to_draw <- if (image$dtype == torch::torch_uint8()) {
    image$to(torch::torch_float())$div(255)$permute(c(2,3,1))$to(device="cpu") %>% as.array()
  } else if (image$dtype == torch::torch_float()) {
    image$permute(c(2,3,1))$to(device="cpu") %>% as.array()
  } else type_error("`image` should be torch_uint8 or torch_float")

  if (keypoints$ndim != 3) {
    cli_abort("keypoints must be shape (num_instances, K, 2), got {.value {keypoints$shape}}")
  }

  img_kpts <- keypoints$cpu() %>% as.array()
  num_instances <- dim(img_kpts)[1]
  K <- dim(img_kpts)[2]

  if (!is.null(connectivity)) {
    if (any(connectivity < 1 | connectivity > K)) value_error("connectivity indices exceed number of keypoints")
  }

  if (is.null(colors)) colors <- grDevices::hcl.colors(n=max(num_instances,1))
  colors <- rep(colors, length.out=num_instances)

  tmp <- tempfile(fileext=".png")
  png::writePNG(img_to_draw, tmp)
  draw <- magick::image_read(tmp) %>% magick::image_draw()
  unlink(tmp)

  H <- dim(img_to_draw)[1]
  for (pose in seq_len(num_instances)) {
    kpt <- img_kpts[pose,,]
    pose_color <- colors[pose]
    if (!is.null(connectivity)) {
      for (i in seq_len(nrow(connectivity))) {
        a <- connectivity[i,1]; b <- connectivity[i,2]
        graphics::segments(round(kpt[a,1]), H-round(kpt[a,2]),
                           round(kpt[b,1]), H-round(kpt[b,2]),
                           col=pose_color, lwd=width)
      }
    }
    graphics::points(round(kpt[,1]), H-round(kpt[,2]), pch=20, col=pose_color, cex=radius)
  }

  grDevices::dev.off()
  draw_tt <- draw %>% magick::image_data(channels="rgb") %>% as.integer() %>%
    torch::torch_tensor(dtype=torch::torch_uint8())
  draw_tt$permute(c(3,1,2))
}

# ============================================================================ #
#' Display image tensor to X11 device
tensor_image_display <- function(image, animate=TRUE) {
  if (image$ndim != 3) value_error("Pass individual `image`, not batches")
  if (!image$size(1) %in% c(1,3)) value_error("Only grayscale and RGB images are supported")

  img_to_draw <- if (image$dtype == torch::torch_uint8()) {
    image$permute(c(2,3,1))$to(device="cpu")$to(torch::torch_float())$div(255) %>% as.array()
  } else image$permute(c(2,3,1))$to(device="cpu") %>% as.array()

  if (dim(img_to_draw)[3]==1) img_to_draw <- img_to_draw[,,1]
  tmp <- tempfile(fileext=".png")
  png::writePNG(img_to_draw, tmp)
  magick::image_read(tmp) %>% magick::image_display(animate=animate)
  unlink(tmp)
  invisible(NULL)
}

# ============================================================================ #
#' Display image tensor in browser
tensor_image_browse <- function(image, browser=getOption("browser")) {
  if (image$ndim != 3) value_error("Pass individual `image`, not batches")
  if (!image$size(1) %in% c(1,3)) value_error("Only grayscale and RGB images are supported")

  img_to_draw <- if (image$dtype == torch::torch_uint8()) {
    image$permute(c(2,3,1))$to(device="cpu")$to(torch::torch_float())$div(255) %>% as.array()
  } else image$permute(c(2,3,1))$to(device="cpu") %>% as.array()

  if (dim(img_to_draw)[3]==1) img_to_draw <- img_to_draw[,,1]
  tmp <- tempfile(fileext=".png")
  png::writePNG(img_to_draw, tmp)
  magick::image_read(tmp) %>% magick::image_browse(browser=browser)
  unlink(tmp)
  invisible(NULL)
}
