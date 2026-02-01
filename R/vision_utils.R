#' @importFrom magrittr %>%
NULL

#' A simplified version of torchvision.utils.make_grid
#'
#' Arranges a batch B of (image) tensors in a grid, with optional padding between
#'   images. Expects a 4d mini-batch tensor of shape (B x C x H x W).
#'
#' @param tensor tensor of shape (B x C x H x W) to arrange in grid.
#' @param scale whether to normalize (min-max-scale) the input tensor.
#' @param num_rows number of rows making up the grid (default 8).
#' @param padding amount of padding between batch images (default 2).
#' @param pad_value pixel value to use for padding.
#'
#' @return  a 3d torch_tensor of shape \eqn{\approx(C , num\_rows \times H , num\_cols \times W)} of all images arranged in a grid.
#' @family image display
#' @export
vision_make_grid <- function(tensor,
                             scale = TRUE,
                             num_rows = 8,
                             padding = 2,
                             pad_value = 0) {

  min_max_scale <- function(x) {
    min <- x$min()$item()
    max <- x$max()$item()
    x$clamp_(min = min, max = max)
    x$add_(-min)$div_(max - min + 1e-5)
    x
  }
  if(scale) tensor <- min_max_scale(tensor)

  nmaps <- tensor$size(1)
  xmaps <- min(num_rows, nmaps)
  ymaps <- ceiling(nmaps / xmaps)
  height <- floor(tensor$size(3) + padding)
  width <- floor(tensor$size(4) + padding)
  num_channels <- tensor$size(2)
  grid <-
    tensor$new_full(c(num_channels, height * ymaps + padding, width * xmaps + padding),
                    pad_value)
  k <- 0

  for (y in 0:(ymaps - 1)) {
    for (x in 0:(xmaps - 1)) {
      if (k >= nmaps)
        break
      grid$narrow(
        dim = 2,
        start =  1 + torch::torch_tensor(y * height + padding, dtype = torch::torch_int64())$sum(dim = 1),
        length = height - padding
      )$narrow(
        dim = 3,
        start = 1 + torch::torch_tensor(x * width + padding, dtype = torch::torch_int64())$sum(dim = 1),
        length = width - padding
      )$copy_(tensor[k + 1, , ,])
      k <- k + 1
    }
  }

  grid
}


#' Draws bounding boxes on image.
#'
#' Draws bounding boxes on top of one image tensor
#'
#' @param x Tensor of shape (C x H x W) and dtype `uint8` or dtype `float`.
#'              In case of dtype float, values are assumed to be in range \eqn{[0, 1]}.
#'              C value for channel can only be 1 (grayscale) or 3 (RGB).
#' @param boxes Tensor of size (N, 4) containing N bounding boxes in
#'            c(\eqn{x_{min}}, \eqn{y_{min}}, \eqn{x_{max}}, \eqn{y_{max}}).
#'            format. Note that the boxes coordinates are absolute with respect
#'            to the image. In other words: \eqn{0  \leq x_{min} < x_{max} < Height } and
#'            \eqn{0  \leq y_{min} < y_{max} < Width }.
#' @param labels character vector containing the labels of bounding boxes.
#' @param colors character vector containing the colors
#'            of the boxes or single color for all boxes. The color can be represented as
#'            strings e.g. "red" or "#FF00FF". By default, viridis colors are generated for boxes.
#' @param fill If `TRUE` fills the bounding box with specified color.
#' @param width  Width of text shift to the bounding box.
#' @param font NULL for the current font family, or a character vector of length 2 for Hershey vector fonts.
#' @param font_size The requested font size in points.
#' @param ... Additional arguments passed to methods.
#'
#' @return  torch_tensor of size (C, H, W) of dtype uint8: Image Tensor with bounding boxes plotted.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' \dontrun{
#' image_tensor <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
#' x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
#' y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
#' boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
#' bboxed <- draw_bounding_boxes(image_tensor, boxes, colors = "black", fill = TRUE)
#' tensor_image_browse(bboxed)
#' }
#' }
#' @family image display
#' @export
draw_bounding_boxes <- function(x, ...) {
  UseMethod("draw_bounding_boxes")
}

#' @rdname draw_bounding_boxes
#' @export
draw_bounding_boxes.default <- function(x, ...) {
  cli_abort("The provided x class {.class {class(x)}} is not supported")
}

#' @rdname draw_bounding_boxes
#' @export
draw_bounding_boxes.torch_tensor <- function(x,
                                             boxes,
                                             labels = NULL,
                                             colors = NULL,
                                             fill = FALSE,
                                             width = 1,
                                             font = c("serif", "plain"),
                                             font_size = 10, ...) {
  rlang::check_installed("magick")

  # manage single batch images
  if (x$ndim == 4 && x$size(1) == 1) {
    x <- x$squeeze(1)
  }
  if (x$ndim != 3) {
    value_error("Pass an individual image as `x`, not a batch")
  }
  if (!x$size(1) %in% c(1, 3)) {
    value_error("Only grayscale and RGB images are supported")
  }
  if (x$dtype == torch::torch_uint8()) {
    img_to_draw <- x$div(255)$permute(c(2, 3, 1))$to(device = "cpu") %>% as.array
  } else if (x$dtype == torch::torch_float()) {
    img_to_draw <- x$permute(c(2, 3, 1))$to(device = "cpu") %>% as.array
  } else {
    type_error("`x` should be of dtype `torch_uint8` or `torch_float`")
  }
  if ((boxes[, 1] >= boxes[, 3])$any() %>% as.logical() || (boxes[, 2] >= boxes[, 4])$any() %>% as.logical()) {
    value_error("Boxes need to be in c(xmin, ymin, xmax, ymax) format. Use `box_convert()` to convert them")
  }
  num_boxes <- boxes$shape[1]
  if (num_boxes == 0) {
    cli_warn("boxes doesn't contain any box. No box was drawn")
    return(x)
  }
  if (!is.null(labels) && (num_boxes %% length(labels) != 0)) {
    cli_abort(
      "Number of labels {.val {length(labels)}} cannot be broadcasted on number of boxes {.val {num_boxes}}"
    )
  }
  if (is.null(colors)) {
    colors <- grDevices::hcl.colors(n = num_boxes)
  }
  if (num_boxes %% length(colors) != 0) {
    value_error("colors vector cannot be broadcasted on boxes")
  }

  if (!fill) {
    fill_col <- NA
  } else {
    fill_col <- colors
  }

  if (is.null(font)) {
    vfont <- c("serif", "plain")
  } else {
    if (is.null(font_size)) font_size <- 10
  }

  if (x$size(1) == 1) {
    x <- x$tile(c(4, 2, 2))
  }

  img_bb <- boxes$to(torch::torch_int64()) %>% as.array

  draw <- png::writePNG(img_to_draw) %>%
    magick::image_read() %>%
    magick::image_draw()

  graphics::rect(img_bb[, 1], img_bb[, 2], img_bb[, 3], img_bb[, 4], col = fill_col, border = colors)

  if (!is.null(labels)) {
    graphics::text(
      img_bb[, 1] + width,
      img_bb[, 2] + width,
      labels = labels,
      col = colors,
      vfont = font,
      cex = font_size / 10
    )
  }

  grDevices::dev.off()

  draw_tt <- draw %>%
    magick::image_data(channels = "rgb") %>%
    as.integer() %>%
    torch::torch_tensor(dtype = torch::torch_uint8())

  return(draw_tt$permute(c(3, 1, 2)))
}

#' @rdname draw_bounding_boxes
#' @export
draw_bounding_boxes.image_with_bounding_box <- function(x, ...) {
  draw_bounding_boxes(
    x = x$x,
    boxes = x$y$boxes,
    labels = x$y$labels,
    ...
  )
}



#' Convert COCO polygon to mask tensor (Robust Version)
#'
#' Converts a COCO-style polygon annotation (list of coordinates) into a binary mask tensor.
#'
#' @param segmentation A list of polygons from COCO annotations (e.g., \code{anns$segmentation[[i]]}).
#' @param height Height of the image
#' @param width Width of the image
#'
#' @return A torch_bool() tensor of shape (height, width)
#'
#' @keywords internal
coco_polygon_to_mask <- function(segmentation, height, width) {
  rlang::check_installed("magick")

  mask_img <- magick::image_blank(width = width, height = height, color = "black")
  mask_img <- magick::image_draw(mask_img)

  for (poly in segmentation) {
    flat <- unlist(poly)
    n_coords <- length(flat)

    # Ensure number of coordinates is even (x, y) pairs
    if (n_coords %% 2 != 0) {
      flat <- flat[-n_coords]  # Drop the last element if odd
    }

    if (length(flat) >= 6) {  # At least 3 points to form a polygon
      coords <- matrix(flat, ncol = 2, byrow = TRUE)
      polygon(coords[, 1], coords[, 2], col = "white", border = NA)
    }
  }

  dev.off()

  gray <- magick::image_data(mask_img, channels = "gray")

  if (length(dim(gray)) == 3) {
    mask_matrix <- t(as.matrix(gray[1, , ]))
  } else if (length(dim(gray)) == 2) {
    mask_matrix <- t(as.matrix(gray))
  } else {
    gray_vec <- as.vector(gray)
    mask_matrix <- matrix(gray_vec, nrow = height, ncol = width, byrow = TRUE)
  }

  if (nrow(mask_matrix) != height || ncol(mask_matrix) != width) {
    stop(sprintf("Mask matrix dimensions (%d x %d) don't match expected (%d x %d)",
                 nrow(mask_matrix), ncol(mask_matrix), height, width))
  }

  mask_logical <- mask_matrix > 0
  mask_tensor <- torch::torch_tensor(mask_logical, dtype = torch::torch_bool())

  return(mask_tensor)
}


#' Draw segmentation masks
#'
#' Draw segmentation masks with their respective colors on top of a given RGB tensor image
#'
#' @param x Tensor of shape (C x H x W) and dtype `uint8` or dtype `float`.
#'              In case of dtype float, values are assumed to be in range \eqn{[0, 1]}.
#'              C value for channel can only be 1 (grayscale) or 3 (RGB).
#' @param masks torch_tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
#' @param alpha number between 0 and 1 denoting the transparency of the masks.
#   0 means full transparency, 1 means no transparency.
#' @param colors character vector containing the colors
#'            of the boxes or single color for all boxes. The color can be represented as
#'            strings e.g. "red" or "#FF00FF". By default, viridis colors are generated for masks
#' @param ... Additional arguments passed to methods.
#'
#' @importFrom graphics polygon
#' @importFrom grDevices dev.off
#' @importFrom torch as_array
#'
#' @return torch_tensor of shape (3, H, W) and dtype uint8 of the image with segmentation masks drawn on top.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' image_tensor <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
#' mask <- torch::torch_tril(torch::torch_ones(c(360, 360)))$to(torch::torch_bool())
#' masked_image <- draw_segmentation_masks(image_tensor, mask, alpha = 0.2)
#' tensor_image_browse(masked_image)
#' }
#' @family image display
#' @export
draw_segmentation_masks <- function(x, ...) {
  UseMethod("draw_segmentation_masks")
}

#' @rdname draw_segmentation_masks
#' @export
draw_segmentation_masks.default <- function(x, ...) {
  type_error("The provided object of class {.cls {class(x)}} is not supported by draw_segmentation_masks.")
}

#' @rdname draw_segmentation_masks
#' @export
draw_segmentation_masks.torch_tensor <- function(x,
                                                 masks,
                                                 alpha = 0.8,
                                                 colors = NULL, ...) {
  rlang::check_installed("magick")
  out_dtype <- torch::torch_uint8()

  if (x$ndim != 3) {
    value_error("Pass individual `image`, not batches")
  }
  if (!x$size(1) %in% c(1, 3)) {
    value_error("Only grayscale and RGB images are supported")
  }
  if (x$dtype == out_dtype) {
    img_to_draw <- x$detach()$clone()
  } else if (x$dtype == torch::torch_float()) {
    img_to_draw <- x$detach()$clone()$mul(255)$to(dtype = out_dtype)
  } else {
    type_error("`x` (image) should be of dtype `torch_uint8` or `torch_float`")
  }

  if (masks$ndim == 2) {
    masks <- masks$unsqueeze(1)
  }
  if (masks$ndim != 3) {
    value_error("`masks` must be of shape (H, W) or (num_masks, H, W)")
  }
  # datasets item include boolean masks, and models inference produce logits floats for masks
  if (masks$dtype != torch::torch_bool() && masks$dtype != torch::torch_float() ) {
    type_error("`masks` is expected to be of dtype torch_bool() or torch_float()")
  }
  if (any(masks$shape[-2:-1] != img_to_draw$shape[-2:-1])) {
    value_error("`masks` and `image` must have the same height and width")
  }
  # if mask is a model inference output, we need to convert float mask to boolean mask
  if (masks$dtype == torch::torch_float() ) {
    mask_id <- masks$argmax(dim = 1)
    masks_seq <- mask_id$aminmax()[[1]]$item():mask_id$aminmax()[[2]]$item()
    # turn mask_id \code{[LongType{H,W}]} into a boolean mask \code{[BoolType{num_masks,H,W}]}
    masks <- torch::torch_stack(lapply(masks_seq, function(x) mask_id$eq(x)), dim = 1)
  } else {
    masks_seq <- seq(masks$size(1))
  }
  num_masks <- length(masks_seq)
  if (num_masks == 0) {
    cli_warn("masks doesn't contain any mask. No mask was drawn")
    return(x)
  }

  if (is.null(colors)) {
    colors <- grDevices::hcl.colors(n = num_masks)
  }
  if (num_masks %% length(colors) != 0) {
    cli_abort("colors vector of size {.value {length(colors)}} cannot be broadcasted on {.value {num_masks}} masks")
  }

  color_tt <- colors %>%
    grDevices::col2rgb() %>%
    t() %>%
    torch::torch_tensor(dtype = out_dtype)

  colored_mask_stack <- torch::torch_stack(lapply(
    masks_seq,
    function(i) color_tt[i, ]$unsqueeze(2)$unsqueeze(2)$mul(masks[i:i, , ])
  ), dim = 1)

  out <- img_to_draw * (1 - alpha) + torch::torch_sum(colored_mask_stack, dim = 1) * alpha
  return(out$to(out_dtype))
}

#' @rdname draw_segmentation_masks
#' @export
draw_segmentation_masks.image_with_segmentation_mask <- function(x,
                                                                 alpha = 0.5,
                                                                 colors = NULL, ...) {
  draw_segmentation_masks(
    x = x$x,
    masks = x$y$masks,
    alpha = alpha,
    colors = colors,
    ...
  )
}


#' Draws Keypoints
#'
#' Draws Keypoints, an object describing a body part (like rightArm or leftShoulder), on given RGB tensor image.
#' @param image Tensor of shape (3 x H x W) and dtype `uint8` or dtype `float`.
#'              In case of dtype float, values are assumed to be in range \eqn{[0, 1]}.
#' @param keypoints Tensor of shape (N, K, 2) the K keypoints location for each of the N detected poses instance,
#         in the format c(x, y).
#' @param connectivity Vector of pair of keypoints to be connected (currently unavailable)
#' @param colors character vector containing the colors
#'            of the boxes or single color for all boxes. The color can be represented as
#'            strings e.g. "red" or "#FF00FF". By default, viridis colors are generated for keypoints
#' @param radius radius of the plotted keypoint.
#' @param width width of line connecting keypoints.
#'
#' @return Image Tensor of dtype uint8 with keypoints drawn.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' \dontrun{
#' image <- torch::torch_randint(190, 255, size = c(3, 360, 360))$to(torch::torch_uint8())
#' keypoints <- torch::torch_randint(low = 60, high = 300, size = c(4, 5, 2))
#' keypoint_image <- draw_keypoints(image, keypoints)
#' tensor_image_browse(keypoint_image)
#' }
#' }
#' @family image display
#' @export
draw_keypoints <- function(image,
                           keypoints,
                           connectivity = NULL,
                           colors = NULL,
                           radius = 2,
                           width = 3) {

  rlang::check_installed("magick")
  if (!inherits(image, "torch_tensor")) {
    type_error("`image` should be a torch_tensor")
  }
  if (image$ndim != 3) {
    value_error("Pass individual `image`, not batches")
  }
  if (!image$size(1) %in% c(1, 3)) {
    value_error("Only grayscale and RGB images are supported")
  }
  if (image$dtype == torch::torch_uint8()) {
    img_to_draw <- image$div(255)$permute(c(2, 3, 1))$to(device = "cpu") %>% as.array
  } else if (image$dtype == torch::torch_float()) {
    img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu") %>% as.array
  } else {
    type_error("`image` should be of dtype `torch_uint8` or `torch_float`")
  }

  if (keypoints$ndim != 3) {
    cli_abort("{.var keypoints} must be of shape (num_instances, K, 2), but is current shape is {.value {keypoints$shape}}")
  }

  img_kpts <- keypoints$to(torch::torch_int64()) %>% as.array
  draw <- png::writePNG(img_to_draw / 255) %>%
    magick::image_read() %>%
    magick::image_draw()

  for (pose in dim(img_kpts)[[1]]) {
    graphics::points(img_kpts[pose,,1],img_kpts[pose,,2], pch = ".", col = colors, cex = radius)

  }
  # TODO need R-isation and vectorisation
  # for (kpt_id, kpt_inst in enumerate(img_kpts)) {
  #     if (connectivity) {
  #         for (connection in connectivity) {
  #             start_pt_x <- kpt_inst[connection[0]][0]
  #             start_pt_y <- kpt_inst[connection[0]][1]
  #
  #             end_pt_x <- kpt_inst[connection[1]][0]
  #             end_pt_y <- kpt_inst[connection[1]][1]
  #
  #             draw$line(
  #                 ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
  #                 widt = width,
  #             )
  #         }
  #     }
  # }
  grDevices::dev.off()
  draw_tt <-
    draw %>% magick::image_data(channels = "rgb") %>% as.integer %>% torch::torch_tensor(dtype = torch::torch_uint8())

  return(draw_tt$permute(c(3, 1, 2)))
}


#' Display image tensor
#'
#' Display image tensor onto the X11 device
#' @param image `torch_tensor()` of shape (1, W, H) for grayscale image or (3, W, H) for
#'  color image to display
#' @param animate support animations in the X11 display
#'
#' @family image display
#' @export
tensor_image_display <- function(image, animate = TRUE) {
  if (image$ndim != 3) {
    value_error("Pass individual `image`, not batches")
  }
  if (!image$size(1) %in% c(1, 3)) {
    value_error("Only grayscale and RGB images are supported")
  }

  if (image$dtype == torch::torch_uint8()) {
    img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu", dtype = torch::torch_long()) %>%
      as.array() / 255
  } else {
    img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu") %>%
      as.array()
  }
  png::writePNG(img_to_draw) %>% magick::image_read() %>% magick::image_display(animate = animate)

  invisible(NULL)
}


#' Display image tensor
#'
#' Display image tensor into browser
#' @param image `torch_tensor()` of shape (1, W, H) for grayscale image or (3, W, H) for
#'  color image to display
#' @param browser argument passed to [browseURL]
#'
#' @family image display
#' @export
tensor_image_browse <- function(image, browser = getOption("browser")) {
  if (image$ndim != 3) {
    value_error("Pass individual `image`, not batches")
  }
  if (!image$size(1) %in% c(1, 3)) {
    value_error("Only grayscale and RGB images are supported")
  }

  if (image$dtype == torch::torch_uint8()) {
    img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu", dtype = torch::torch_long()) %>%
      as.array() / 255
  } else {
    img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu") %>%
      as.array()
  }

  png::writePNG(img_to_draw) %>% magick::image_read() %>% magick::image_browse(browser = browser)

  invisible(NULL)
}
