#' Rotate dataset item
#'
#' Rotates a dataset item by a given angle around its center.
#' The canvas is expanded so that the entire rotated image is visible with no
#' cropping. Empty regions are filled with given fill color.
#'
#' The bounding boxes (if present) are shifted to account for the expanded
#' canvas and converted to rotated format via
#' \code{\link{target_transform_rotate}}. For segmentation items, the masks are
#' rotated alongside the image using nearest-neighbour sampling.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @inheritParams transform_rotate
#'
#' @return A dataset item with the image and target rotated. Detection items are
#'   returned as \code{image_with_rotated_box} with boxes in xyxyr format;
#'   segmentation items keep their class.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' angle <- 30
#'
#' img <- base_loader(url) |>
#'   transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"CAT"}))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_rotate(before, angle = angle)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' grid <- vision_make_grid(
#'   torch_stack(list(transform_resize(before_plot, c(600, 600)),
#'                    transform_resize(after_plot, c(600, 600)))),
#'   scale = TRUE
#' )
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @importFrom torch nnf_affine_grid nnf_grid_sample
#' @export
item_transform_rotate <- function(x, angle, interpolation = 2, expand = TRUE, fill = 0) {
  UseMethod("item_transform_rotate", x)
}

#' @export
item_transform_rotate.dataset <- function(x, angle, interpolation = 2, expand = TRUE, fill = 0) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_rotate(item, angle = angle, interpolation = interpolation,
                          expand = expand, fill = fill)
  }
  x
}

#' @export
item_transform_rotate.default <- function(x, angle, interpolation = 2, expand = TRUE, fill = 0) {
  cli_abort(
    "{.fn item_transform_rotate} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To rotate a raw image tensor, use {.fn transform_rotate} instead."
  )
}

#' @export
item_transform_rotate.image_with_bounding_box <- function(
  x,
  angle,
  interpolation = 2,
  expand = TRUE,
  fill = 0
) {

  rotated_img <- transform_rotate(
    x$x,
    angle = angle,
    expand = expand,
    interpolation = interpolation, # 2 = bilinear
    fill = fill # 0 is padding with black
  )

  orig_spatial <- tail(x$x$shape, 2)   # e.g., c(H_orig, W_orig)
  new_spatial  <- tail(rotated_img$shape, 2)

  x$x <- rotated_img
  x$y$boxes <- .rotate_item_boxes(x$y$boxes, angle = angle,
                                  orig_spatial = orig_spatial,
                                  new_spatial = new_spatial)
  x$y$image_height <- new_spatial[1]  # First spatial dim in torch = height
  x$y$image_width <- new_spatial[2]

  class(x) <- c("image_with_rotated_box", "list")
  x
}

#' @export
item_transform_rotate.image_with_segmentation_mask <- function(x, angle, interpolation = 2, expand = TRUE, fill = 0) {
  rotated_img <- transform_rotate(
    x$x,
    angle = angle,
    expand = expand,
    interpolation = interpolation,
    fill = fill
  )
  rotated_masks <- transform_rotate(
    x$y$masks,
    angle = angle,
    expand = expand,
    interpolation = 0,
    fill = 0
  )

  new_spatial <- tail(rotated_img$shape, 2)

  x$x <- rotated_img
  x$y$masks <- rotated_masks
  x$y$image_height <- new_spatial[1]
  x$y$image_width  <- new_spatial[2]
  x
}

#' @export
item_transform_rotate.image_with_rotated_box <- function(x, angle, interpolation = 2, expand = TRUE, fill = 0) {
  rotated_img <- transform_rotate(
    x$x,
    angle = angle,
    expand = expand,
    interpolation = interpolation,
    fill = fill
  )

  orig_spatial <- tail(x$x$shape, 2)
  new_spatial  <- tail(rotated_img$shape, 2)

  x$x <- rotated_img
  x$y$boxes <- .rotate_item_boxes(x$y$boxes, angle = angle,
                                  orig_spatial = orig_spatial,
                                  new_spatial = new_spatial)
  x$y$image_height <- new_spatial[1]
  x$y$image_width  <- new_spatial[2]
  x
}

# Rotate detection boxes so they follow an image rotated by `angle` (degrees)
# around its centre. The box centres are rotated around the original image
# centre and shifted onto the (possibly expanded) output canvas, while their
# physical size is left unchanged: unlike `target_transform_rotate`, the boxes
# are not expanded to their enclosing axis-aligned rectangle.
# `boxes` may be in xyxy (N x 4) or xyxyr (N x 5) format; the result is always
# xyxyr with the accumulated rotation angle.
#' @noRd
.rotate_item_boxes <- function(boxes, angle, orig_spatial, new_spatial) {
  n <- boxes$size(1)
  is_rotated <- boxes$size(2) == 5

  if (n == 0) {
    if (is_rotated) {
      return(boxes$clone())
    }
    angle_t <- torch_zeros(0, 1, dtype = boxes$dtype, device = boxes$device)
    return(torch_cat(list(boxes, angle_t), dim = -1L))
  }

  H <- orig_spatial[1]
  W <- orig_spatial[2]
  H2 <- new_spatial[1]
  W2 <- new_spatial[2]

  x1 <- boxes[, 1]
  y1 <- boxes[, 2]
  x2 <- boxes[, 3]
  y2 <- boxes[, 4]

  cx <- (x1 + x2) * 0.5
  cy <- (y1 + y2) * 0.5
  hw <- (x2 - x1) * 0.5
  hh <- (y2 - y1) * 0.5

  angle_rad <- deg2rad(angle)
  ct <- torch_cos(angle_rad)
  st <- torch_sin(angle_rad)

  # Rotate the box centres around the original image centre, then shift them
  # onto the output canvas. This matches the geometry of transform_rotate.
  cx_new <- W2 * 0.5 + (cx - W * 0.5) * ct + (cy - H * 0.5) * st
  cy_new <- H2 * 0.5 - (cx - W * 0.5) * st + (cy - H * 0.5) * ct

  new_xyxy <- torch_stack(list(cx_new - hw, cy_new - hh, cx_new + hw, cy_new + hh), dim = -1L)

  total_angle <- if (is_rotated) {
    boxes[, 5, drop = FALSE] + angle
  } else {
    torch_tensor(angle, dtype = boxes$dtype, device = boxes$device)$reshape(c(-1, 1))$expand(c(n, 1))
  }

  torch_cat(list(new_xyxy, total_angle), dim = -1L)
}

#' Crop a dataset item
#'
#' Crops the image inside a dataset item at the specified location and output
#' size. For detection items, bounding box coordinates are adjusted to remain
#' correct after cropping. For segmentation items, both the image and the masks
#' are cropped. For rotated-box items, box coordinates are adjusted and the
#' rotation angle is preserved.
#' If the crop area extends beyond the image boundaries, boxes that fall
#' completely outside the crop are removed, and boxes that intersect the crop
#' boundary are clipped to the cropped region.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @param top integer: Vertical component of the top left corner of the crop box.
#' @param left integer: Horizontal component of the top left corner of the crop box.
#' @param height integer: Height of the crop box.
#' @param width integer: Width of the crop box.
#'
#' @return A dataset item of the same class with the cropped image and
#'   adjusted target.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_crop(before, top = 100, left = 200, height = 500, width = 800)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' before_plot <- transform_resize(before_plot, c(2000, 3000))
#' after_plot <- transform_resize(after_plot, c(2000, 3000))
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_crop <- function(x, top, left, height, width) {
  UseMethod("item_transform_crop", x)
}

#' @export
item_transform_crop.dataset <- function(x, top, left, height, width) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_crop(item, top, left, height, width)
  }
  x
}

#' @export
item_transform_crop.default <- function(x, top, left, height, width) {
  cli_abort(
    "{.fn item_transform_crop} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To crop a raw image tensor, use {.fn transform_crop} instead."
  )
}

#' @export
item_transform_crop.image_with_bounding_box <- function(x, top, left, height, width) {
  x$x <- transform_crop(x$x, top, left, height, width)

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    orig_x1 <- boxes[, 1]
    orig_y1 <- boxes[, 2]
    orig_x2 <- boxes[, 3]
    orig_y2 <- boxes[, 4]

    offset_x <- as.numeric(left) - 1
    offset_y <- as.numeric(top) - 1

    new_x1 <- torch_clamp(orig_x1 - offset_x, 0, width)
    new_y1 <- torch_clamp(orig_y1 - offset_y, 0, height)
    new_x2 <- torch_clamp(orig_x2 - offset_x, 0, width)
    new_y2 <- torch_clamp(orig_y2 - offset_y, 0, height)

    keep <- as.logical(as_array((new_x2 > new_x1) & (new_y2 > new_y1)))
    boxes <- torch_stack(list(new_x1, new_y1, new_x2, new_y2), dim = -1L)

    if (!all(keep)) {
      boxes <- boxes[keep, ]
      if (!is.null(x$y$labels)) {
        x$y$labels <- x$y$labels[keep]
      }
      if (!is.null(x$y$area)) {
        x$y$area <- x$y$area[keep]
      }
      if (!is.null(x$y$iscrowd)) {
        x$y$iscrowd <- x$y$iscrowd[keep]
      }
    }
  }
  x$y$boxes <- boxes
  x$y$image_height <- as.integer(height)
  x$y$image_width <- as.integer(width)

  x
}

#' @export
item_transform_crop.image_with_segmentation_mask <- function(x, top, left, height, width) {
  x$x <- transform_crop(x$x, top, left, height, width)
  x$y$masks <- transform_crop(x$y$masks, top, left, height, width)
  x$y$image_height <- as.integer(height)
  x$y$image_width <- as.integer(width)

  x
}

#' @export
item_transform_crop.image_with_rotated_box <- function(x, top, left, height, width) {
  x$x <- transform_crop(x$x, top, left, height, width)

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    orig_x1 <- boxes[, 1]
    orig_y1 <- boxes[, 2]
    orig_x2 <- boxes[, 3]
    orig_y2 <- boxes[, 4]
    angle_col <- boxes[, 5, drop = FALSE]

    offset_x <- as.numeric(left) - 1
    offset_y <- as.numeric(top) - 1

    new_x1 <- torch_clamp(orig_x1 - offset_x, 0, width)
    new_y1 <- torch_clamp(orig_y1 - offset_y, 0, height)
    new_x2 <- torch_clamp(orig_x2 - offset_x, 0, width)
    new_y2 <- torch_clamp(orig_y2 - offset_y, 0, height)

    keep <- as.logical(as_array((new_x2 > new_x1) & (new_y2 > new_y1)))
    new_xy <- torch_stack(list(new_x1, new_y1, new_x2, new_y2), dim = -1L)

    if (!all(keep)) {
      new_xy <- new_xy[keep, ]
      angle_col <- angle_col[keep, ]
      if (!is.null(x$y$labels)) {
        x$y$labels <- x$y$labels[keep]
      }
      if (!is.null(x$y$area)) {
        x$y$area <- x$y$area[keep]
      }
      if (!is.null(x$y$iscrowd)) {
        x$y$iscrowd <- x$y$iscrowd[keep]
      }
    }
    boxes <- torch_cat(list(new_xy, angle_col), dim = -1L)
  }
  x$y$boxes <- boxes
  x$y$image_height <- as.integer(height)
  x$y$image_width <- as.integer(width)

  x
}

#' Horizontally flip a dataset item
#'
#' Flips the image inside a dataset item horizontally. For detection items,
#' bounding box x-coordinates are adjusted to remain correct after the flip.
#' For segmentation items, both the image and the masks are flipped.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#'
#' @return A dataset item of the same class with the image and target
#'   horizontally flipped.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"CAT"}))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_hflip(before)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_hflip <- function(x) {
  UseMethod("item_transform_hflip", x)
}

#' @export
item_transform_hflip.dataset <- function(x) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_hflip(item)
  }
  x
}

#' @export
item_transform_hflip.default <- function(x) {
  cli_abort(
    "{.fn item_transform_hflip} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To flip a raw image tensor, use {.fn transform_hflip} instead."
  )
}

#' @export
item_transform_hflip.image_with_bounding_box <- function(x) {
  orig_w <- as.numeric(x$x$shape[length(x$x$shape)])

  x$x <- transform_hflip(x$x)

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    x1 <- boxes[, 1]$clone()
    x3 <- boxes[, 3]$clone()
    boxes[, 1] <- orig_w - x3
    boxes[, 3] <- orig_w - x1
  }
  x$y$boxes <- boxes

  x
}

#' @export
item_transform_hflip.image_with_segmentation_mask <- function(x) {
  x$x <- transform_hflip(x$x)
  x$y$masks <- transform_hflip(x$y$masks)

  x
}

#' @export
item_transform_hflip.image_with_rotated_box <- function(x) {
  orig_w <- as.numeric(x$x$shape[length(x$x$shape)])

  x$x <- transform_hflip(x$x)

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    x1 <- boxes[, 1]$clone()
    x3 <- boxes[, 3]$clone()
    boxes[, 1] <- orig_w - x3
    boxes[, 3] <- orig_w - x1
    boxes[, 5] <- -boxes[, 5]
  }
  x$y$boxes <- boxes

  x
}

#' Vertically flip a dataset item
#'
#' Flips the image inside a dataset item vertically. For detection items,
#' bounding box y-coordinates are adjusted to remain correct after the flip.
#' For segmentation items, both the image and the masks are flipped.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#'
#' @return A dataset item of the same class with the image and target
#'   vertically flipped.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_vflip(before)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_vflip <- function(x) {
  UseMethod("item_transform_vflip", x)
}

#' @export
item_transform_vflip.dataset <- function(x) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_vflip(item)
  }
  x
}

#' @export
item_transform_vflip.default <- function(x) {
  cli_abort(
    "{.fn item_transform_vflip} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To flip a raw image tensor, use {.fn transform_vflip} instead."
  )
}

#' @export
item_transform_vflip.image_with_bounding_box <- function(x) {
  orig_h <- get_image_size(x$x)[2]

  x$x <- transform_vflip(x$x)

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    y1 <- boxes[, 2]$clone()
    y2 <- boxes[, 4]$clone()
    boxes[, 2] <- orig_h - y2
    boxes[, 4] <- orig_h - y1
  }
  x$y$boxes <- boxes

  x
}

#' @export
item_transform_vflip.image_with_segmentation_mask <- function(x) {
  x$x <- transform_vflip(x$x)
  x$y$masks <- transform_vflip(x$y$masks)

  x
}

#' @export
item_transform_vflip.image_with_rotated_box <- function(x) {
  orig_h <- get_image_size(x$x)[2]

  x$x <- transform_vflip(x$x)

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    y1 <- boxes[, 2]$clone()
    y2 <- boxes[, 4]$clone()
    boxes[, 2] <- orig_h - y2
    boxes[, 4] <- orig_h - y1
    boxes[, 5] <- -boxes[, 5]
  }
  x$y$boxes <- boxes

  x
}

# ---- item_transform_center_crop ----

#' Center crop a dataset item
#'
#' Center crops the image inside a dataset item. For detection items,
#' bounding box coordinates are adjusted to remain correct after cropping.
#' For segmentation items, both the image and the masks are center cropped.
#' If the image is smaller than the requested crop size along any edge,
#' the image is padded with zeros and then center cropped.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @param size Desired output size. If \code{size} is an integer vector of
#'   length 2 like \code{c(h, w)}, the output will be matched to this.
#'   If \code{size} is a bare integer, a square crop of \code{c(size, size)}
#'   is made.
#'
#' @return A dataset item of the same class with the image and target
#'   center cropped.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_center_crop(before, size = c(1500, 1500))
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' before_plot <- transform_resize(before_plot, c(1500, 1500))
#' after_plot <- transform_resize(after_plot, c(1500, 1500))
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_center_crop <- function(x, size) {
  UseMethod("item_transform_center_crop", x)
}

#' @export
item_transform_center_crop.default <- function(x, size) {
  cli_abort(
    "{.fn item_transform_center_crop} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To center crop a raw image tensor, use {.fn transform_center_crop} instead."
  )
}

#' @export
item_transform_center_crop.dataset <- function(x, size) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_center_crop(item, size = size)
  }
  x
}

#' @export
item_transform_center_crop.image_with_bounding_box <- function(x, size) {
  output_size <- as.integer(if (length(size) == 1) rep(size, 2) else size)
  crop_h <- output_size[1]
  crop_w <- output_size[2]

  img_size <- get_image_size(x$x)
  img_w <- img_size[1]
  img_h <- img_size[2]

  # crop offsets are relative to the (possibly padded) image, see
  # transform_center_crop
  crop_top <- as.integer((max(img_h, crop_h) - crop_h) / 2)
  crop_left <- as.integer((max(img_w, crop_w) - crop_w) / 2)

  if (crop_top == 0L) crop_top <- 1L
  if (crop_left == 0L) crop_left <- 1L

  x$x <- transform_center_crop(x$x, size)

  left_offset <- crop_left - 1L
  top_offset <- crop_top - 1L

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    boxes[, 1] <- torch_clamp(boxes[, 1] - left_offset, 0, crop_w)
    boxes[, 3] <- torch_clamp(boxes[, 3] - left_offset, 0, crop_w)
    boxes[, 2] <- torch_clamp(boxes[, 2] - top_offset, 0, crop_h)
    boxes[, 4] <- torch_clamp(boxes[, 4] - top_offset, 0, crop_h)

    keep <- as.logical((boxes[, 3] > boxes[, 1]) & (boxes[, 4] > boxes[, 2]))
    if (!all(keep)) {
      boxes <- boxes[keep, ]
      x$y$labels <- x$y$labels[keep]
      if (!is.null(x$y$area)) {
        x$y$area <- x$y$area[keep]
      }
      if (!is.null(x$y$iscrowd)) {
        x$y$iscrowd <- x$y$iscrowd[keep]
      }
    }
  }
  x$y$boxes <- boxes
  x$y$image_height <- crop_h
  x$y$image_width <- crop_w

  x
}

#' @export
item_transform_center_crop.image_with_segmentation_mask <- function(x, size) {
  output_size <- as.integer(if (length(size) == 1) rep(size, 2) else size)

  x$x <- transform_center_crop(x$x, size)
  x$y$masks <- transform_center_crop(x$y$masks, size)
  x$y$image_height <- output_size[1]
  x$y$image_width <- output_size[2]

  x
}

#' @export
item_transform_center_crop.image_with_rotated_box <- function(x, size) {
  output_size <- as.integer(if (length(size) == 1) rep(size, 2) else size)
  crop_h <- output_size[1]
  crop_w <- output_size[2]

  img_size <- get_image_size(x$x)
  img_w <- img_size[1]
  img_h <- img_size[2]

  # crop offsets are relative to the (possibly padded) image, see
  # transform_center_crop
  crop_top <- as.integer((max(img_h, crop_h) - crop_h) / 2)
  crop_left <- as.integer((max(img_w, crop_w) - crop_w) / 2)

  if (crop_top == 0L) crop_top <- 1L
  if (crop_left == 0L) crop_left <- 1L

  x$x <- transform_center_crop(x$x, size)

  left_offset <- crop_left - 1L
  top_offset <- crop_top - 1L

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    boxes[, 1] <- torch_clamp(boxes[, 1] - left_offset, 0, crop_w)
    boxes[, 3] <- torch_clamp(boxes[, 3] - left_offset, 0, crop_w)
    boxes[, 2] <- torch_clamp(boxes[, 2] - top_offset, 0, crop_h)
    boxes[, 4] <- torch_clamp(boxes[, 4] - top_offset, 0, crop_h)

    keep <- as.logical((boxes[, 3] > boxes[, 1]) & (boxes[, 4] > boxes[, 2]))
    if (!all(keep)) {
      boxes <- boxes[keep, ]
      x$y$labels <- x$y$labels[keep]
      if (!is.null(x$y$area)) {
        x$y$area <- x$y$area[keep]
      }
      if (!is.null(x$y$iscrowd)) {
        x$y$iscrowd <- x$y$iscrowd[keep]
      }
    }
  }
  x$y$boxes <- boxes
  x$y$image_height <- crop_h
  x$y$image_width <- crop_w

  x
}

#' Apply an affine transformation on a dataset item
#'
#' Applies an affine transformation (rotation, translation, scale and shear) to
#' the image inside a dataset item, keeping the image size unchanged. The target
#' is transformed with the same geometry so that it stays aligned with the
#' image.
#'
#' For detection items, the bounding boxes are transformed via
#' [target_transform_affine()] so that they follow the image, and the item is
#' returned as an \code{image_with_rotated_box} with boxes in xyxyr format. For
#' segmentation items, the masks are transformed alongside the image using
#' nearest-neighbour sampling.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box},
#'   \code{image_with_rotated_box} or \code{image_with_segmentation_mask} object
#'   containing an image tensor and associated target data.
#' @inheritParams transform_affine
#'
#' @return A dataset item with the image and target transformed. Detection items
#'   are returned as \code{image_with_rotated_box}; segmentation items keep their
#'   class.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"CAT"},
#'                                   image_height = img$shape[2], image_width = img$shape[3]))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_affine(before, angle = 20, translate = c(50, 30),
#'                                scale = 1.1, shear = 10)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)
#' tensor_image_browse(before_plot)
#' tensor_image_browse(after_plot)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_affine <- function(x, angle = 0, translate = c(0, 0), scale = 1,
                                  shear = 0, interpolation = 0, fill = NULL,
                                  center = NULL) {
  UseMethod("item_transform_affine", x)
}

#' @export
item_transform_affine.default <- function(x, angle = 0, translate = c(0, 0),
                                          scale = 1, shear = 0, interpolation = 0,
                                          fill = NULL, center = NULL) {
  cli_abort(
    "{.fn item_transform_affine} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To transform a raw image tensor, use {.fn transform_affine} instead."
  )
}

#' @export
item_transform_affine.image_with_bounding_box <- function(x, angle = 0,
                                                          translate = c(0, 0),
                                                          scale = 1, shear = 0,
                                                          interpolation = 0,
                                                          fill = NULL,
                                                          center = NULL) {
  x$x <- transform_affine(x$x, angle = angle, translate = translate,
                          scale = scale, shear = shear,
                          interpolation = interpolation, fill = fill,
                          center = center)
  x$y <- target_transform_affine(x$y, angle = angle, translate = translate,
                                 scale = scale, shear = shear, center = center)
  class(x) <- c("image_with_rotated_box", "list")
  x
}

#' @export
item_transform_affine.image_with_rotated_box <- function(x, angle = 0,
                                                         translate = c(0, 0),
                                                         scale = 1, shear = 0,
                                                         interpolation = 0,
                                                         fill = NULL,
                                                         center = NULL) {
  item_transform_affine.image_with_bounding_box(
    x, angle = angle, translate = translate, scale = scale, shear = shear,
    interpolation = interpolation, fill = fill, center = center
  )
}

#' @export
item_transform_affine.image_with_segmentation_mask <- function(x, angle = 0,
                                                               translate = c(0, 0),
                                                               scale = 1, shear = 0,
                                                               interpolation = 0,
                                                               fill = NULL,
                                                               center = NULL) {
  x$x <- transform_affine(x$x, angle = angle, translate = translate,
                          scale = scale, shear = shear,
                          interpolation = interpolation, fill = fill,
                          center = center)

  masks <- x$y$masks
  if (!is.null(masks) && masks$ndim >= 3) {
    dtype <- masks$dtype
    masks <- transform_affine(masks, angle = angle, translate = translate,
                              scale = scale, shear = shear,
                              interpolation = 0, fill = fill, center = center)
    x$y$masks <- masks$to(dtype = dtype)
  }
  x
}

#' @export
item_transform_affine.dataset <- function(x, angle = 0, translate = c(0, 0),
                                          scale = 1, shear = 0, interpolation = 0,
                                          fill = NULL, center = NULL) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_affine(item, angle = angle, translate = translate,
                          scale = scale, shear = shear,
                          interpolation = interpolation, fill = fill,
                          center = center)
  }
  x
}

#' Pad a dataset item
#'
#' Pads the image inside a dataset item on all sides with the given "pad"
#' value. For detection items, bounding box coordinates are shifted to remain
#' correct after padding. For segmentation items, both the image and the masks
#' are padded.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @inheritParams transform_pad
#'
#' @return A dataset item of the same class with the padded image and adjusted
#'   target.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_pad(before, padding = 100)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' before_plot <- transform_resize(before_plot, c(2000, 3000))
#' after_plot <- transform_resize(after_plot, c(2000, 3000))
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_pad <- function(x, padding, fill = 0, padding_mode = "constant") {
  UseMethod("item_transform_pad", x)
}

#' @export
item_transform_pad.dataset <- function(x, padding, fill = 0, padding_mode = "constant") {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_pad(item, padding = padding, fill = fill, padding_mode = padding_mode)
  }
  x
}

#' @export
item_transform_pad.default <- function(x, padding, fill = 0, padding_mode = "constant") {
  cli_abort(
    "{.fn item_transform_pad} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To pad a raw image tensor, use {.fn transform_pad} instead."
  )
}

#' @export
item_transform_pad.image_with_bounding_box <- function(x, padding, fill = 0, padding_mode = "constant") {
  x$x <- transform_pad(x$x, padding, fill, padding_mode)

  if (length(padding) == 1) {
    pad_left <- pad_top <- as.numeric(padding)
  } else if (length(padding) == 2) {
    pad_left <- as.numeric(padding[1])
    pad_top <- as.numeric(padding[2])
  } else {
    pad_left <- as.numeric(padding[1])
    pad_top <- as.numeric(padding[3])
  }

  img_size <- get_image_size(x$x)
  new_w <- img_size[1]
  new_h <- img_size[2]

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    boxes[, 1] <- torch_clamp(boxes[, 1] + pad_left, 0, new_w)
    boxes[, 2] <- torch_clamp(boxes[, 2] + pad_top, 0, new_h)
    boxes[, 3] <- torch_clamp(boxes[, 3] + pad_left, 0, new_w)
    boxes[, 4] <- torch_clamp(boxes[, 4] + pad_top, 0, new_h)

    keep <- as.logical((boxes[, 3] > boxes[, 1]) & (boxes[, 4] > boxes[, 2]))
    if (!all(keep)) {
      boxes <- boxes[keep, ]
      x$y$labels <- x$y$labels[keep]
      if (!is.null(x$y$area)) {
        x$y$area <- x$y$area[keep]
      }
      if (!is.null(x$y$iscrowd)) {
        x$y$iscrowd <- x$y$iscrowd[keep]
      }
    }
  }
  x$y$boxes <- boxes
  x$y$image_height <- new_h
  x$y$image_width <- new_w

  x
}

#' @export
item_transform_pad.image_with_segmentation_mask <- function(x, padding, fill = 0, padding_mode = "constant") {
  x$x <- transform_pad(x$x, padding, fill, padding_mode)
  x$y$masks <- transform_pad(x$y$masks, padding, fill = 0, padding_mode = padding_mode)

  img_size <- get_image_size(x$x)
  x$y$image_height <- img_size[2]
  x$y$image_width <- img_size[1]

  x
}

#' @export
item_transform_pad.image_with_rotated_box <- function(x, padding, fill = 0, padding_mode = "constant") {
  x$x <- transform_pad(x$x, padding, fill, padding_mode)

  if (length(padding) == 1) {
    pad_left <- pad_top <- as.numeric(padding)
  } else if (length(padding) == 2) {
    pad_left <- as.numeric(padding[1])
    pad_top <- as.numeric(padding[2])
  } else {
    pad_left <- as.numeric(padding[1])
    pad_top <- as.numeric(padding[3])
  }

  img_size <- get_image_size(x$x)
  new_w <- img_size[1]
  new_h <- img_size[2]

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    boxes[, 1] <- torch_clamp(boxes[, 1] + pad_left, 0, new_w)
    boxes[, 2] <- torch_clamp(boxes[, 2] + pad_top, 0, new_h)
    boxes[, 3] <- torch_clamp(boxes[, 3] + pad_left, 0, new_w)
    boxes[, 4] <- torch_clamp(boxes[, 4] + pad_top, 0, new_h)

    keep <- as.logical((boxes[, 3] > boxes[, 1]) & (boxes[, 4] > boxes[, 2]))
    if (!all(keep)) {
      boxes <- boxes[keep, ]
      x$y$labels <- x$y$labels[keep]
      if (!is.null(x$y$area)) {
        x$y$area <- x$y$area[keep]
      }
      if (!is.null(x$y$iscrowd)) {
        x$y$iscrowd <- x$y$iscrowd[keep]
      }
    }
  }
  x$y$boxes <- boxes
  x$y$image_height <- new_h
  x$y$image_width <- new_w

  x
}

# ---- item_transform_perspective ----

#' Apply a perspective transform to a dataset item
#'
#' Applies a perspective transform to the image inside a dataset item, as
#' defined by \code{\link{transform_perspective}}. For detection items,
#' bounding box coordinates are transformed accordingly and clamped to the
#' image canvas, boxes that end up completely outside the image are removed.
#' For segmentation items, both the image and the masks are transformed,
#' using nearest neighbour interpolation for the masks.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} or
#'   \code{image_with_segmentation_mask} object containing an image tensor
#'   and associated target data.
#' @inheritParams transform_perspective
#'
#' @return A dataset item of the same class with the image and target
#'   perspective transformed.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/b/b6/Felis_catus-cat_on_snow.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(600, 200, 2880, 1860), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = "cat"))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' startpoints <- list(c(0, 0), c(2880, 0), c(2880, 1860), c(0, 1860))
#' endpoints <- list(c(100, 50), c(2700, 0), c(2800, 1800), c(80, 1900))
#'
#' after <- item_transform_perspective(before, startpoints, endpoints)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)$to(torch_float())$div(255)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)$to(torch_float())$div(255)
#'
#' grid <- vision_make_grid(torch_stack(list(before_plot, after_plot)), scale = TRUE)
#' tensor_image_browse(grid)
#' }
#'
#' @family item_unitary_transforms
#'
#' @export
item_transform_perspective <- function(x, startpoints, endpoints, interpolation = 2,
                                       fill = NULL) {
  UseMethod("item_transform_perspective", x)
}

#' @export
item_transform_perspective.default <- function(x, startpoints, endpoints, interpolation = 2,
                                               fill = NULL) {
  cli_abort(
    "{.fn item_transform_perspective} requires a dataset item (a list with {.var x} and {.var y} fields), not {.obj_type_friendly {x}}.
    To transform a raw image tensor, use {.fn transform_perspective} instead."
  )
}

#' @export
item_transform_perspective.dataset <- function(x, startpoints, endpoints, interpolation = 2,
                                               fill = NULL) {
  original_getitem <- x$.getitem
  unlockBinding(".getitem", as.environment(x))
  x$.getitem <- function(index) {
    item <- original_getitem(index)
    item_transform_perspective(item,
      startpoints = startpoints,
      endpoints = endpoints,
      interpolation = interpolation,
      fill = fill
    )
  }
  x
}

#' @export
item_transform_perspective.image_with_bounding_box <- function(x, startpoints, endpoints,
                                                               interpolation = 2, fill = NULL) {
  x$x <- transform_perspective(
    x$x,
    startpoints = startpoints,
    endpoints = endpoints,
    interpolation = interpolation,
    fill = fill
  )

  boxes <- x$y$boxes$clone()
  if (boxes$size(1) > 0) {
    boxes <- perspective_boxes(boxes, startpoints = startpoints, endpoints = endpoints)

    img_size <- get_image_size(x$x)
    img_w <- img_size[1]
    img_h <- img_size[2]

    boxes[, 1] <- torch_clamp(boxes[, 1], 0, img_w)
    boxes[, 3] <- torch_clamp(boxes[, 3], 0, img_w)
    boxes[, 2] <- torch_clamp(boxes[, 2], 0, img_h)
    boxes[, 4] <- torch_clamp(boxes[, 4], 0, img_h)

    keep <- as.logical((boxes[, 3] > boxes[, 1]) & (boxes[, 4] > boxes[, 2]))
    if (!all(keep)) {
      boxes <- boxes[keep, ]
      x$y$labels <- x$y$labels[keep]
      if (!is.null(x$y$area)) {
        x$y$area <- x$y$area[keep]
      }
      if (!is.null(x$y$iscrowd)) {
        x$y$iscrowd <- x$y$iscrowd[keep]
      }
    }
  }
  x$y$boxes <- boxes

  x
}

#' @export
item_transform_perspective.image_with_segmentation_mask <- function(x, startpoints, endpoints,
                                                                    interpolation = 2, fill = NULL) {
  x$x <- transform_perspective(
    x$x,
    startpoints = startpoints,
    endpoints = endpoints,
    interpolation = interpolation,
    fill = fill
  )
  x$y$masks <- transform_perspective(
    x$y$masks,
    startpoints = startpoints,
    endpoints = endpoints,
    interpolation = 0, # nearest is used for masks
    fill = NULL
  )

  x
}

