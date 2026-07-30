#' Rotate dataset item
#'
#' Rotates a dataset item by a given angle around its center.
#' The canvas is expanded so that the entire rotated image is visible with no
#' cropping. Empty regions are filled with given fill color.
#'
#' The bounding boxes (if present) are shifted to account for the expanded
#' canvas and converted to rotated format via
#' \code{\link{target_transform_rotate}}.
#'
#' @param x A dataset item, typically an \code{image_with_bounding_box} object
#'   containing an image tensor and associated target data (boxes, labels).
#' @inheritParams transform_rotate
#'
#' @return An \code{image_with_rotated_box} object with the rotated image and
#'   converted boxes in xyxyr format.
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/6/66/The_Leaning_Tower_of_Pisa_SB.jpeg"
#'
#' img <- base_loader(url) |>
#'   transform_to_tensor()
#'
#' boxes <- torch_tensor(matrix(c(720, 620, 1900, 3700), ncol = 4), dtype = torch_float32())
#'
#' before <- list(x = img, y = list(boxes = boxes, labels = {"Leaning Tower of Pisa"}))
#' class(before) <- c("image_with_bounding_box", "list")
#'
#' after <- item_transform_rotate(before, angle = 30)
#'
#' before_plot <- draw_bounding_boxes(before, colors = "blue", width = 10)
#' after_plot <- draw_bounding_boxes(after, colors = "red", width = 10)
#' tensor_image_browse(before_plot)
#' tensor_image_browse(after_plot)
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
  shifts <- (new_spatial - orig_spatial) / 2
  dx <- shifts[2]  # shift along second spatial dim (width in torch convention)
  dy <- shifts[1]  # shift along first spatial dim (height in torch convention)

  # Shift boxes safely
  x1 <- x$y$boxes[, 1]
  y1 <- x$y$boxes[, 2]
  x2 <- x$y$boxes[, 3]
  y2 <- x$y$boxes[, 4]
  shifted_boxes <- torch_stack(list(x1 + dx, y1 + dy, x2 + dx, y2 + dy), dim = -1L)

  x$x <- rotated_img
  x$y$boxes <- box_xyxy_to_xyxyr(shifted_boxes, angle = angle)
  x$y$image_height <- new_spatial[1]  # First spatial dim in torch = height
  x$y$image_width <- new_spatial[2]

  class(x) <- c("image_with_rotated_box", "list")
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
  shifts <- (new_spatial - orig_spatial) / 2
  dx <- shifts[2]
  dy <- shifts[1]

  # Shift existing xyxyr boxes
  x1 <- x$y$boxes[, 1]
  y1 <- x$y$boxes[, 2]
  x2 <- x$y$boxes[, 3]
  y2 <- x$y$boxes[, 4]
  angle_col <- x$y$boxes[, 5, drop = FALSE]

  shifted_xy <- torch_stack(list(x1 + dx, y1 + dy, x2 + dx, y2 + dy), dim = -1L)
  shifted_boxes <- torch_cat(list(shifted_xy, angle_col), dim = -1L)

  x$x <- rotated_img
  x$y$boxes <- box_xyxy_to_xyxyr(shifted_boxes, angle = angle)
  x$y$image_height <- new_spatial[1]
  x$y$image_width  <- new_spatial[2]
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
#' after <- item_transform_center_crop(before, size = c(300, 300))
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
  center_crop_item(x, size)
}

#' @export
item_transform_center_crop.image_with_segmentation_mask <- function(x, size) {
  center_crop_item(x, size)
}

#' @export
item_transform_center_crop.image_with_rotated_box <- function(x, size) {
  center_crop_item(x, size)
}

center_crop_item <- function(x, size) {
  img_h <- as.numeric(x$x$shape[2])
  img_w <- as.numeric(x$x$shape[3])

  output_size <- size
  if (length(size) == 1) {
    output_size <- rep(size, 2)
  }
  output_size <- as.integer(output_size)
  crop_h <- output_size[1]
  crop_w <- output_size[2]

  # Handle padding if crop larger than image
  pad_left <- 0L
  pad_right <- 0L
  pad_top <- 0L
  pad_bottom <- 0L
  if (crop_w > img_w) {
    pad_left <- floor((crop_w - img_w) / 2)
    pad_right <- ceiling((crop_w - img_w) / 2)
  }
  if (crop_h > img_h) {
    pad_top <- floor((crop_h - img_h) / 2)
    pad_bottom <- ceiling((crop_h - img_h) / 2)
  }

  if (pad_left > 0L || pad_right > 0L || pad_top > 0L || pad_bottom > 0L) {
    padding <- c(pad_left, pad_right, pad_top, pad_bottom)
    x$x <- transform_pad(x$x, padding, fill = 0)
    if (!is.null(x$y$masks)) {
      x$y$masks <- transform_pad(x$y$masks, padding, fill = 0)
    }
    img_h <- img_h + pad_top + pad_bottom
    img_w <- img_w + pad_left + pad_right

    if (crop_w == img_w && crop_h == img_h) {
      x$y$image_height <- crop_h
      x$y$image_width <- crop_w
      return(x)
    }
  }

  crop_top <- as.integer((img_h - crop_h) / 2)
  crop_left <- as.integer((img_w - crop_w) / 2)

  if (crop_top == 0L) crop_top <- 1L
  if (crop_left == 0L) crop_left <- 1L

  # Crop image
  x$x <- transform_crop(x$x, crop_top, crop_left, crop_h, crop_w)

  # Crop masks if present
  if (!is.null(x$y$masks)) {
    x$y$masks <- transform_crop(x$y$masks, crop_top, crop_left, crop_h, crop_w)
  }

  # Adjust boxes if present
  if (!is.null(x$y$boxes)) {
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
  }

  x$y$image_height <- crop_h
  x$y$image_width <- crop_w

  x
}
