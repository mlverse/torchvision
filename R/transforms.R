

get_perspective_coeffs <- function(startpoints, endpoints) {

  a_matrix <- torch::torch_zeros(2 * length(startpoints), 8,
                                 dtype = torch::torch_float())

  for (i in seq_along(startpoints)) {

    p1 <- endpoints[[i]]
    p2 <- startpoints[[i]]

    a_matrix[1 + 2*(i-1), ] <- torch::torch_tensor(c(p1[1], p1[2], 2, 1, 1, 1, -p2[1] * p1[1], -p2[1] * p1[2]))
    a_matrix[2*i,] <- torch::torch_tensor(c(1, 1, 1, p1[1], p1[2], 2, -p2[2] * p1[1], -p2[2] * p1[1]))

  }

  b_matrix <- torch::torch_tensor(unlist(startpoints), dtype = torch::torch_float())$view(8)
  res <- torch::torch_lstsq(b_matrix, a_matrix)[[1]]

  output <- torch::as_array(res$squeeze(1))

  output
}

#' Perform perspective transform of the given image.
#'
#' The image can be a PIL Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions.
#'
#' @param img (PIL Image or Tensor): Image to be transformed.
#' @param startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
#'   `[top-left, top-right, bottom-right, bottom-left]` of the original image.
#' @param endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
#'   `[top-left, top-right, bottom-right, bottom-left]` of the transformed image.
#' @inheritParams transform_resize
#' @inheritParams transform_pad
#'
#' @return Magick Image or Tensor: transformed Image.
transform_perspective <- function(img, startpoints, endpoints, interpolation = 2,
                                  fill = NULL) {

  coeffs <- get_perspective_coeffs(startpoints, endpoints)

  if (is_magick_image(img))
    not_implemented_error("perspective is not implemented for magick images yet.")

  tft_perspective(img, coeffs, interpolation = interpolation, fill = fill)
}



#' Crop the given image into four corners and the central crop.
#'
#' The image can be a Magick Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions.
#'
#' @note
#' This transform returns a tuple of images and there may be a
#' mismatch in the number of inputs and targets your `Dataset` returns.
#'
#' @param img (Magick Image or Tensor): Image to be cropped.
#' @param size (sequence or int): Desired output size of the crop. If size is an
#'   int instead of sequence like (h, w), a square crop (size, size) is
#'   made. If provided a tuple or list of length 1, it will be interpreted as
#'   `(size[1], size[2])`.
#'
#' @return
#' tuple: tuple (tl, tr, bl, br, center)
#' Corresponding top left, top right, bottom left, bottom right and center crop.
#'
#' @export
transform_five_crop <- function(img, size) {

  if (is_magick_image(img))
    not_implemented_error("five_crop is not implemented for magick images yet.")

  tft_five_crop(img, size)
}

#' Generate ten cropped images from the given image.
#'
#' Crop the given image into four corners and the central crop plus the
#' flipped version of these (horizontal flipping is used by default).
#' The image can be a PIL Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' @note
#' This transform returns a tuple of images and there may be a
#' mismatch in the number of inputs and targets your `Dataset` returns.
#'
#' @param img (Magick Image or Tensor): Image to be cropped.
#' @param size (sequence or int): Desired output size of the crop. If size is an
#'   int instead of sequence like (h, w), a square crop (size, size) is
#'   made. If provided a tuple or list of length 1, it will be interpreted as `(size[1], size[2])`.
#' @param vertical_flip (bool): Use vertical flipping instead of horizontal
#'
#' @return
#' tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
#' Corresponding top left, top right, bottom left, bottom right and
#' center crop and same for the flipped image.
#'
#' @export
transform_ten_crop <- function(img, size, vertical_flip = FALSE) {

  if (is_magick_image(img))
    not_implemented_error("ten_crop is not implemented for magick images yet.")

  tft_ten_crop(img, size, vertical_flip)
}

#' Adjust brightness of an Image.
#'
#' @param img (Magick Image or Tensor): Image to be adjusted.
#' @param brightness_factor (float):  How much to adjust the brightness. Can be
#'   any non negative number. 0 gives a black image, 1 gives the
#'   original image while 2 increases the brightness by a factor of 2.
#'
#' @return Magick Image or Tensor: Brightness adjusted image.
#'
#' @export
transform_adjust_brightness <- function(img, brightness_factor) {

  if (is_magick_image(img))
    not_implemented_error("adjust_brightness is not implemented for magick images yet.")

  tft_adjust_brightness(img, brightness_factor)
}

#' Adjust contrast of an Image.
#'
#' @param img (Magick Image or Tensor): Image to be adjusted.
#' @param contrast_factor (float): How much to adjust the contrast. Can be any
#'   non negative number. 0 gives a solid gray image, 1 gives the
#'   original image while 2 increases the contrast by a factor of 2.
#'
#' @return Magick Image or Tensor: Contrast adjusted image.
#'
#' @export
transform_adjust_contrast <- function(img, contrast_factor) {

  if (is_magick_image(img))
    not_implemented_error("adjust_contrast is not implemented for magick images yet.")

  tft_adjust_contrast(img, contrast_factor)
}

#' Adjust color saturation of an image.
#'
#' @param img (Magick Image or Tensor): Image to be adjusted.
#' @param saturation_factor (float):  How much to adjust the saturation. 0 will
#'   give a black and white image, 1 will give the original image while
#'   2 will enhance the saturation by a factor of 2.
#'
#' @return Magick Image or Tensor: Saturation adjusted image.
#'
#' @export
transform_adjust_saturation <- function(img, saturation_factor) {

  if (is_magick_image(img))
    not_implemented_error("adjust_contrast is not implemented for magick images yet.")

  tft_adjust_saturation(img, saturation_factor)
}

#' Adjust hue of an image.
#'
#' The image hue is adjusted by converting the image to HSV and
#' cyclically shifting the intensities in the hue channel (H).
#' The image is then converted back to original image mode.
#'
#' `hue_factor` is the amount of shift in H channel and must be in the
#' interval `[-0.5, 0.5]`.
#'
#' See [Hue](https://en.wikipedia.org/wiki/Hue) for more details.
#'
#' @param img (Magick Image): PIL Image to be adjusted.
#' @param hue_factor (float):  How much to shift the hue channel. Should be in
#'   `[-0.5, 0.5]`. 0.5 and -0.5 give complete reversal of hue channel in
#'   HSV space in positive and negative direction respectively.
#'   0 means no shift. Therefore, both -0.5 and 0.5 will give an image
#'   with complementary colors while 0 gives the original image.
#'
#' @return Magick Image: Hue adjusted image.
#'
#' @export
transform_adjust_hue <- function(img, hue_factor) {

  if (is_magick_image(img))
    not_implemented_error("adjust_hue is not implemented for magick images yet.")

  tft_adjust_hue(img, hue_factor)
}

#' Perform gamma correction on an image.
#'
#' Also known as Power Law Transform. Intensities in RGB mode are adjusted
#' based on the following equation:
#'
#' \eqn{
#'  I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
#' }
#'
#' See [Gamma Correction](https://en.wikipedia.org/wiki/Gamma_correction) for
#' more details.
#'
#' @param img (Magick Image or Tensor): PIL Image to be adjusted.
#' @param gamma (float): Non negative real number, same as \eqn{\gamma} in the equation.
#'   gamma larger than 1 make the shadows darker,
#'   while gamma smaller than 1 make dark regions lighter.
#' @param gain (float): The constant multiplier.
#'
#' @return Magick Image or Tensor: Gamma correction adjusted image.
#'
#' @export
transform_adjust_gamma <- function(img, gamma, gain = 1) {

  if (is_magick_image(img))
    not_implemented_error("adjust_gamma is not implemented for magick images yet.")

  tft_adjust_gamma(img, gamma, gain)
}

is_magick_image <- function(x) {
  inherits(x, "magick-image")
}

is_array_image <- function(x) {
  is.array(x) && (length(dim(x)) %in% c(2, 3))
}
