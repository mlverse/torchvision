is_tensor_image <- function(x) {
  x$ndim() >= 2
}

check_img <- function(x) {
  if (is_tensor_image(x))
    type_error("tensor is not a torch image.")
}

blend <- function(img1, img2, ratio) {

  if (img1$is_floating_point())
    bound <- 1
  else
    bound <- 255

  (ratio * img1 + (1 - ratio) * img2)$clamp(0, bound)$to(img1$dtype())
}

#' Vertically flip the given the Image Tensor.
#'
#' @param img (Tensor): Image Tensor to be flipped in the form [C, H, W].
#'
#' @return Vertically flipped image Tensor.
#'
#' @export
tft_vflip <- function(img) {
  check_img(img)
  img$flip(-2)
}

#' Horizontally flip the given the Image Tensor.
#'
#' @param img (Tensor): Image Tensor to be flipped in the form [C, H, W].
#'
#' @return Horizontally flipped image Tensor.
#'
#' @export
tft_hflip <- function(img) {
  check_img(img)
  img$flip(-1)
}

#' Crop the given Image Tensor.
#'
#' @param img (Tensor): Image to be cropped in the form [..., H, W]. (0,0) denotes the top left corner of the image.
#' @param top (int): Vertical component of the top left corner of the crop box.
#' @param left (int): Horizontal component of the top left corner of the crop box.
#' @param height (int): Height of the crop box.
#' @param width (int): Width of the crop box.
#'
#' @return Cropped image.
tft_crop <- function(img, top, left, height, width) {
  check_img(img)

  img[.., top:(top + height), left:(left + width)]
}

#' Convert the given RGB Image Tensor to Grayscale.
#' For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
#' is L = R * 0.2989 + G * 0.5870 + B * 0.1140
#'
#' @param img (Tensor): Image to be converted to Grayscale in the
#' form `[C, H, W]`.
#'
#' @return Grayscale image.
#'
#' @export
tft_rgb_to_grayscale <- function(img) {
  check_img(img)
  (0.2989 * img[1] + 0.5870 * img[2] + 0.1140 * img[3])$to(img$dtype())
}

#' Adjust brightness of an RGB image.
#'
#' @param img (Tensor): Image to be adjusted.
#' @param brightness_factor (float):  How much to adjust the brightness. Can be
#'   any non negative number. 0 gives a black image, 1 gives the
#'   original image while 2 increases the brightness by a factor of 2.
#'
#' @return Brightness adjusted image.
#'
#' @export
tft_adjust_brightness <- function(img, brightness_factor) {
  if (brightness_factor < 0)
    value_error("brightness factor is negative")

  check_img(img)

  blend(img, torch::torch_zeros_like(img), brightness_factor)
}

#' Adjust contrast of an RGB image.
#'
#' @param img (Tensor): Image to be adjusted.
#' @param contrast_factor (float): How much to adjust the contrast. Can be any
#'   non negative number. 0 gives a solid gray image, 1 gives the
#'   original image while 2 increases the contrast by a factor of 2.
#'
#' @return Contrast adjusted image.
#'
#' @export
tft_adjust_contrast <- function(img, contrast_factor) {

  if (contrast_factor < 0)
    value_error("contrast must be positive")

  check_img(img)

  mean <- torch::torch_mean(tft_rgb_to_grayscale(img)$to(torch_float()))

  blend(img, mean, contrast_factor)
}
