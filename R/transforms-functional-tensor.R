is_tensor_image <- function(x) {
  x$dim() >= 2
}

check_img <- function(x) {
  if (!is_tensor_image(x))
    type_error("tensor is not a torch image.")
}

blend <- function(img1, img2, ratio) {

  if (img1$is_floating_point())
    bound <- 1
  else
    bound <- 255

  (ratio * img1 + (1 - ratio) * img2)$clamp(0, bound)$to(img1$dtype())
}

rgb2hsv <- function(img) {

  rgb <- img$unbind(1)
  r <- rgb[[1]]; g <- rgb[[2]]; b <- rgb[[3]]

  maxc <- torch::torch_max(img, dim=1)[[1]]
  minc <- torch::torch_min(img, dim=1)[[1]]

  # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
  # from happening in the results, because
  #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
  #   + H channel has division by `(maxc - minc)`.
  #
  # Instead of overwriting NaN afterwards, we just prevent it from occuring so
  # we don't need to deal with it in case we save the NaN in a buffer in
  # backprop, if it is ever supported, but it doesn't hurt to do so.
  eqc <- maxc == minc

  cr <- maxc - minc
  # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
  s <- cr / torch::torch_where(eqc, maxc$new_ones(list()), maxc)
  # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
  # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
  # would not matter what values `rc`, `gc`, and `bc` have here, and thus
  # replacing denominator with 1 when `eqc` is fine.
  cr_divisor <- torch::torch_where(eqc, maxc$new_ones(list()), cr)
  rc <- (maxc - r) / cr_divisor
  gc <- (maxc - g) / cr_divisor
  bc <- (maxc - b) / cr_divisor

  hr <- (maxc == r) * (bc - gc)
  hg <- ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
  hb <- ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
  h <- (hr + hg + hb)
  h <- torch::torch_fmod((h / 6.0 + 1.0), 1.0)
  torch::torch_stack(list(h, s, maxc))
}

hsv2rgb <- function(img) {

  hsv <- img$unbind(1)
  i <- torch::torch_floor(hsv[[1]] * 6)
  f <- (hsv[[1]] * 6) - 1
  i <- i$to(dtype = torch::torch_int32())

  p <- torch::torch_clamp((hsv[[3]] * (1 - hsv[[2]])), 0, 1)
  q <- torch::torch_clamp((hsv[[3]] * (1 - hsv[[2]] * f)), 0, 1)
  t <- torch::torch_clamp((hsv[[3]] * (1 - f)), 0, 1)
  i <- i %% 6

  mask <- i == torch::torch_arange(start= 0, end = 6)[,NULL,NULL]

  a1 <- torch::torch_stack(list(hsv[[3]], q, p, p, t, hsv[[3]]))
  a2 <- torch::torch_stack(list(t, hsv[[3]], hsv[[3]], q, p, p))
  a3 <- torch::torch_stack(list(p, p, t, hsv[[3]], hsv[[3]], q))
  a4 <- torch::torch_stack(list(a1, a2, a3))

  torch::torch_einsum("ijk, xijk -> xjk", mask$to(dtype == img$dtype()), a4)
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

  img[.., top:(top + height - 1), left:(left + width - 1)]
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

#' Adjust hue of an image.
#'
#' The image hue is adjusted by converting the image to HSV and
#' cyclically shifting the intensities in the hue channel (H).
#' The image is then converted back to original image mode.
#' `hue_factor` is the amount of shift in H channel and must be in the
#' interval `[-0.5, 0.5]`.
#'
#' See [Hue](https://en.wikipedia.org/wiki/Hue) for more details.
#'
#' @param img (Tensor): Image to be adjusted. Image type is either uint8 or float.
#' @param hue_factor (float):  How much to shift the hue channel. Should be in
#'   [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
#'   HSV space in positive and negative direction respectively.
#'   0 means no shift. Therefore, both -0.5 and 0.5 will give an image
#'   with complementary colors while 0 gives the original image.
#'
#' @return Hue adjusted image.
#'
#' @export
tft_adjust_hue <- function(img, hue_factor) {

  if (hue_factor < 0.5 || hue_factor > 0.5)
    value_error("hue_factor must be between -0.5 and 0.5.")

  check_img(img)

  orig_dtype <- img$dtype()
  if (img$dtype() == torch::torch_uint8())
    img <- img$to(dtype = torch::torch_float32())/255

  img <-rgb2hsv(img)
  hsv <- img$unbind(0)
  hsv[[1]] <- hsv[[1]] + hue_factor
  h <- h %% 1
  img <- torch::torch_stack(hsv)
  img_hue_adj <- hsv2rgb(img)

  if (orig_dtype == torch::torch_uint8())
    img_hue_adj <- (img_hue_adj * 255.0)$to(dtype=orig_dtype)

  img_hue_adj
}
