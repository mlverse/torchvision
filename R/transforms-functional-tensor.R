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
  s <- cr / torch::torch_where(eqc, maxc$new_full(list(), 1), maxc)
  # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
  # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
  # would not matter what values `rc`, `gc`, and `bc` have here, and thus
  # replacing denominator with 1 when `eqc` is fine.
  cr_divisor <- torch::torch_where(eqc, maxc$new_full(list(), 1), cr)
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

  torch::torch_einsum("ijk, xijk -> xjk", list(mask$to(dtype = img$dtype()), a4))
}

#' Vertically flip the given the Image Tensor.
#'
#' @param img (Tensor): Image Tensor to be flipped in the form `[C, H, W]`.
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
#' @param img (Tensor): Image Tensor to be flipped in the form `[C, H, W]`.
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
#' @param img (Tensor): Image to be cropped in the form `[..., H, W]`. (0,0) denotes the top left corner of the image.
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

  mean <- torch::torch_mean(tft_rgb_to_grayscale(img)$to(torch::torch_float()))

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
#'   `[-0.5, 0.5]`. 0.5 and -0.5 give complete reversal of hue channel in
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
  hsv <- img$unbind(1)
  hsv[[1]] <- hsv[[1]] + hue_factor
  hsv[[1]] <- hsv[[1]] %% 1
  img <- torch::torch_stack(hsv)
  img_hue_adj <- hsv2rgb(img)

  if (orig_dtype == torch::torch_uint8())
    img_hue_adj <- (img_hue_adj * 255.0)$to(dtype=orig_dtype)

  img_hue_adj
}

#' Adjust color saturation of an RGB image.
#'
#' @param img (Tensor): Image to be adjusted.
#' @param saturation_factor (float):  How much to adjust the saturation. Can be any
#'   non negative number. 0 gives a black and white image, 1 gives the
#'   original image while 2 enhances the saturation by a factor of 2.
#'
#' @return Tensor: Saturation adjusted image.
#'
#' @export
tft_adjust_saturation <- function(img, saturation_factor) {

  if (saturation_factor < 0)
    value_error("saturation factor must be positive.")

  check_img(img)


  blend(img, tft_rgb_to_grayscale(img), saturation_factor)
}

#' Adjust gamma of an RGB image.
#'
#' Also known as Power Law Transform. Intensities in RGB mode are adjusted
#' based on the following equation:
#'
#' \deqn{
#'   I_{\mbox{out}} = 255 \times \mbox{gain} \times \left(\frac{I_{\mbox{in}}}{255}\right)^{\gamma}
#' }
#'
#' See [Gamma Correction](https://en.wikipedia.org/wiki/Gamma_correction)
#' for more details.
#'
#' @param img (Tensor): Tensor of RBG values to be adjusted.
#' @param gamma (float): Non negative real number, same as \eqn{\gamma} in the equation.
#'   gamma larger than 1 make the shadows darker,
#'   while gamma smaller than 1 make dark regions lighter.
#' @param gain (float): The constant multiplier.
#'
#' @export
tft_adjust_gamma <- function(img, gamma, gain = 1) {

  check_img(img)

  if (gamma < 0)
    value_error("gamma must be non-negative")

  result <- img
  dtype <- img$dtype()

  if (!torch::torch_is_floating_point(img))
    result <- result/255.0

  result <- (gain * result ^ gamma)$clamp(0, 1)

  if (!result$dtype() == dtype) {
    eps <- 1e-3
    result <- (255 + 1.0 - eps) * result
  }

  result <- result$to(dtype = dtype)
  result
}

#' Crop the Image Tensor and resize it to desired size.
#'
#' @param img (Tensor): Image to be cropped.
#' @param output_size (sequence or int): (height, width) of the crop box. If int,
#'   it is used for both directions
#'
#' @return Tensor: Cropped image.
#'
#' @export
tft_center_crop <- function(img, output_size) {

  check_img(img)

  image_width <- img$size(2)
  image_height <- img$size(3)

  crop_height <- output_size[1]
  crop_width <- output_size[2]

  crop_top <- as.integer((image_height - crop_height + 1) * 0.5)
  crop_left <- as.integer((image_width - crop_width + 1) * 0.5)

  tft_crop(img, crop_top, crop_left, crop_height, crop_width)
}


#' Crop the given Image Tensor into four corners and the central crop.
#'
#' @note
#' This transform returns a List of Tensors and there may be a
#' mismatch in the number of inputs and targets your ``Dataset`` returns.
#'
#' @param img (Tensor): Image to be cropped.
#' @param size (sequence or int): Desired output size of the crop. If size is an
#'   int instead of sequence like (h, w), a square crop (size, size) is
#'   made.
#'
#' @return
#' List: List (tl, tr, bl, br, center)
#' Corresponding top left, top right, bottom left, bottom right and center crop.
#'
#' @export
tft_five_crop <- function(img, size) {

  check_img(img)

  if (!length(size) == 2)
    value_error("Please provide only 2 dimensions (h, w) for size.")

  image_width <- img$size(2)
  image_height <- img$size(3)

  crop_height <- size[1]
  crop_width <- size[2]

  if (crop_width > image_width || crop_height > image_height)
    value_error("Requested crop size is bigger than input size.")

  tl <- tft_crop(img, 1, 1, crop_width, crop_height)
  tr <- tft_crop(img, image_width - crop_width + 1, 1, image_width, crop_height)
  bl <- tft_crop(img, 1, image_height - crop_height + 1, crop_width, image_height)
  br <- tft_crop(img, image_width - crop_width + 1, image_height - crop_height + 1, image_width, image_height)
  center <- tft_center_crop(img, c(crop_height, crop_width))

  list(tl, tr, bl, br, center)
}
