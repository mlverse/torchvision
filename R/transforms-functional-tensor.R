







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




pad_symmetric <- function(img, padding) {

  in_sizes <- img$size()

  x_indices <- seq_len(tail(in_sizes, 1))
  left_indices <- rev(seq_len(padding[1]))
  right_indices <- -seq_len(padding[2])
  x_indices <- torch::torch_tensor(c(left_indices, x_indices, right_indices),
                             dtype = torch::torch_long())


  y_indices <- seq_len(rev(in_sizes)[2])
  top_indices <- rev(seq_len(padding[3]))
  bottom_indices <- -seq_len(padding[4])
  y_indices <- torch::torch_tensor(c(top_indices, y_indices, bottom_indices),
                                   dtype = torch::torch_long())


  ndim <- length(dim(img))
  if (ndim == 3)
    img[, y_indices[, NULL], x_indices[, NULL]]
  else if (ndim == 4)
    img[,,y_indices[, NULL], x_indices[, NULL]]
  else
    runtime_error("Symmetric padding of N-D tensors are not supported yet")
}

#' Pad the given Tensor Image on all sides with specified
#' padding mode and fill value.
#'
#' @param img (Tensor): Image to be padded.
#' @param padding (int or tuple or list): Padding on each border. If a single int is provided this
#'   is used to pad all borders. If a tuple or list of length 2 is provided this is the padding
#'   on left/right and top/bottom respectively. If a tuple or list of length 4 is provided
#'   this is the padding for the left, top, right and bottom borders
#'   respectively. In torchscript mode padding as single int is not supported, use a tuple or
#'   list of length 1: `[padding, ]`.
#' @param fill (int): Pixel fill value for constant fill. Default is 0.
#'   This value is only used when the padding_mode is constant
#' @param padding_mode (str): Type of padding. Should be: constant, edge or reflect. Default is constant.
#'   Mode symmetric is not yet supported for Tensor inputs.
#'   - constant: pads with a constant value, this value is specified with fill
#'   - edge: pads with the last value on the edge of the image
#'   - reflect: pads with reflection of image (without repeating the last value on the edge)
#'     padding `[1, 2, 3, 4]` with 2 elements on both sides in reflect mode
#'     will result in `[3, 2, 1, 2, 3, 4, 3, 2]`
#'   - symmetric: pads with reflection of image (repeating the last value on the edge)
#'     padding `[1, 2, 3, 4]` with 2 elements on both sides in symmetric mode
#'     will result in `[2, 1, 1, 2, 3, 4, 4, 3]`
#'
#' @return Tensor: Padded image.
#'
#' @export
tft_pad <- function(img, padding, fill = 0, padding_mode = "constant") {

  check_img(img)

  if (!length(padding) %in% c(1,2,4) || !is.numeric(padding))
    value_error("Padding must be an int or a 1, 2, or 4 element numeric vector")

  if (!padding_mode %in% c("constant", "edge", "reflect", "symmetric"))
    value_error("Padding mode should be either constant, edge, reflect or symmetric")


  if (length(padding) == 1)
    pad_left <- pad_right <- pad_top <- pad_bottom <- padding
  else if (length(padding) == 2) {
    pad_left <- pad_right <- padding[1]
    pad_top <- pad_bottom <- padding[2]
  } else if (length(padding == 4)) {
    pad_left <- padding[1]
    pad_right <- padding[2]
    pad_top <- padding[3]
    pad_bottom <- padding[4]
  }

  p <- c(pad_left, pad_right, pad_top, pad_bottom)

  if (padding_mode == "edge")
    padding_mode <- "replicate"
  else if (padding_mode == "symmetric") {
    if (any(p < 0))
      value_error("Padding can not be negative for symmetric padding_mode")

    return(pad_symmetric(img, p))
  }

  need_squeeze <- FALSE

  if (img$dim() < 4) {
    img <- img$unsqueeze(1)
    need_squeeze <- TRUE
  }

  out_dtype <- img$dtype()
  need_cast <- FALSE

  if (padding_mode != "constant" &&
      (img$dtype() == torch::torch_float32() || img$dtype() == torch::torch_float64())) {
    need_cast <- TRUE
    img <- img$to(torch::torch_float32())
  }

  img <- torch::nnf_pad(img, p, mode = padding_mode, value = as.numeric(fill))

  if (need_squeeze)
    img <- img$squeeze(dim = 1)

  if (need_cast)
    img <- img$to(dtype = out_dtype)

  img
}




#' Apply affine transformation on the Tensor image keeping image center invariant.
#'
#' @param img (Tensor): image to be rotated.
#' @param matrix (list of floats): list of 6 float values representing inverse matrix for affine transformation.
#' @param resample (int, optional): An optional resampling filter. Default is nearest (=2). Other supported values:
#'   bilinear(=2).
#' @param fillcolor (int, optional): this option is not supported for Tensor input. Fill value for the area outside the
#'   transform in the output image is always 0.
#'
#' @return Tensor: Transformed image.
tft_affine <- function(img, matrix, resample = 0, fillcolor = NULL) {

  check_img(img)

  if (!is.null(fillcolor))
    rlang::warn("Argument fillcolor is not supported for Tensor input")

  interpolation_modes <- c(
    "0" = "nearest",
    "2" = "bilinear"
  )

  if (!resample %in% names(interpolation_modes))
    value_error("This resampling is unsupported with Tensor inputs")

  theta <- torch::torch_tensor(matrix, dtype = torch::torch_float())$reshape(c(1,2,3))
  shape <- img$shape
  grid <- torch::nnf_affine_grid(theta, size = c(1, tail(shape, 3)),
                                 align_corners = FALSE)

  need_squeeze <- FALSE
  if (img$dim() < 4) {
    img <- img$unsqueeze(1)
    need_squeeze <- TRUE
  }

  mode <- interpolation_modes[resample == names(interpolation_modes)]

  out_dtype <- img$dtype()
  need_cast <- FALSE
  if (!img$dtype() == torch::torch_float32() &&
      !img$dtype() == torch::torch_float64()) {
    need_cast <- TRUE
    img <- img$to(dtype = torch::torch_float32())
  }

  img <- torch::nnf_grid_sample(img, grid, mode = mode, padding_mode = "zeros",
                                align_corners = FALSE)


  if (need_squeeze)
    img <- img$squeeze(1)

  if (need_cast) {
    img <- torch::torch_round(img)$to(out_dtype)
  }

  img
}

tft_perspective <- function(img, perspective_coeffs, interpolation = 2,
                            fill = NULL) {

  not_implemented_error("tft perspective is not implemented yet.")
}
