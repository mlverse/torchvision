

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
