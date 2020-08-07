#' Convert a `Magick Image` or `array` to tensor.
#'
#' @param pic (`Magick Image` or `array`): Image to be converted to tensor.
#'
#' @return Tensor: Converted image.
#'
#' @export
transform_to_tensor <- function(pic) {

  if (!(is_magick_image(pic) || is_array_image(pic)))
    type_error("pic should be Magick image or array (with 2 or 3 dims)")


  # handle array
  if (is_array_image(pic)) {

    if (length(dim(pic)) == 2)
      dim(pic) <- c(dim(pic), 1)

    img <- torch::torch_tensor(pic)$transpose(c(3, 1, 2))

    return(img)
  }

  # handle magick
  if (is_magick_image(pic)) {
    return(transform_magick_to_tensor(pic))
  }

}

#' Convert a `PIL Image` to a tensor of the same type.
#'
#' @param pic (PIL Image): Image to be converted to tensor.
#' @return Tensor: Converted image.
#'
#' @export
transform_magick_to_tensor <- function(pic) {

  if (!is_magick_image(pic))
    type_error("pic must be a magick image")

  img <- as.integer(magick::image_data(pic))
  img <- torch::torch_tensor(img)$permute(c(3,1,2))
  img <- img$to(dtype = torch::torch_float32())
  img <- img$contiguous()
  img <- img$div(255)

  img
}

#' Convert a tensor image to the given `dtype` and scale the values accordingly
#'
#' @param image (torch.Tensor): Image to be converted
#' @param dtype (torch.dtype): Desired data type of the output
#'
#' @return (torch.Tensor): Converted image
#'
#' @note
#' When converting from a smaller to a larger integer `dtype` the maximum values
#' are **not** mapped exactly. If converted back and forth, this mismatch has no
#' effect.
#'
#' @section Raises:
#'
#' runtime_error: When trying to cast `torch_float32` to `torch_int32` or
#' `torch_int64` as well as for trying to cast `torch_float64` to `torch_int64`.
#' These conversions might lead to overflow errors since the floating point
#' `dtype` cannot store consecutive integers over the whole range of the integer
#' `dtype`.
#'
#' @export
transform_convert_image_dtype <- function(image, dtype = torch::torch_float()) {

  if (image$dtype() == dtype)
    return(image)

  if (image$is_floating_point()) {

    # float to float
    if (dtype$is_floating_point)
      return(image$to(dtype = dtype))

    # float to int
    if ((image$dtype() == torch::torch_float32() &&
        (dtype == torch::torch_float32() || dtype == torch::torch_float64())) ||
        (image$dtype() == torch::torch_float64() && dtype == torch::torch_int64())
        )
      runtime_error("The cast from {image$dtype()} to {dtype} cannot be performed safely.")

    # For data in the range 0-1, (float * 255).to(uint) is only 255
    # when float is exactly 1.0.
    # `max + 1 - epsilon` provides more evenly distributed mapping of
    # ranges of floats to ints.
    eps <- 1e-3
    result <- image$mul(torch::torch_iinfo(dtype)$max + 1 - eps)
    result <- result$to(dtype = dtype)

    return(result)
  } else {
    # int to float

    if (dtype$is_floating_point) {
      max <- torch::torch_iinfo(image$dtype())$max
      image <- image$to(dtype)

      return(image/max)
    }


    # int to int
    input_max <- torch::torch_iinfo(image$dtype())$max
    output_max <- torch::torch_iinfo(dtype)$max

    if (input_max > output_max) {
      factor <- (input_max + 1) %/% (output_max + 1)
      image = image %/% factor
      return(image$to(dtype = dtype))
    } else {

      factor <- (output_max + 1) %/% (input_max + 1)
      image <- image$to(dtype = dtype)

      return(image * factor)
    }
  }
}

#' Normalize a tensor image with mean and standard deviation.
#'
#' @note
#' This transform acts out of place by default, i.e., it does not mutates the
#' input tensor.
#'
#' @param tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#' @param mean (sequence): Sequence of means for each channel.
#' @param std (sequence): Sequence of standard deviations for each channel.
#' @param inplace (bool,optional): Bool to make this operation inplace.
#'
#' @return Tensor: Normalized Tensor image.
#'
#' @export
transform_normalize <- function(tensor, mean, std, inplace = FALSE) {

  check_img(tensor)

  if (!inplace)
    tensor <- tensor$clone()

  dtype <- tensor$dtype()
  mean <- torch::torch_tensor(mean, dtype=dtype, device=tensor$device())
  std <- torch::torch_tensor(std, dtype=dtype, device=tensor$device())

  if (torch::as_array((std == 0)$any())) {
    value_error("std evaluated to zero after conversion to {dtype}, leading to division by zero.")
  }

  if (mean$dim() == 1)
    mean <- mean[,NULL,NULL]

  if (std$dim() == 1)
    std <- std[,NULL,NULL]

  tensor$sub_(mean)$div_(std)

  tensor
}

#' Resize the input image to the given size.
#' The image can be a Magick Image or a torch Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading dimensions
#'
#' @param img (Magick Image or Tensor): Image to be resized.
#' @param size (sequence or int): Desired output size. If size is a sequence
#'   like (h, w), the output size will be matched to this. If size is an int,
#'   the smaller edge of the image will be matched to this number maintaining
#'   the aspect ratio. i.e, if height > width, then image will be rescaled to
#'   \eqn{\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)}.
#' @param interpolation (int, optional): Desired interpolation enum defined by `filters`.
#'   Default is `2 = BILINEAR`. If input is Tensor, only `0 = NEAREST`, `2 = BILINEAR`
#'   and `3 = BICUBIC` are supported.
#'
#' @return Magick Image or Tensor: Resized image.
#'
#' @export
transform_resize <- function(img, size, interpolation = 2) {
  if (is_magick_image(img))
    tfm_resize(img, size, interpolation)
  else
    tft_resize(img, size, interpolation)
}

#' Pad the given image on all sides with the given "pad" value.
#'
#' The image can be a Magick Image or a torch Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading dimensions
#'
#' @param img (Magick Image or Tensor): Image to be padded.
#' @param padding (int or tuple or list): Padding on each border. If a single int is provided this
#'   is used to pad all borders. If tuple of length 2 is provided this is the padding
#'   on left/right and top/bottom respectively. If a tuple of length 4 is provided
#'   this is the padding for the left, top, right and bottom borders respectively.
#' @param fill (int or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
#'   length 3, it is used to fill R, G, B channels respectively.
#'   This value is only used when the padding_mode is constant. Only int value is
#'   supported for Tensors.
#' @param padding_mode Type of padding. Should be: constant, edge, reflect or symmetric.
#'   Default is constant.
#'   Mode symmetric is not yet supported for Tensor inputs.
#'     - constant: pads with a constant value, this value is specified with fill
#'     - edge: pads with the last value on the edge of the image
#'     - reflect: pads with reflection of image (without repeating the last value on the edge)
#'                padding `[1, 2, 3, 4]` with 2 elements on both sides in reflect mode
#'                will result in `[3, 2, 1, 2, 3, 4, 3, 2]`
#'     - symmetric: pads with reflection of image (repeating the last value on the edge)
#'                  padding `[1, 2, 3, 4]` with 2 elements on both sides in symmetric mode
#'                  will result in `[2, 1, 1, 2, 3, 4, 4, 3]`
#' @return Magick Image or Tensor: Padded image.
#'
#' @return
transform_pad <- function(img, padding, fill = 0, padding_mode = "constant") {

  if (is_magick_image(img))
    not_implemented_error("pad is not implemented for magick images yet.")

  tft_pad(img, padding = padding, fill = fill, padding_mode = padding_mode)
}

#' Crop the given image at specified location and output size.
#'
#' The image can be a Magick Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' @param img (Magick Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
#' @param top (int): Vertical component of the top left corner of the crop box.
#' @param left (int): Horizontal component of the top left corner of the crop box.
#' @param height (int): Height of the crop box.
#' @param width (int): Width of the crop box.
#'
#' @return Magick Image or Tensor: Cropped image.
#'
#' @export
transform_crop <- function(img, top, left, height, width) {

  if (is_magick_image(img))
    not_implemented_error("crop is not implemented for magick images yet.")

  tft_crop(img, top = top, left = left, height = height, width = width)
}

#' Crops the given image at the center.
#'
#' The image can be a PIL Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' @param img (Magick Image or Tensor): Image to be cropped.
#' @param output_size (sequence or int): (height, width) of the crop box. If int
#'   or sequence with single int it is used for both directions.
#'
#' @return Magick Image or Tensor: Cropped image.
#'
#' @export
transform_center_crop <- function(img, output_size) {

  if (length(output_size) == 1)
    output_size <- rep(output_size, 2)

  output_size <- as.integer(output_size)

  size <- get_image_size(img)

  image_width <- size[1]
  image_height <- size[2]

  crop_height <- output_size[1]
  crop_width <- output_size[2]

  crop_top <- as.integer((image_height - crop_height) / 2)
  crop_left <- as.integer((image_width - crop_width) / 2)

  transform_crop(img, crop_top, crop_left, crop_height, crop_width)
}

#' Crop the given image and resize it to desired size.
#'
#' The image can be a Magick Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' @param img (Magick Image or Tensor): Image to be cropped. (1,1) denotes the
#'   top left corner of the image.
#' @param top (int): Vertical component of the top left corner of the crop box.
#' @param left (int): Horizontal component of the top left corner of the crop box.
#' @param height (int): Height of the crop box.
#' @param width (int): Width of the crop box.
#' @inheritParams transform_resize
#'
#' @return Magick Image or Tensor: Cropped image.
#'
#' @export
transform_resized_crop <- function(img, top, left, height, width, size,
                                   interpolation = 2) {
  img <- transform_crop(img, top, left, height, width)
  img <- transform_resize(img, size, interpolation)
  img
}

is_magick_image <- function(x) {
  inherits(x, "magick-image")
}

is_array_image <- function(x) {
  is.array(x) && (length(dim(x)) %in% c(2, 3))
}
