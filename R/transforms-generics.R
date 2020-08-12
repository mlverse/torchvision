#' Convert an image to tensor
#'
#' Converts a Magick Image or array (H x W x C) in the range
#' `[0, 255]` to a `torch_tensor` of shape (C x H x W) in the range `[0.0, 1.0]`
#'
#' In the other cases, tensors are returned without scaling.
#'
#' @note
#' Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
#' transforming target image masks.
#'
#' @param img A `magick-image`, `array` or `torch_tensor`.
#'
#' @family transforms
#'
#' @export
transform_to_tensor <- function(img) {
  UseMethod("transform_to_tensor", img)
}

#' Convert a tensor image to the given `dtype` and scale the values accordingly
#'
#' @inheritParams transform_to_tensor
#' @param dtype (torch.dtype): Desired data type of the output
#'
#' @note
#' When converting from a smaller to a larger integer `dtype` the maximum values
#' are **not** mapped exactly. If converted back and forth, this mismatch has
#' no effect.
#'
#' @family transforms
#'
#' @export
transform_convert_image_dtype <- function(img, dtype = torch::torch_float()) {
  UseMethod("transform_convert_image_dtype", img)
}

#' Normalize a tensor image with mean and standard deviation.
#'
#' Given mean: `(mean[1],...,mean[n])` and std: `(std[1],..,std[n])` for `n`
#' channels, this transform will normalize each channel of the input
#' `torch_tensor` i.e.,
#' `output[channel] = (input[channel] - mean[channel]) / std[channel]`
#'
#' @note
#' This transform acts out of place, i.e., it does not mutate the input tensor.
#'
#' @inheritParams transform_to_tensor
#' @param mean (sequence): Sequence of means for each channel.
#' @param std (sequence): Sequence of standard deviations for each channel.
#' @param inplace(bool,optional): Bool to make this operation in-place.
#'
#' @family transforms
#'
#' @export
transform_normalize <- function(img, mean, std, inplace = FALSE) {
  UseMethod("transform_normalize", img)
}

#' Resize the input image to the given size.
#'
#' The image can be a Magic Image or a torch Tensor, in which case it is expected
#' to have [..., H, W] shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' @inheritParams transform_to_tensor
#' @param size (sequence or int): Desired output size. If size is a sequence like
#'   (h, w), output size will be matched to this. If size is an int,
#'   smaller edge of the image will be matched to this number.
#'   i.e, if height > width, then image will be rescaled to
#'   (size * height / width, size).
#' @param interpolation (int, optional) Desired interpolation. An integer `0 = nearest`,
#'   `2 = bilinear` and `3 = bicubic` or a name from [magick::filter_types()].
#'
#' @family transforms
#'
#' @export
transform_resize <- function(img, size, interpolation = 2) {
  UseMethod("transform_resize", img)
}

#' Crops the given image at the center.
#'
#' The image can be a Magick Image or a torch Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' @inheritParams transform_to_tensor
#' @param size (sequence or int): Desired output size of the crop. If size is an
#'   int instead of sequence like (h, w), a square crop (size, size) is
#'   made. If provided a tuple or list of length 1, it will be interpreted as
#'   `(size, size)`.
#'
#' @family transforms
#'
#' @export
transform_center_crop <- function(img, size) {
  UseMethod("transform_center_crop", img)
}

#' Pad the given image on all sides with the given "pad" value.
#'
#' The image can be a Magick Image or a torch Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading dimensions
#'
#' @inheritParams transform_to_tensor
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
#'
#' @family transforms
#'
#' @export
transform_pad <- function(img, padding, fill = 0, padding_mode = "constant") {
  UseMethod("transform_pad", img)
}

#' Apply randomly a list of transformations with a given probability
#'
#' @inheritParams transform_to_tensor
#' @param transforms (list or tuple): list of transformations
#' @param p (float): probability
#'
#' @family transforms
#'
#' @export
transform_random_apply <- function(img, transforms, p = 0.5) {
  UseMethod("transform_random_apply", img)
}

#' Apply single transformation randomly picked from a list
#'
#' @inheritParams transform_random_apply
#'
#' @family transforms
#'
#' @export
transform_random_choice <- function(img, transforms) {
  UseMethod("transform_random_choice", img)
}

#' Apply a list of transformations in a random order
#'
#' @inheritParams transform_random_apply
#' @family transforms
#' @export
transform_random_order <- function(img) {
  UseMethod("transform_random_order", img)
}

#' Crop the given image at a random location.
#'
#' The image can be a Magick Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions.
#'
#' @inheritParams transform_resize
#' @inheritParams transform_pad
#'
#' @family transforms
#'
#' @export
transform_random_crop <- function(img, size, padding=None, pad_if_needed=False,
                                  fill=0, padding_mode="constant") {
  UseMethod("transform_random_crop", img)
}

#' Horizontally flip the given image randomly with a given probability.
#'
#' The image can be a Magick Image or a torch Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' @inheritParams transform_to_tensor
#' @param p (float): probability of the image being flipped. Default value is 0.5
#'
#' @family transforms
#' @export
transform_random_horizontal_flip <- function(img, p = 0.5) {
  UseMethod("transform_random_horizontal_flip", img)
}

#' Vertically flip the given image randomly with a given probability.
#'
#' The image can be a PIL Image or a torch Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where `...` means an arbitrary number of leading
#' dimensions
#'
#' @inheritParams transform_random_horizontal_flip
#'
#' @family transforms
#' @export
transform_random_vertical_flip <- function(img, p = 0.5) {
  UseMethod("transform_random_vertical_flip", img)
}

#' Crop the given image to random size and aspect ratio.
#'
#' The image can be a Magick Image or a Tensor, in which case it is expected
#' to have `[..., H, W]` shape, where ... means an arbitrary number of leading
#' dimensions
#'
#' A crop of random size (default: of 0.08 to 1.0) of the original size and a random
#' aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made.
#' This crop is finally resized to given size.
#'
#' This is popularly used to train the Inception networks.
#'
#' @inheritParams transform_resize
#' @inheritParams transform_pad
#' @param scale (tuple of float): range of size of the origin size cropped
#' @param ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
#'
#' @family transforms
#' @export
transform_random_resized_crop <- function(img, size, scale=c(0.08, 1.0),
                                          ratio=c(3. / 4., 4. / 3.),
                                          interpolation=2) {
  UseMethod("transform_random_resized_crop", img)
}

#' Crop the given image into four corners and the central crop.
#'
#' This transform returns a tuple of images and there may be a mismatch in the
#' number of inputs and targets your Dataset returns.
#'
#' @inheritParams transform_resize
#'
#' @family transforms
#' @export
transform_five_crop <- function(img, size) {
  UseMethod("transform_five_crop", img)
}

#' Crop the given image into four corners and the central crop plus the flipped version of
#' these (horizontal flipping is used by default).
#'
#' This transform returns a tuple of images and there may be a mismatch in the number of
#' inputs and targets your Dataset returns.
#'
#' @inheritParams transform_five_crop
#' @param vertical_flip (bool): Use vertical flipping instead of horizontal
#'
#' @family transforms
#' @export
transform_ten_crop <- function(img) {
  UseMethod("transform_ten_crop", img)
}

#' Transform a tensor image with a square transformation matrix and a mean_vector
#' computed offline.
#'
#' Given transformation_matrix and mean_vector, will flatten the `torch_tensor` and
#' subtract mean_vector from it which is then followed by computing the dot
#' product with the transformation matrix and then reshaping the tensor to its
#' original shape.
#'
#' @section Applications:
#' whitening transformation: Suppose X is a column vector zero-centered data.
#' Then compute the data covariance matrix [D x D] with torch.mm(X.t(), X),
#' perform SVD on this matrix and pass it as transformation_matrix.
#'
#' @inheritParams transform_to_tensor
#' @param transformation_matrix (Tensor): tensor [D x D], D = C x H x W
#' @param mean_vector (Tensor): tensor [D], D = C x H x W
#'
#' @family transforms
#' @export
transform_linear_transformation <- function(img, transformation_matrix, mean_vector) {
  UseMethod("transform_linear_transformation", img)
}

#' Randomly change the brightness, contrast and saturation of an image.
#'
#' @param brightness (float or tuple of float (min, max)): How much to jitter brightness.
#'   brightness_factor is chosen uniformly from `[max(0, 1 - brightness), 1 + brightness]`
#'   or the given `[min, max]`. Should be non negative numbers.
#' @param contrast (float or tuple of float (min, max)): How much to jitter contrast.
#'   contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
#'   or the given `[min, max]`. Should be non negative numbers.
#' @param saturation (float or tuple of float (min, max)): How much to jitter saturation.
#'   saturation_factor is chosen uniformly from `[max(0, 1 - saturation), 1 + saturation]`
#'   or the given `[min, max]`. Should be non negative numbers.
#' @param hue (float or tuple of float (min, max)): How much to jitter hue.
#'   hue_factor is chosen uniformly from `[-hue, hue]` or the given `[min, max]`.
#'   Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
#' @inheritParams transform_to_tensor
#'
#' @family transforms
#' @export
transform_color_jitter <- function(img, brightness=0, contrast=0, saturation=0, hue=0) {
  UseMethod("transform_color_jitter", img)
}

#' Rotate the image by angle.
#'
#' @param degrees (sequence or float or int): Range of degrees to select from.
#'   If degrees is a number instead of sequence like (min, max), the range of degrees
#'   will be (-degrees, +degrees).
#' @param resample (int, optional): An optional resampling filter.
#' @param expand (bool, optional): Optional expansion flag.
#'   If true, expands the output to make it large enough to hold the entire rotated image.
#'   If false or omitted, make the output image the same size as the input image.
#'   Note that the expand flag assumes rotation around the center and no translation.
#' @param center (list or tuple, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
#'   Default is the center of the image.
#' @param fill (n-tuple or int or float): Pixel fill value for area outside the rotated
#'   image. If int or float, the value is used for all bands respectively.
#'   Defaults to 0 for all bands. This option is only available for Pillow>=5.2.0.
#'   This option is not supported for Tensor input. Fill value for the area outside
#'   the transform in the output image is always 0.
#' @inheritParams transform_to_tensor
#'
#' @family transforms
#' @export
transform_random_rotation <- function(img, degrees, resample=FALSE, expand=FALSE,
                                      center=NULL, fill=NULL) {
  UseMethod("transform_random_rotation", img)
}

#' Random affine transformation of the image keeping center invariant.
#'
#' @inheritParams transform_random_rotation
#' @param translate (tuple, optional): tuple of maximum absolute fraction for horizontal
#'   and vertical translations. For example translate=(a, b), then horizontal shift
#'   is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
#'   randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
#' @param scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
#'   randomly sampled from the range a <= scale <= b. Will keep original scale by default.
#' @param shear (sequence or float or int, optional): Range of degrees to select from.
#'   If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
#'   will be applied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
#'   range `(shear[1], shear[2])` will be applied. Else if shear is a tuple or list of 4 values,
#'   a x-axis shear in `(shear[1], shear[2])` and y-axis shear in `(shear[3], shear[4])` will be applied.
#'   Will not apply shear by default.
#' @param fillcolor (tuple or int): Optional fill color (Tuple for RGB Image and int for grayscale) for the area
#'   outside the transform in the output image (Pillow>=5.0.0). This option is not supported for Tensor
#'   input. Fill value for the area outside the transform in the output image is always 0.
#'
#' @family transforms
#' @export
transform_random_affine <- function(img, degrees, translate=NULL, scale=NULL,
                                    shear=NULL, resample=0, fillcolor=0) {
  UseMethod("transform_random_affine", img)
}

#' Convert image to grayscale.
#'
#' @inheritParams transform_to_tensor
#' @param num_output_channels (int): (1 or 3) number of channels desired for output image
#'
#' @family transforms
#' @export
transform_grayscale <- function(img, num_output_channels) {
  UseMethod("transform_grayscale", img)
}

#' Randomly convert image to grayscale with a probability of p (default 0.1).
#'
#' @inheritParams transform_to_tensor
#' @param p (float): probability that image should be converted to grayscale.
#'
#' @family transforms
#' @export
transform_random_grayscale <- function(img, p = 0.1) {
  UseMethod("transform_random_grayscale", img)
}


#' Performs a random perspective transformation of the given image with a given probability.
#'
#' @param distortion_scale (float): argument to control the degree of distortion
#'   and ranges from 0 to 1.
#'   Default is 0.5.
#' @param p (float): probability of the image being transformed. Default is 0.5.
#' @inheritParams transform_resize
#' @inheritParams transform_pad
#'
#' @family transforms
#' @export
transform_random_perspective <- function(img, distortion_scale=0.5, p=0.5,
                                         interpolation=2, fill=0) {
  UseMethod("transform_random_perspective", img)
}

#' Randomly selects a rectangle region in an image and erases its pixels.
#'
#' 'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/pdf/1708.04896.pdf
#'
#' @inheritParams transform_to_tensor
#' @param p probability that the random erasing operation will be performed.
#' @param scale range of proportion of erased area against input image.
#' @param ratio range of aspect ratio of erased area.
#' @param value erasing value. Default is 0. If a single int, it is used to
#'   erase all pixels. If a tuple of length 3, it is used to erase
#'   R, G, B channels respectively.
#'   If a str of 'random', erasing each pixel with random values.
#' @param inplace: boolean to make this transform inplace. Default set to FALSE.
#'
#' @family transforms
#' @export
transform_random_erasing <- function(img, p=0.5, scale=c(0.02, 0.33), ratio=c(0.3, 3.3),
                                     value=0, inplace=FALSE) {
  UseMethod("transform_random_erasing", img)
}

# Other generics ----------------------------------------------------------

#' Crop the given image at specified location and output size.
#'
#' @inheritParams transform_to_tensor
#' @param top (int): Vertical component of the top left corner of the crop box.
#' @param left (int): Horizontal component of the top left corner of the crop box.
#' @param height (int): Height of the crop box.
#' @param width (int): Width of the crop box.
#'
#' @family transforms
#' @export
transform_crop <- function(img, top, left, height, width) {
  UseMethod("transform_crop", img)
}

#' Horizontally flip the given PIL Image or Tensor.
#'
#' @inheritParams transform_to_tensor
#'
#' @family transforms
#' @export
transform_hflip <- function(img) {
  UseMethod("transform_hflip", img)
}

#' Vertically flip the given PIL Image or torch Tensor.
#'
#' @inheritParams transform_to_tensor
#'
#' @family transforms
#' @export
transform_vflip <- function(img) {
  UseMethod("transform_vflip", img)
}

#' Crop the given image and resize it to desired size.
#'
#' @param top (int): Vertical component of the top left corner of the crop box.
#' @param left (int): Horizontal component of the top left corner of the crop box.
#' @param height (int): Height of the crop box.
#' @param width (int): Width of the crop box.
#' @inheritParams transform_resize
#'
#' @family transforms
#' @export
transform_resized_crop <- function(img, top, left, height, width, size,
                                   interpolation = 2) {
  UseMethod("transform_resized_crop", img)
}


#' Adjust brightness of an Image.
#'
#' @param brightness_factor (float):  How much to adjust the brightness. Can be
#'   any non negative number. 0 gives a black image, 1 gives the
#'   original image while 2 increases the brightness by a factor of 2.
#' @inheritParams transform_resize
#'
#' @family transforms
#' @export
transform_adjust_brightness <- function(img, brightness_factor) {
  UseMethod("transform_adjust_brightness", img)
}

#' Adjust contrast of an Image.
#'
#' @param contrast_factor (float): How much to adjust the contrast. Can be any
#'   non negative number. 0 gives a solid gray image, 1 gives the
#'   original image while 2 increases the contrast by a factor of 2.
#'
#' @inheritParams transform_resize
#'
#' @family transforms
#' @export
transform_adjust_contrast <- function(img, contrast_factor) {
  UseMethod("transform_adjust_contrast", img)
}

#' Adjust color saturation of an image.
#'
#' @param saturation_factor (float):  How much to adjust the saturation. 0 will
#'   give a black and white image, 1 will give the original image while
#'   2 will enhance the saturation by a factor of 2.
#'
#'
#' @inheritParams transform_resize
#'
#' @family transforms
#' @export
transform_adjust_saturation <- function(img, saturation_factor) {
  UseMethod("transform_adjust_saturation", img)
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
#' @param hue_factor (float):  How much to shift the hue channel. Should be in
#'   `[-0.5, 0.5]`. 0.5 and -0.5 give complete reversal of hue channel in
#'   HSV space in positive and negative direction respectively.
#'   0 means no shift. Therefore, both -0.5 and 0.5 will give an image
#'   with complementary colors while 0 gives the original image.
#'
#' @inheritParams transform_resize
#'
#' @family transforms
#' @export
transform_adjust_hue <- function(img, hue_factor) {
  UseMethod("transform_adjust_hue", img)
}


#' Rotate the image by angle.
#'
#' @inheritParams transform_to_tensor
#' @inheritParams transform_random_rotation
#'
#' @family transforms
#' @export
transform_rotate <- function(img, angle, resample = 0, expand = FALSE,
                             center = NULL, fill = NULL) {
  UseMethod("transform_rotate", img)
}
