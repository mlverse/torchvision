#' @export
transform_to_tensor.default <- function(img) {
  not_implemented_for_class(img)
}

#' @export
transform_convert_image_dtype.default <- function(img) {
  not_implemented_for_class(img)
}


#' @export
transform_normalize.default <- function(img) {
  not_implemented_for_class(img)
}

#' @export
transform_resize.default <- function(img) {

}

#' @export
transform_scale.default <- function(img) {

}
#' @export
transform_center_crop.default <- function(img) {

}
#' @export
transform_pad.default <- function(img) {

}
#' @export
transform_lambda.default <- function(img) {

}
#' @export
transform_random_apply.default <- function(img) {

}
#' @export
transform_random_choice.default <- function(img) {

}
#' @export
transform_random_order.default <- function(img) {

}
#' @export
transform_random_crop.default <- function(img) {

}
#' @export
transform_random_horizontal_flip.default <- function(img) {

}
#' @export
transform_random_vertical_flip.default <- function(img) {

}
#' @export
transform_random_resized_crop.default <- function(img) {

}
#' @export
transform_random_sized_crop.default <- function(img) {

}
#' @export
transform_five_crop.default <- function(img) {

}
#' @export
transform_ten_crop.default <- function(img) {

}
#' @export
transform_linear_transformation.default <- function(img) {

}
#' @export
transform_color_jitter.default <- function(img) {

}
#' @export
transform_random_rotation.default <- function(img) {

}
#' @export
transform_random_affine.default <- function(img) {

}
#' @export
transform_grayscale.default <- function(img) {

}
#' @export
transform_random_grayscale.default <- function(img) {

}
#' @export
transform_random_perspective.default <- function(img) {

}
#' @export
transform_random_erasing.default <- function(img) {

}

# Helpers -----------------------------------------------------------------

not_implemented_for_class <- function(x) {
  not_implemented_error("not implemented for ", class(x))
}

