#' Coerce a detection target to the `object_detection_target` class
#'
#' @param target A list holding at least `boxes`, as built by the `.getitem()`
#'   method of an object detection dataset.
#'
#' @return `target` with `object_detection_target` prepended to its class.
#' @keywords internal
#' @noRd
as_object_detection_target <- function(target) {
  if (!"boxes" %in% names(target)) {
    cli_abort("The provided target with fields {.field {names(target)}} is not an object detection target: it has no {.field boxes}.")
  }
  class(target) <- c("object_detection_target", class(target))
  target
}

#' Add the `object_detection_dataset` class to a dataset instance
#'
#' The class is inserted right before `dataset`, so that detection methods take
#' precedence over the generic `dataset` ones and the dataset's own name stays
#' first, as dataset messages report `class(self)[[1]]`.
#'
#' @param self A `dataset` instance, modified in place.
#' @keywords internal
#' @noRd
as_object_detection_dataset <- function(self) {
  cls <- class(self)
  class(self) <- append(cls, "object_detection_dataset", after = match("dataset", cls) - 1L)
  invisible(self)
}
