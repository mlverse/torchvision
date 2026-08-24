dataset_task_classes <- c("object_detection_dataset", "segmentation_dataset")

#' Give a dataset instance its task class
#'
#' The class is inserted right before `dataset`, so that task methods take
#' precedence over the generic `dataset` ones while the dataset's own name stays
#' first, as dataset messages report `class(self)[[1]]`. Any task class inherited
#' from the generator this dataset was built on is dropped, so that a
#' segmentation dataset built on top of a detection one is not both.
#'
#' @param self A `dataset` instance, modified in place.
#' @param task_class One of `dataset_task_classes`.
#' @keywords internal
#' @noRd
set_dataset_task_class <- function(self, task_class) {
  cls <- setdiff(class(self), dataset_task_classes)
  class(self) <- append(cls, task_class, after = match("dataset", cls) - 1L)
  invisible(self)
}

#' @rdname set_dataset_task_class
#' @noRd
as_object_detection_dataset <- function(self) {
  set_dataset_task_class(self, "object_detection_dataset")
}

#' @rdname set_dataset_task_class
#' @noRd
as_segmentation_dataset <- function(self) {
  set_dataset_task_class(self, "segmentation_dataset")
}

#' Coerce a detection target to the `object_detection_target` class
#'
#' @param target A list holding at least `boxes`, as built by the `.getitem()`
#'   method of an object detection dataset.
#'
#' @return `target` with `object_detection_target` prepended to its class.
#' @keywords internal
#' @noRd
as_object_detection_target <- function(target) {
  if (inherits(target, "object_detection_target"))
    return(target)

  if (!"boxes" %in% names(target))
    cli_abort("The provided target with fields {.field {names(target)}} is not an object detection target: it has no {.field boxes}.")

  class(target) <- c("object_detection_target", class(target))
  target
}

#' Coerce a segmentation target to the `segmentation_target` class
#'
#' Segmentation datasets carry very different annotations -- polygons, trimaps,
#' per-class masks, instance ids -- so the fields are left to the dataset and
#' only checked by the target transform that consumes them.
#'
#' @param target A list, as built by the `.getitem()` method of a segmentation
#'   dataset.
#'
#' @return `target` with `segmentation_target` prepended to its class.
#' @keywords internal
#' @noRd
as_segmentation_target <- function(target) {
  if (inherits(target, "segmentation_target"))
    return(target)

  class(target) <- c("segmentation_target", class(target))
  target
}
