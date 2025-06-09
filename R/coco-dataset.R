#' MS COCO Dataset
#'
#' Loads MS COCO Detection Dataset for object detection and segmentation.
#'
#' @param root Root directory where data is downloaded or stored.
#' @param split One of "train", "val".
#' @param year Dataset year, e.g., "2017".
#' @param download Whether to download the dataset.
#' @param ... Additional arguments passed to the dataset.
#'
#' @export
coco_dataset <- dataset(
  "coco",
  initialize = function(root, split = "train", year = "2017", download = FALSE, ...) {
    self$root <- fs::path(root, glue::glue("coco/{year}/{split}"))
    if (download) {
      self$download(split, year)
    }

    # TODO: load images and annotations here
  },
  download = function(split, year) {
    rlang::inform("Downloading COCO dataset...")
    # TODO: add actual download logic using download_and_cache()
  }
)
