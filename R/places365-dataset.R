#' Places365 dataset
#'
#' Prepares the Places365 dataset and optionally downloads it.
#'
#' @param root directory path to download the dataset.
#' @param split dataset split, `train`, `val`, or `test`.
#' @param download whether to download the dataset.
#' @param ... other arguments passed to [image_folder_dataset()].
#'
#' @family dataset
#'
#' @export
places365_dataset <- torch::dataset(
  "places365",
  inherit = image_folder_dataset,
  tar_name = "places365standard_easyformat",
  url = "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar",
  initialize = function(root, split = "train", download = FALSE, ...) {
    root <- normalizePath(root, mustWork = FALSE)

    if (!fs::dir_exists(root))
      fs::dir_create(root)

    self$root_path <- root

    if (download)
      self$download()

    super$initialize(root = fs::path_join(c(root, self$tar_name, split)), ...)
  },

  download = function() {
    p <- fs::path_join(c(self$root_path, self$tar_name))
    if (fs::dir_exists(p))
      return(NULL)

    raw_path <- fs::path_join(c(self$root_path, "places365standard_easyformat.tar"))

    rlang::inform("Downloading Places365 dataset!")

    p <- download_and_cache(self$url)
    fs::file_copy(p, raw_path, overwrite = TRUE)

    rlang::inform("Download complete. Now unzipping.")

    utils::untar(raw_path, exdir = self$root_path)

    rlang::inform("Done!")
  }
)
