
#' Tiny ImageNet dataset
#'
#' Prepares the Tiny ImageNet dataset and optionally downloads it.
#'
#' @param root directory path to download the dataset.
#' @param split dataset split, `train`, `validation` or `test`.
#' @param download whether to download or not the dataset.
#' @param ... other arguments passed to [image_folder_dataset()].
#'
#' @family dataset
#'
#' @export
tiny_imagenet_dataset <- torch::dataset(
  "tiny_imagenet",
  inherit = image_folder_dataset,
  tar_name = "tiny-imagenet-200",
  url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
  initialize = function(root, split='train', download = FALSE, ...) {

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

    raw_path <- fs::path_join(c(self$root_path, "tiny-imagenet-200.zip"))

    rlang::inform("Downloding tiny imagenet dataset!")

    p <- download_and_cache(self$url)
    fs::file_copy(p, raw_path)

    rlang::inform("Download complete. Now unzipping.")

    if (is.null(rlang::check_installed("zip"))) {
      zip::unzip(raw_path, exdir = self$root_path)
    }

    # organize validation images
    val_path <- fs::path_join(c(self$root_path, self$tar_name, "val"))
    val_images <- read.table(fs::path_join(c(val_path, "val_annotations.txt")))

    fs::dir_create(
      fs::path(val_path, unique(val_images$V2))
    )

    fs::file_move(
      fs::path(val_path, "images", val_images$V1),
      fs::path(val_path, val_images$V2, val_images$V1)
    )

    fs::dir_delete(fs::path(val_path, "images"))

    rlang::inform("Done!")

  }
)
