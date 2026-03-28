
#' Tiny ImageNet dataset
#'
#' Prepares the Tiny ImageNet dataset and optionally downloads it.
#'
#' @param root directory path to download the dataset.
#' @param split dataset split, `train`, `validation` or `test`.
#' @param download whether to download or not the dataset.
#' @param ... other arguments passed to [image_folder_dataset()].
#'
#' @family classification_dataset
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

    cli_inform("Downloading {.cls {class(self)[[1]]}} ...")

    p <- download_and_cache(self$url)
    fs::file_copy(p, raw_path)

    cli_inform("Processing {.cls {class(self)[[1]]}} ...")

    utils::unzip(raw_path, exdir = self$root_path)

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

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")

  }
)

#' ImageNet Class Labels
#'
#' Utilities for resolving ImageNet-1k class identifiers to their corresponding
#' human readable labels. The labels are retrieved from the same source used by
#' PyTorch's reference implementation.
#'
#' @param class_id Integer vector of 1-based class identifiers.
#' @return A character vector with 1000 entries representing the ImageNet-1k
#'   class labels.
#' @family class_resolution
#' @export
imagenet_classes <- function(class_id = 1:1000) {
  if (any(class_id > 1000)) {
    cli_warn("Imagenet {.var class_id} cannot be > 1000")
  }
  url <- download_and_cache("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
  labels <- readLines(url, warn = FALSE)
  labels[nzchar(labels)][class_id]
}

#' @family class_resolution
#' @rdname imagenet_classes
#' @export
imagenet_1k_classes <- imagenet_classes

#' @param class_id Integer vector of 1-based class identifiers.
#' @return A data.frame containing columns `id` and `label` representing 
#'   the ImageNet-21k class identifiers and labels. By default, returns all 21.8K rows.
#' @family class_resolution
#' @rdname imagenet_classes
#' @export
imagenet_21k_df <- function(class_id = 1:21843) {
  if (any(class_id > 21843)) {
    cli_warn("Imagenet {.var class_id} cannot be > 21843")
  }
  url <- download_and_cache("https://raw.githubusercontent.com/ailia-ai/ailia-models/refs/heads/master/object_detection/detic/datasets_detic/imagenet21k_wordnet_ids.txt")
  ids <- readLines(url, warn = FALSE)
  url <- download_and_cache("https://raw.githubusercontent.com/vkinakh/stable-diffusion-imagenet/refs/heads/main/imagenet21k_wordnet_lemmas.txt")
  labels <- readLines(url, warn = FALSE)

  data.frame(id = ids, label = labels)[class_id,]
}

#' @param class_id Integer vector of 1-based class identifiers.
#' @return A character vector with the labels associated with `class_id`.
#' @family class_resolution
#' @rdname imagenet_classes
#' @export
imagenet_21k_classes <- function(class_id) {
  imagenet_21k_df(class_id)$label
}

