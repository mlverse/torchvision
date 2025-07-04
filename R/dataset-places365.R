#' Places365 Dataset
#'
#' Loads the MIT Places365 dataset for scene classification.
#'
#' The dataset is split into:
#' - `"train"`: images are organized by class folder.
#' - `"val"`: images are in a flat directory and matched with a label file.
#' - `"test"`: images are in a flat directory with no labels.
#'
#' @param root Root directory for dataset storage. The dataset will be stored under `root/places365`.
#' @param split One of `"train"`, `"val"`, or `"test"`.
#' @param transform Optional image transform.
#' @param target_transform Optional label transform (ignored for the test split).
#' @param download Logical. Whether to download the required files.
#'
#' @return An object of class \code{places365_dataset}. Each element is a named
#'   list with:
#'   - `x`: the image as loaded (or transformed if `transform` is set).
#'   - `y`: the integer class label (not returned for the test split).
#'
#' @examples
#' \dontrun{
#' ds <- places365_dataset(
#'   split = "val",
#'   download = TRUE,
#'   transform = transform_to_tensor
#' )
#' item <- ds[1]
#' item$x
#' }
#'
#' @family classification_dataset
#' @export

places365_dataset <- torch::dataset(
  name = "places365_dataset",
  initialize = function(
    root = tempdir(),
    split = c("train", "val", "test"),
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$split <- match.arg(split)
    self$root <- normalizePath(root, mustWork = FALSE)
    self$base_dir <- file.path(self$root, "places365")
    self$transform <- transform
    self$target_transform <- target_transform
    if (download)
      places365_download(self$base_dir, self$split)

    if (self$split == "train") {
      split_dir <- file.path(self$base_dir, "data_256")
      if (!fs::dir_exists(split_dir))
        cli_abort("Training folder not found at '{split_dir}'")
      files <- fs::dir_ls(split_dir, recurse = TRUE, type = "file", glob = "*.jpg")
      self$samples <- lapply(files, function(p) {
        parts <- fs::path_split(p)[[1]]
        list(path = p, label = parts[length(parts) - 1])
      })
      classes <- sort(unique(vapply(self$samples, function(s) s$label, character(1))))
      self$class_to_idx <- setNames(seq_along(classes), classes)
    } else if (self$split == "val") {
      ann <- file.path(self$base_dir, "places365_val.txt")
      val_dir <- file.path(self$base_dir, "val_256")
      if (!fs::file_exists(ann) || !fs::dir_exists(val_dir))
        cli_abort("Missing validation files. Use `download = TRUE` to fetch them.")
      lines <- suppressWarnings(readLines(ann))
      parsed <- strsplit(lines, " ")
      self$samples <- lapply(parsed, function(x) list(path = file.path(val_dir, x[[1]]), label = as.integer(x[[2]]) + 1))
    } else if (self$split == "test") {
      test_dir <- file.path(self$base_dir, "test_256")
      if (!fs::dir_exists(test_dir))
        cli_abort("Test folder not found at '{test_dir}'")
      self$files <- fs::dir_ls(test_dir, recurse = FALSE, type = "file", glob = "*.jpg")
    } else {
      cli_abort("Unknown split '{self$split}'")
    }
  },

  .length = function() {
    if (self$split == "test")
      length(self$files)
    else
      length(self$samples)
  },

  .getitem = function(index) {
    if (self$split == "test") {
      path <- self$files[[index]]
      if (!fs::file_exists(path))
        cli_abort("Image file does not exist: {path}")
      img <- magick::image_read(path) %>% image_auto_orient()
      if (!is.null(self$transform))
        img <- self$transform(img)
      list(x = img)
    } else {
      s <- self$samples[[index]]
      if (!fs::file_exists(s$path))
        cli_abort("Image file does not exist: {s$path}")
      img <- magick::image_read(s$path) %>% image_auto_orient()
      if (!is.null(self$transform))
        img <- self$transform(img)
      label <- if (self$split == "train") self$class_to_idx[[s$label]] else s$label
      if (!is.null(self$target_transform))
        label <- self$target_transform(label)
      list(x = img, y = label)
    }
  }
)


places365_download <- function(base_dir, split) {
  fs::dir_create(base_dir, recurse = TRUE)

  if (split == "val") {
    # Download val labels
    ann_file <- file.path(base_dir, "places365_val.txt")
    if (!fs::file_exists(ann_file)) {
      devkit_url <- "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
      devkit_tar <- file.path(base_dir, "filelist_places365-standard.tar")
      download.file(devkit_url, devkit_tar, mode = "wb", quiet = TRUE)
      utils::untar(devkit_tar, exdir = base_dir)
    }
  }

  if (split == "test") {
    test_dir <- file.path(base_dir, "test_256")
    if (!fs::dir_exists(test_dir)) {
      test_url <- "http://data.csail.mit.edu/places/places365/test_256.tar"
      test_tar <- download_and_cache(test_url, prefix = "places365")
      utils::untar(test_tar, exdir = base_dir)
    }
  }
}
