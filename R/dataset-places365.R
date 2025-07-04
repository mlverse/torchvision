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
places365_dataset <- function(root = tempdir(),
                              split = c("train", "val", "test"),
                              transform = NULL,
                              target_transform = NULL,
                              download = FALSE) {
  split <- match.arg(split)
  root <- normalizePath(root, mustWork = FALSE)
  base_dir <- file.path(root, "places365")

  # Download if needed
  if (download)
    places365_download(base_dir, split)

  if (split == "train") {
    split_dir <- file.path(base_dir, "data_256")
    if (!fs::dir_exists(split_dir))
      cli_abort("Training folder not found at '{split_dir}'")

    # Get all image paths
    files <- fs::dir_ls(split_dir, recurse = TRUE, type = "file", glob = "*.jpg")

    # Parse into samples with class names from folder structure
    samples <- lapply(files, function(path) {
      # Extract folder name (e.g., .../a/airfield/00000001.jpg → airfield)
      parts <- fs::path_split(path)[[1]]
      label <- parts[length(parts) - 1]  # second-to-last is the class
      list(path = path, label = label)
    })

    # Create class-to-index mapping
    class_names <- sort(unique(vapply(samples, function(s) s$label, character(1))))
    class_to_idx <- setNames(seq_along(class_names), class_names)

    dataset <- torch::dataset(
      name = "places365_dataset",
      initialize = function() {
        self$samples <- samples
        self$class_to_idx <- class_to_idx
        self$transform <- transform
        self$target_transform <- target_transform
      },
      .length = function() length(self$samples),
      .getitem = function(index) {
        s <- self$samples[[index]]
        path <- s$path
        class_name <- s$label
        label <- self$class_to_idx[[class_name]]

        img <- magick::image_read(path)

        if (!inherits(img, "magick-image")) {
          cli::cli_abort("Expected magick-image, got {class(img)} for path: {path}")
        }

        if (!is.null(self$transform)) {
          img <- self$transform(img)
        }

        if (!is.null(self$target_transform)) {
          label <- self$target_transform(label)
        }

        list(x = img, y = label)
      }
    )

    return(dataset())
  }



  if (split == "val") {
    annotation_file <- file.path(base_dir, "places365_val.txt")
    val_dir <- file.path(base_dir, "val_256")

    if (!fs::file_exists(annotation_file) || !fs::dir_exists(val_dir))
      cli_abort("Missing validation files. Use `download = TRUE` to fetch them.")

    lines <- suppressWarnings(readLines(annotation_file))
    parsed <- strsplit(lines, " ")
    samples <- lapply(parsed, function(x) list(
      path = file.path(val_dir, x[[1]]),
      label = as.integer(x[[2]]) + 1
    ))

    dataset <- torch::dataset(
      name = "places365_dataset",
      initialize = function() {
        self$samples <- samples
        self$transform <- transform
        self$target_transform <- target_transform
      },
      .length = function() length(self$samples),
      .getitem = function(index) {
        s <- self$samples[[index]]
        path <- s$path

        if (!fs::file_exists(path)) {
          cli_abort("Image file does not exist: {path}")
        }

        img <- magick::image_read(path)

        if (!inherits(img, "magick-image")) {
          cli_abort("Expected magick-image, got {class(img)} for path: {path}")
        }

        # Apply transform (e.g. transform_to_tensor)
        if (!is.null(self$transform)) {
          img <- self$transform(img)
        }

        label <- s$label

        if (!is.null(self$target_transform)) {
          label <- self$target_transform(label)
        }

        list(x = img, y = label)
      }

    )

    return(dataset())
  }

  if (split == "test") {
    test_dir <- file.path(base_dir, "test_256")
    if (!fs::dir_exists(test_dir))
      cli_abort("Test folder not found at '{test_dir}'")

    files <- fs::dir_ls(test_dir, recurse = FALSE, type = "file", glob = "*.jpg")

    dataset <- torch::dataset(
      name = "places365_dataset",
      initialize = function() {
        self$files <- files
        self$transform <- transform
      },
      .length = function() length(self$files),
      .getitem = function(index) {
        path <- self$files[[index]]

        if (!fs::file_exists(path)) {
          cli_abort("Image file does not exist: {path}")
        }

        img <- magick::image_read(path)

        if (!inherits(img, "magick-image")) {
          cli_abort("Expected magick-image, got {class(img)} for path: {path}")
        }

        # Apply transform (e.g., transform_to_tensor)
        if (!is.null(self$transform)) {
          img <- self$transform(img)
        }

        list(x = img)
      }

    )

    return(dataset())
  }

  cli_abort("Unknown split '{split}'")
}


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
