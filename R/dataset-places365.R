#' Places365 Dataset
#'
#' Loads the Places365 dataset for scene classification.
#'
#' - For `split = "train"`: images are organized by class folder.
#' - For `split = "val"`: images are in a flat directory and matched with a label file.
#' - For `split = "test"`: images are in a flat directory with no labels.
#'
#' @inheritParams fgvc_aircraft_dataset
#' @param split One of `"train"`, `"val"`, or `"test"`.
#' @param transform Optional image transform.
#' @param target_transform Optional label transform (ignored for test split).
#' @param download Whether to download the required files.
#'
#' @return A torch dataset object.
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

    return(image_folder_dataset(
      root = split_dir,
      transform = transform,
      target_transform = target_transform
    ))
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
      name = "places365_val_dataset",
      initialize = function() {
        self$samples <- samples
        self$transform <- transform
        self$target_transform <- target_transform
      },
      .length = function() length(self$samples),
      .getitem = function(index) {
        s <- self$samples[[index]]
        img <- magick::image_read(s$path)
        img <- transform_to_tensor(img)
        if (!is.null(self$transform))
          img <- self$transform(img)
        label <- s$label
        if (!is.null(self$target_transform))
          label <- self$target_transform(label)
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
      name = "places365_test_dataset",
      initialize = function() {
        self$files <- files
        self$transform <- transform
      },
      .length = function() length(self$files),
      .getitem = function(index) {
        path <- self$files[[index]]
        img <- magick::image_read(path)
        img <- transform_to_tensor(img)
        if (!is.null(self$transform))
          img <- self$transform(img)
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

    # Download val images
    val_dir <- file.path(base_dir, "val_256")
    if (!fs::dir_exists(val_dir)) {
      val_url <- "http://data.csail.mit.edu/places/places365/val_256.tar"
      val_tar <- download_and_cache(val_url, prefix = "places365")
      utils::untar(val_tar, exdir = base_dir)
    }
  }

  if (split == "train") {
    train_dir <- file.path(base_dir, "data_256")
    if (!fs::dir_exists(train_dir)) {
      train_url <- "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
      train_tar <- withr::with_options(
        list(timeout = 6000),
        download_and_cache(train_url, prefix = "places365")
      )

      utils::untar(train_tar, exdir = base_dir)
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
