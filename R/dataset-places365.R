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
#' @name places365_dataset
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
#' tensor_image_browse(item$x)
#'
#' dl <- dataloader(ds, batch_size = 2)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#' batch$x
#' }
#'
#' @family classification_dataset
#' @export

places365_dataset <- torch::dataset(
  name = "places365_dataset",
  archive_size = "34 GB",

  initialize = function(
    root = tempdir(),
    split = c("train", "val", "test"),
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$split <- match.arg(split)
    self$root_path <- normalizePath(root, mustWork = FALSE)
    self$transform <- transform
    self$target_transform <- target_transform

    if (download) {
      cli_inform("{.cls {class(self)[[1]]}} Downloading split '{self$split}' if needed.")
      self$download()
    }

    if (!self$check_exists())
      cli_abort("Missing {self$split} files. Use `download = TRUE` to fetch them.")

    if (self$split == "train") {
      files <- fs::dir_ls(self$train_dir, recurse = TRUE, type = "file", glob = "*.jpg")
      self$samples <- lapply(files, function(p) {
        parts <- fs::path_split(p)[[1]]
        list(path = p, label = parts[length(parts) - 1])
      })
      classes <- sort(unique(vapply(self$samples, function(s) s$label, character(1))))
      self$class_to_idx <- setNames(seq_along(classes), classes)
    } else if (self$split == "val") {
      lines <- suppressWarnings(readLines(self$val_ann))
      parsed <- strsplit(lines, " ")
      self$samples <- lapply(parsed, function(x) list(path = file.path(self$val_dir, x[[1]]), label = as.integer(x[[2]]) + 1))
    } else if (self$split == "test") {
      self$files <- fs::dir_ls(self$test_dir, recurse = FALSE, type = "file", glob = "*.jpg")
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
      img <- magick::image_read(path)
      if (!is.null(self$transform))
        img <- self$transform(img)
      list(x = img)
    } else {
      s <- self$samples[[index]]
      if (!fs::file_exists(s$path))
        cli_abort("Image file does not exist: {s$path}")
      img <- magick::image_read(s$path)
      if (!is.null(self$transform))
        img <- self$transform(img)
      label <- if (self$split == "train") self$class_to_idx[[s$label]] else s$label
      if (!is.null(self$target_transform))
        label <- self$target_transform(label)
      list(x = img, y = label)
    }
  },

  download = function() {
    if (self$check_exists())
      return()

    fs::dir_create(self$base_dir, recurse = TRUE)

    if (self$split == "val") {
      if (!fs::file_exists(self$val_ann)) {
        devkit_url <- "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
        devkit_tar <- file.path(self$base_dir, "filelist_places365-standard.tar")
        download.file(devkit_url, devkit_tar, mode = "wb", quiet = TRUE)
        utils::untar(devkit_tar, exdir = self$base_dir)
      }

      if (!fs::dir_exists(self$val_dir)) {
        val_url <- "http://data.csail.mit.edu/places/places365/val_256.tar"
        val_tar <- download_and_cache(val_url, prefix = "places365")
        utils::untar(val_tar, exdir = self$base_dir)
      }
    }

    if (self$split == "train") {
      if (!fs::dir_exists(self$train_dir)) {
        train_url <- "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
        train_tar <- withr::with_options(
          list(timeout = 6000),
          download_and_cache(train_url, prefix = "places365")
        )
        utils::untar(train_tar, exdir = self$base_dir)
      }
    }

    if (self$split == "test") {
      if (!fs::dir_exists(self$test_dir)) {
        test_url <- "http://data.csail.mit.edu/places/places365/test_256.tar"
        test_tar <- download_and_cache(test_url, prefix = "places365")
        utils::untar(test_tar, exdir = self$base_dir)
      }
    }
  },

  check_exists = function() {
    if (self$split == "train")
      fs::dir_exists(self$train_dir)
    else if (self$split == "val")
      fs::file_exists(self$val_ann) && fs::dir_exists(self$val_dir)
    else if (self$split == "test")
      fs::dir_exists(self$test_dir)
    else
      FALSE
  },

  active = list(
    base_dir = function() file.path(self$root_path, "places365"),
    train_dir = function() file.path(self$base_dir, "data_256"),
    val_dir = function() file.path(self$base_dir, "val_256"),
    test_dir = function() file.path(self$base_dir, "test_256"),
    val_ann = function() file.path(self$base_dir, "places365_val.txt")
  )
)
