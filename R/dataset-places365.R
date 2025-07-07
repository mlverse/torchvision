#' Places365 Dataset
#'
#' Loads the MIT Places365 dataset for scene classification.
#'
#' The dataset is split into:
#' - `"train"`: images are organized by class folder.
#' - `"val"`: images are in a flat directory and matched with a label file.
#' - `"test"`: images are in a flat directory with no labels.
#'
#' @inheritParams mnist_dataset
#' @param split One of `"train"`, `"val"`, or `"test"`.
#'
#' @name places365_dataset
#'
#' @return A torch dataset of class \code{places365_dataset}. Each element is a named
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
  archive_size = "24 GB",

  devkit_url = "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar",
  devkit_md5 = "1a930ef26e7794c2c53d6c5f8c3c43d2",

  val_url = "http://data.csail.mit.edu/places/places365/val_256.tar",
  val_md5 = "c50ef2655c708abf8bfa9e523101c4ec",

  train_url = "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar",
  train_md5 = "53ca1c756c3d1e7809517cc47c5561c5",

  test_url = "http://data.csail.mit.edu/places/places365/test_256.tar",
  test_md5 = "2c6e8f4d279c616dc1ce34a4490e8937",

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
      cli_inform("{.cls {class(self)[[1]]}} Dataset (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    if (self$split == "train") {
      files <- fs::dir_ls(self$train_dir, recurse = TRUE, type = "file", glob = "*.jpg")
      self$items <- lapply(files, function(p) {
        parts <- fs::path_split(p)[[1]]
        list(path = p, label = parts[length(parts) - 1])
      })
      classes <- sort(unique(vapply(self$items, function(e) e$label, character(1))))
      self$class_to_idx <- setNames(seq_along(classes), classes)
    } else if (self$split == "val") {
      ann <- read.delim(self$val_ann, header = FALSE, col.names = c("path", "label"), sep = " ")
      ann$label <- ann$label + 1L
      self$items <- lapply(seq_len(nrow(ann)), function(i) {
        list(path = file.path(self$val_dir, ann$path[i]), label = ann$label[i])
      })
    } else if (self$split == "test") {
      self$files <- fs::dir_ls(self$test_dir, recurse = FALSE, type = "file", glob = "*.jpg")
    }

    cli_inform("{.cls {class(self)[[1]]}} Split '{self$split}' loaded with {length(self)} samples.")
  },

  .length = function() {
    if (self$split == "test")
      length(self$files)
    else
      length(self$items)
  },

  .getitem = function(index) {
    if (self$split == "test") {
      path <- self$files[[index]]
      if (!fs::file_exists(path))
        cli_abort("Image file does not exist: {path}")
      x <- magick::image_read(path)
      if (!is.null(self$transform))
        x <- self$transform(x)
      list(x = x, y = NA_integer_)
    } else {
      item <- self$items[[index]]
      if (!fs::file_exists(item$path))
        cli_abort("Image file does not exist: {item$path}")
      x <- magick::image_read(item$path)
      if (!is.null(self$transform))
        x <- self$transform(x)
      y <- if (self$split == "train") self$class_to_idx[[item$label]] else item$label
      if (!is.null(self$target_transform))
        y <- self$target_transform(y)
      list(x = x, y = y)
    }
  },

  download = function() {
    if (self$check_exists())
      return()

    cli_inform("Downloading {.cls {class(self)[[1]]}} split: '{self$split}'")
    fs::dir_create(self$base_dir, recurse = TRUE)

    if (self$split == "val") {
      if (!fs::file_exists(self$val_ann)) {
        devkit_tar <- download_and_cache(self$devkit_url, prefix = "places365")
        if (!is.na(self$devkit_md5) && tools::md5sum(devkit_tar) != self$devkit_md5)
          cli_abort("Corrupt file! Delete the file in {.file {devkit_tar}} and try again.")
        utils::untar(devkit_tar, exdir = self$base_dir)
      }

      if (!fs::dir_exists(self$val_dir)) {
        val_tar <- download_and_cache(self$val_url, prefix = "places365")
        if (!is.na(self$val_md5) && tools::md5sum(val_tar) != self$val_md5)
          cli_abort("Corrupt file! Delete the file in {.file {val_tar}} and try again.")
        utils::untar(val_tar, exdir = self$base_dir)
      }
    }

    if (self$split == "train") {
      if (!fs::dir_exists(self$train_dir)) {
        train_tar <- withr::with_options(
          list(timeout = 6000),
          download_and_cache(self$train_url, prefix = "places365")
        )
        if (!is.na(self$train_md5) && tools::md5sum(train_tar) != self$train_md5)
          cli_abort("Corrupt file! Delete the file in {.file {train_tar}} and try again.")
        utils::untar(train_tar, exdir = self$base_dir)
      }
    }

    if (self$split == "test") {
      if (!fs::dir_exists(self$test_dir)) {
        test_tar <- download_and_cache(self$test_url, prefix = "places365")
        if (!is.na(self$test_md5) && tools::md5sum(test_tar) != self$test_md5)
          cli_abort("Corrupt file! Delete the file in {.file {test_tar}} and try again.")
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
