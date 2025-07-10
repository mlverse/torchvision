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
#' # Show class index and label
#' label_idx <- item$y
#' label_name <- ds$classes[label_idx]
#' cat("Label index:", label_idx, "Class name:", label_name, "\n")
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

  resources = list(
    devkit = c(
      "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar",
      "35a0585fee1fa656440f3ab298f8479c"
    ),
    val = c(
      "http://data.csail.mit.edu/places/places365/val_256.tar",
      "e27b17d8d44f4af9a78502beb927f808"
    ),
    train = c(
      "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar",
      "53ca1c756c3d1e7809517cc47c5561c5"
    ),
    test = c(
      "http://data.csail.mit.edu/places/places365/test_256.tar",
      "f532f6ad7b582262a2ec8009075e186b"
    )
  ),

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
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    classes_df <- read.delim(self$categories_file, header = FALSE, sep = " ", col.names = c("category", "index"))
    self$classes <- sub(".*/", "", classes_df$category)
    self$class_to_idx <- setNames(seq_along(self$classes), self$classes)

    if (self$split == "train") {
      ann <- read.delim(self$train_ann, header = FALSE, col.names = c("path", "label"), sep = " ")
      ann$label <- ann$label + 1L
      ann$path <- file.path(self$train_dir, ann$path)
      exist <- fs::file_exists(ann$path)
      self$items <- ann[exist, c("path", "label")]
    } else if (self$split == "val") {
      ann <- read.delim(self$val_ann, header = FALSE, col.names = c("path", "label"), sep = " ")
      ann$label <- ann$label + 1L
      ann$path <- file.path(self$val_dir, ann$path)
      exist <- fs::file_exists(ann$path)
      self$items <- ann[exist, c("path", "label")]
    } else if (self$split == "test") {
      self$files <- fs::dir_ls(self$test_dir, recurse = FALSE, type = "file", glob = "*.jpg")
    }

    cli_inform("{.cls {class(self)[[1]]}} Split '{self$split}' loaded with {length(self)} samples.")
  },

  .length = function() {
    if (self$split == "test")
      length(self$files)
    else
      nrow(self$items)
  },

  .getitem = function(index) {
    if (self$split == "test") {
      path <- self$files[[index]]
      x <- magick::image_read(path)
      if (!is.null(self$transform))
        x <- self$transform(x)
      list(x = x, y = NA_integer_)
    } else if (self$split %in% c("train", "val")) {
      item <- self$items[index, ]
      x <- magick::image_read(item$path)
      if (!is.null(self$transform))
        x <- self$transform(x)
      y <- item$label
      if (!is.null(self$target_transform))
        y <- self$target_transform(y)
      list(x = x, y = y)
    } else {
      cli_abort("Invalid split: {self$split}")
    }
  },

  download = function() {
    if (self$check_exists())
      return()

    cli_inform("Downloading {.cls {class(self)[[1]]}} split '{self$split}'...")
    fs::dir_create(self$base_dir, recurse = TRUE)

    needed <- switch(
      self$split,
      train = c("devkit", "train"),
      val = c("devkit", "val"),
      test = "test"
    )

    for (n in needed) {
      res <- self$resources[[n]]

      # Use extended timeout only for the large train archive
      tar <- if (n == "train") {
        withr::with_options(list(timeout = 6000),
                            download_and_cache(res[1], prefix = "places365")
        )
      } else {
        download_and_cache(res[1], prefix = "places365")
      }

      if (!is.na(res[2]) && tools::md5sum(tar) != res[2])
        cli_abort("Corrupt file! Delete the file in {.file {tar}} and try again.")

      utils::untar(tar, exdir = self$base_dir)
    }
  },

  check_exists = function() {
    if (self$split == "train")
      fs::file_exists(self$train_ann) && fs::dir_exists(self$train_dir)
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
    val_ann = function() file.path(self$base_dir, "places365_val.txt"),
    train_ann = function() file.path(self$base_dir, "places365_train_standard.txt"),
    categories_file = function() file.path(self$base_dir, "categories_places365.txt")
  )
)
