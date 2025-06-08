#' Caltech 101 Dataset
#'
#' Loads the Caltech 101 Dataset, containing images from 101 object categories.
#' The dataset includes images and their corresponding class labels.
#' There is no built-in train/test split.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/caltech101`.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform labels.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#' @param image_size Integer vector of length 2. Size to resize images to (width, height). Default is `c(224, 224)`.
#'
#' @return A caltech101_dataset object representing the dataset.
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' caltech <- caltech101_dataset(root = root_dir, download = TRUE)
#' first_item <- caltech[1]
#' # image tensor of first item
#' first_item$x
#' # label (class name) of first item
#' first_item$y
#' }
#'
#' @name caltech101_dataset
#' @aliases caltech101_dataset
#' @title Caltech 101 Dataset
#' @export
caltech101_dataset <- dataset(
  name = "caltech101",
  classes = NULL,
  resources = list(
    list(
      url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
      filename = "caltech-101.zip",
      md5 = "3138e1922a9193bfa496528edbbc45d0"
    )
  ),
  initialize = function(root = rappdirs::user_cache_dir("torch"), transform = NULL, target_transform = NULL, download = FALSE, image_size = c(224, 224)) {
    self$root <- file.path(root, "caltech101")
    self$transform <- transform
    self$target_transform <- target_transform
    self$image_size <- image_size

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to download.")

    all_dirs <- fs::dir_ls(file.path(self$root, "caltech-101", "101_ObjectCategories"), type = "directory")
    self$classes <- sort(basename(all_dirs))
    self$classes <- self$classes[self$classes != "BACKGROUND_Google"]

    self$samples <- list()
    self$labels <- integer()

    for (i in seq_along(self$classes)) {
      class_dir <- file.path(self$root, "caltech-101", "101_ObjectCategories", self$classes[[i]])
      images <- fs::dir_ls(class_dir, glob = "*.jpg")
      self$samples <- c(self$samples, images)
      self$labels <- c(self$labels, rep(i, length(images)))
    }
  },
  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]

    img <- magick::image_read(img_path)
    img <- magick::image_resize(img, paste(self$image_size, collapse = "x"))
    img_tensor <- torchvision::transform_to_tensor(img)

    if (!is.null(self$transform))
      img_tensor <- self$transform(img_tensor)
    if (!is.null(self$target_transform))
      label <- self$target_transform(label)

    structure(list(x = img_tensor, y = label), class = "caltech101_item")
  },
  .length = function() {
    length(self$samples)
  },
  download = function() {
    rlang::inform(glue::glue("Downloading Caltech101 Dataset..."))
    if (self$check_exists()) return()
    fs::dir_create(self$root)
    for (res in self$resources) {
      zip_path <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- file.path(self$root, basename(res$filename))
      fs::file_copy(zip_path, dest, overwrite = TRUE)
      md5_actual <- tools::md5sum(dest)[[1]]
      if (md5_actual != res$md5)
        runtime_error(sprintf("MD5 mismatch for file: %s (expected %s, got %s)", res$filename, res$md5, md5_actual))
      rlang::inform("Extracting archive and processing metadata...")
      utils::unzip(dest, exdir = self$root)
      extracted_dir <- file.path(self$root, "caltech-101")
      tar_gz_path <- file.path(extracted_dir, "101_ObjectCategories.tar.gz")
      if (fs::file_exists(tar_gz_path)) {
        utils::untar(tar_gz_path, exdir = extracted_dir)
      } else {
        runtime_error("Expected 101_ObjectCategories.tar.gz not found after unzip.")
      }
      annotations_path <- file.path(extracted_dir, "Annotations.tar")
      if (fs::file_exists(annotations_path)) {
        utils::untar(annotations_path, exdir = extracted_dir)
      }
    }
    rlang::inform(glue::glue("Dataset Caltech101 processed successfully !"))
  },
  check_exists = function() {
    fs::dir_exists(file.path(self$root, "caltech-101", "101_ObjectCategories"))
  }
)

#' Caltech 256 Dataset
#'
#' Loads the Caltech 256 Dataset, containing images from 256 object categories.
#' The dataset includes images and their corresponding class labels.
#' There is no built-in train/test split.
#'
#' @inheritParams caltech101_dataset
#'
#' @return A caltech256_dataset object representing the dataset.
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' caltech <- caltech256_dataset(root = root_dir, download = TRUE)
#' first_item <- caltech[1]
#' # image tensor of first item
#' first_item$x
#' # label (class name) of first item
#' first_item$y
#' }
#'
#' @name caltech256_dataset
#' @aliases caltech256_dataset
#' @title Caltech 256 Dataset
#' @export
caltech256_dataset <- dataset(
  name = "caltech256",
  classes = NULL,
  resources = list(
    list(
      url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar",
      filename = "256_ObjectCategories.tar",
      md5 = "67b4f42ca05d46448c6bb8ecd2220f6d"
    )
  ),
  initialize = function(root = rappdirs::user_cache_dir("torch"), transform = NULL, target_transform = NULL, download = FALSE, image_size = c(224, 224)) {
    self$root <- file.path(root, "caltech256")
    self$transform <- transform
    self$target_transform <- target_transform
    self$image_size <- image_size

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to download.")

    all_dirs <- fs::dir_ls(file.path(self$root, "256_ObjectCategories"), type = "directory")
    self$classes <- sort(basename(all_dirs))

    self$samples <- list()
    self$labels <- integer()

    for (i in seq_along(self$classes)) {
      class_dir <- file.path(self$root, "256_ObjectCategories", self$classes[[i]])
      images <- fs::dir_ls(class_dir, glob = "*.jpg")
      self$samples <- c(self$samples, images)
      self$labels <- c(self$labels, rep(i, length(images)))
    }
  },
  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]

    img <- magick::image_read(img_path)
    img <- magick::image_resize(img, paste(self$image_size, collapse = "x"))
    img_tensor <- torchvision::transform_to_tensor(img)

    if (!is.null(self$transform))
      img_tensor <- self$transform(img_tensor)
    if (!is.null(self$target_transform))
      label <- self$target_transform(label)

    list(x = img_tensor, y = label)
  },
  .length = function() {
    length(self$samples)
  },
  download = function() {
    rlang::inform(glue::glue("Downloading Caltech256 Dataset..."))
    if (self$check_exists()) return()
    fs::dir_create(self$root)
    for (res in self$resources) {
      tar_path <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- file.path(self$root, basename(res$filename))
      fs::file_copy(tar_path, dest, overwrite = TRUE)
      md5_actual <- tools::md5sum(dest)[[1]]
      if (md5_actual != res$md5)
        runtime_error(sprintf("MD5 mismatch for file: %s (expected %s, got %s)", res$filename, res$md5, md5_actual))

      rlang::inform("Extracting archive and preparing dataset...")
      utils::untar(dest, exdir = self$root)
    }
    rlang::inform(glue::glue("Dataset Caltech256 processed successfully !"))
  },
  check_exists = function() {
    fs::dir_exists(file.path(self$root, "256_ObjectCategories"))
  }
)
