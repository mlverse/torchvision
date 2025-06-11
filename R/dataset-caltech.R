#' Caltech 101 Dataset
#'
#' Loads the Caltech 101 Dataset, containing images from 101 object categories.
#' The dataset includes images and optionally their object annotations.
#' There is no built-in train/test split.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/caltech101`.
#' @param target_type Character or character vector. Type of target to load. Either `"category"`, `"annotation"`, or both.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform labels or annotations.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @return An R6 dataset object inheriting from `dataset`. Each item is a named list with elements:
#' \describe{
#'   \item{x}{An image tensor of shape `(3, H, W)` with values in `[0, 1]`.}
#'   \item{y}{The target label. If `target_type = "category"`, this is the class name (character string).
#'            If `target_type = "annotation"`, this is a list containing:
#'            \describe{
#'              \item{box_coord}{Numeric vector of length 4 with bounding box coordinates.}
#'              \item{obj_contour}{Matrix of shape `(N, 2)` with object contour coordinates.}
#'            }}
#' }
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' caltech <- caltech101_dataset(root = root_dir, target_type = "category", download = TRUE)
#' first_item <- caltech[1]
#' # image tensor of first item
#' first_item$x
#' # label (class name) of first item
#' first_item$y
#' 
#' # If annotations are requested
#' caltech_ann <- caltech101_dataset(root = root_dir, target_type = "annotation", download = FALSE)
#' annotation_item <- caltech_ann[1]
#' annotation_item$y$box_coord
#' annotation_item$y$obj_contour
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

  initialize = function(root = tempdir(),
                        target_type = "category",
                        transform = NULL,
                        target_transform = NULL,
                        download = FALSE) {

    self$root <- fs::path(root, "caltech101")
    self$transform <- transform
    self$target_transform <- target_transform
    self$resize_shape <- c(224, 224)

    if (is.character(target_type))
      target_type <- list(target_type)
    valid_types <- c("category", "annotation")
    for (t in target_type) {
      if (!(t %in% valid_types))
        runtime_error(sprintf("Invalid target_type: %s", t))
    }
    self$target_type <- target_type

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to download.")

    all_dirs <- fs::dir_ls(fs::path(self$root, "caltech-101", "101_ObjectCategories"), type = "directory")
    self$classes <- sort(base::basename(all_dirs))
    self$classes <- self$classes[self$classes != "BACKGROUND_Google"]
    name_map <- list("Faces"="Faces_2", "Faces_easy"="Faces_3", "Motorbikes"="Motorbikes_16", "airplanes"="Airplanes_Side_2")
    self$annotation_classes <- vapply(self$classes, function(x) if (x %in% names(name_map)) name_map[[x]] else x, character(1))
    self$samples <- list()
    self$labels <- integer()
    self$image_indices <- integer()
    for (i in seq_along(self$classes)) {
      class_dir <- fs::path(self$root, "caltech-101", "101_ObjectCategories", self$classes[[i]])
      images <- fs::dir_ls(class_dir, glob = "*.jpg")
      images <- sort(images)
      self$samples <- c(self$samples, images)
      self$labels <- c(self$labels, rep(i, length(images)))
      self$image_indices <- c(self$image_indices, seq_along(images))
    }
  },
  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]

    img <- magick::image_read(img_path)
    img <- magick::image_resize(img, "224x224!")
    img_tensor <- torchvision::transform_to_tensor(img)
    
    target_list <- list()
    for (t in self$target_type) {
      if (t == "category") {
        target_list <- c(target_list, label)
      } else if (t == "annotation") {
        ann_class <- self$annotation_classes[label_idx]
        ann_file <- fs::path(self$root, "caltech-101", "Annotations", ann_class,
                              sprintf("annotation_%04d.mat", self$image_indices[[index]]))
        if (!fs::file_exists(ann_file)) {
          target_list <- c(target_list, NULL)
        } else {
          scipy <- reticulate::import("scipy.io")
          mat_data <- scipy$loadmat(as.character(ann_file))
          box_coord <- mat_data[["box_coord"]]
          box_coord <- as.numeric(box_coord)
          obj_contour <- mat_data[["obj_contour"]]
          obj_contour <- as.matrix(obj_contour)
          obj_contour <- apply(obj_contour, 2, as.numeric)
          obj_contour <- t(obj_contour)
          annotation <- list(box_coord = box_coord, obj_contour = obj_contour)
          target_list <- c(target_list, list(annotation))
        }
      }
    }
    target <- if (length(target_list) == 1) target_list[[1]] else target_list
    if (!is.null(self$transform))
      img_tensor <- self$transform(img_tensor)
    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img_tensor, y = target)
  },
  .length = function() {
    length(self$samples)
  },
  download = function() {
    rlang::inform("Downloading Caltech101 Dataset...")
    if (self$check_exists()) return()
    fs::dir_create(self$root)
    for (res in self$resources) {
      zip_path <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- fs::path(self$root, fs::path_file(res$filename))
      fs::file_copy(zip_path, dest, overwrite = TRUE)
      md5_actual <- tools::md5sum(dest)[[1]]
      if (md5_actual != res$md5)
        runtime_error(sprintf("MD5 mismatch for file: %s (expected %s, got %s)", res$filename, res$md5, md5_actual))
      rlang::inform("Extracting archive and processing metadata...")
      utils::unzip(dest, exdir = self$root)
      extracted_dir <- fs::path(self$root, "caltech-101")
      tar_gz_path <- fs::path(extracted_dir, "101_ObjectCategories.tar.gz")
      if (fs::file_exists(tar_gz_path)) {
        utils::untar(tar_gz_path, exdir = extracted_dir)
      } else {
        runtime_error("Expected 101_ObjectCategories.tar.gz not found after unzip.")
      }
      annotations_path <- fs::path(extracted_dir, "Annotations.tar")
      if (fs::file_exists(annotations_path)) {
        utils::untar(annotations_path, exdir = extracted_dir)
      }
    }
    rlang::inform("Dataset Caltech101 processed successfully!")
  },
  check_exists = function() {
    fs::dir_exists(fs::path(self$root, "caltech-101", "101_ObjectCategories"))
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
  initialize = function(root = tempdir(), transform = NULL, target_transform = NULL, download = FALSE, image_size = c(224, 224)) {
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
