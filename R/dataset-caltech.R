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
    rlang::inform("Caltech101 Dataset (~131MB) will be downloaded and processed if not already available.")
    if (is.character(target_type))
      target_type <- list(target_type)
    valid_types <- c("category", "annotation")
    invalid <- setdiff(target_type, valid_types)
    if (length(invalid) > 0) {
      invalid_str <- glue::glue_collapse(invalid, sep = ", ", last = " and ")
      runtime_error(glue::glue("Invalid target_type(s): {invalid_str}"))
    }
    self$target_type <- target_type

    if (download) {
      rlang::inform("Downloading and preparing the Caltech101 dataset...")
      self$download()
    }
    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to download.")
    rlang::inform("Loading Caltech101 image paths and labels...")
    all_dirs <- fs::dir_ls(fs::path(self$root, "caltech-101", "101_ObjectCategories"), type = "directory")
    self$classes <- sort(base::basename(all_dirs))
    self$classes <- self$classes[self$classes != "BACKGROUND_Google"]
    name_map <- list("Faces"="Faces_2", "Faces_easy"="Faces_3", "Motorbikes"="Motorbikes_16", "airplanes"="Airplanes_Side_2")
    self$annotation_classes <- vapply(self$classes, function(x) if (x %in% names(name_map)) name_map[[x]] else x, character(1))

    samples_per_class <- lapply(seq_along(self$classes), function(i) {
      images <- fs::dir_ls(fs::path(self$root, "caltech-101", "101_ObjectCategories", self$classes[[i]]), glob = "*.jpg")
      images <- sort(images)
      list(samples = images, labels = rep(i, length(images)), indices = seq_along(images))
    })

    self$samples <- unlist(lapply(samples_per_class, `[[`, "samples"), recursive = FALSE)
    self$labels <- unlist(lapply(samples_per_class, `[[`, "labels"))
    self$image_indices <- unlist(lapply(samples_per_class, `[[`, "indices"))

    rlang::inform(glue::glue("Caltech101 dataset loaded with {length(self$samples)} images across {length(self$classes)} classes."))
  },
  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]

    img <- magick::image_read(img_path)
    img <- magick::image_resize(img, "224x224!")
    img_tensor <- torchvision::transform_to_tensor(img)

    target_list <- lapply(self$target_type, function(t) {
      if (t == "category") {
        label
      } else if (t == "annotation") {
        ann_class <- self$annotation_classes[label_idx]
        index_str <- formatC(self$image_indices[[index]], width = 4, flag = "0")
        ann_file <- fs::path(self$root, "caltech-101", "Annotations", ann_class, glue::glue("annotation_{index_str}.mat"))
        if (!fs::file_exists(ann_file)) {
          NULL
        } else {
          if (!requireNamespace("R.matlab", quietly = TRUE)) {
            runtime_error("Package 'R.matlab' is needed for this dataset. Please install it.")
          }
          mat_data <- R.matlab::readMat(as.character(ann_file))
          box_coord <- as.numeric(mat_data[["box.coord"]])
          obj_contour <- mat_data[["obj.contour"]] |>
            as.matrix() |>
            apply(2, as.numeric) |>
            t()
          list(box_coord = box_coord, obj_contour = obj_contour)
        }
      } else {
        runtime_error(glue::glue("Invalid target_type: {t}"))
      }
    })
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
    if (self$check_exists()) return()
    rlang::inform("Downloading Caltech101 archive...")
    fs::dir_create(self$root)
    invisible(lapply(self$resources, function(res) {
      zip_path <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- fs::path(self$root, fs::path_file(res$filename))
      fs::file_copy(zip_path, dest, overwrite = TRUE)
      md5_actual <- tools::md5sum(dest)[[1]]
      if (md5_actual != res$md5) {
        runtime_error(glue::glue("MD5 mismatch for file: {res$filename} (expected {res$md5}, got {md5_actual})"))
      }
      rlang::inform("Extracting main archive...")
      utils::unzip(dest, exdir = self$root)
      extracted_dir <- fs::path(self$root, "caltech-101")
      tar_gz_path <- fs::path(extracted_dir, "101_ObjectCategories.tar.gz")
      if (fs::file_exists(tar_gz_path)) {
        rlang::inform("Extracting 101_ObjectCategories...")
        utils::untar(tar_gz_path, exdir = extracted_dir)
      } else {
        runtime_error("Expected 101_ObjectCategories.tar.gz not found after unzip.")
      }
      annotations_path <- fs::path(extracted_dir, "Annotations.tar")
      if (fs::file_exists(annotations_path)) {
        rlang::inform("Extracting Annotations...")
        utils::untar(annotations_path, exdir = extracted_dir)
      }
    }))
    rlang::inform("Caltech101 dataset downloaded and extracted successfully.")
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
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/caltech256`.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform labels.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @return An R6 dataset object inheriting from `dataset`. Each item is a named list with elements:
#' \describe{
#'   \item{x}{An image tensor of shape `(3, H, W)` with values in `[0, 1]`.}
#'   \item{y}{The class label as a character string.}
#' }
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' caltech <- caltech256_dataset(root = root_dir, download = TRUE)
#' first_item <- caltech[1]
#' first_item$x
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
  
  initialize = function(root = tempdir(), transform = NULL, target_transform = NULL, download = FALSE) {
    self$root <- fs::path(root, "caltech256")
    self$transform <- transform
    self$target_transform <- target_transform

    rlang::inform("Caltech256 Dataset (~1.2GB) will be downloaded and processed if not already cached.")

    if (download) {
      rlang::inform("Downloading Caltech256 dataset archive...")
      self$download()
    }
    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use `download = TRUE` to download.")
    }
    rlang::inform("Preparing file list and labels...")

    all_dirs <- fs::dir_ls(fs::path(self$root, "256_ObjectCategories"), type = "directory")
    self$classes <- sort(fs::path_file(all_dirs))

    class_dirs <- fs::path(self$root, "256_ObjectCategories", self$classes)
    images_per_class <- lapply(class_dirs, function(class_dir) {
      imgs <- fs::dir_ls(class_dir, glob = "*.jpg")
      sort(imgs)
    })
    self$samples <- unlist(images_per_class, use.names = FALSE)
    self$labels <- unlist(
      mapply(function(i, imgs) {
        rep(i, length(imgs))
      }, seq_along(self$classes), images_per_class, SIMPLIFY = FALSE),
      use.names = FALSE
    )
    rlang::inform(glue::glue("Loaded {length(self$samples)} samples across {length(self$classes)} classes."))
  },
  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]

    img <- magick::image_read(img_path)
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
    if (self$check_exists()) return()
    fs::dir_create(self$root)
    rlang::inform("Starting download and integrity check...")
    lapply(self$resources, function(res) {
      tar_path <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- fs::path(self$root, fs::path_file(res$filename))
      fs::file_copy(tar_path, dest, overwrite = TRUE)
      md5_actual <- tools::md5sum(dest)[[1]]
      if (md5_actual != res$md5) {
        runtime_error(glue::glue("MD5 mismatch for file: {res$filename} (expected {res$md5}, got {md5_actual})"))
      }

      rlang::inform("Extracting dataset (this may take a moment)...")
      utils::untar(dest, exdir = self$root)
    })
    rlang::inform("Caltech256 dataset downloaded and extracted successfully!")
  },
  check_exists = function() {
    fs::dir_exists(fs::path(self$root, "256_ObjectCategories"))
  }
)