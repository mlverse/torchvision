#' Caltech-101 Dataset
#'
#' Loads the Caltech-101 dataset consisting of 101 object categories, with images of varied dimensions.
#' Each class contains between 40 and 800 images. Optional annotations include bounding box coordinates and object contours.
#'
#'
#' @param root Character. Root directory for dataset storage. Default is a temporary directory. Data will be stored under `root/caltech101`.
#' @param transform Optional function to apply to each image (e.g., resizing or normalization).
#' @param target_transform Optional function to transform the target. Default is `NULL`.
#' @param download Logical. Whether to download and extract the dataset if not already available. Default is `FALSE`.
#'
#' @return An object of class \code{caltech101_dataset}, which behaves like a torch dataset.
#' Each element is a named list:
#' - `x`: a 3 x W x H integer array representing an RGB image.
#' - `y`: either a character label, annotation list, or both depending on `target_type`.
#'
#' @examples
#' \dontrun{
#' # Category-only target
#' ds1 <- caltech101_dataset(download = TRUE)
#' first_item <- ds1[1]
#' first_item$x  # RGB image array
#' first_item$y  # e.g., "accordion"
#'
#' # Annotation-only target
#' ds2 <- caltech101_dataset(download = TRUE)
#' first_item <- ds2[1]
#' first_item$x  # RGB image array
#' first_item$y$box_coord      # Numeric vector: [x1, y1, x2, y2]
#' first_item$y$obj_contour    # Matrix of contour points: N x 2
#'
#' # Category + Annotation target (All target)
#' ds3 <- caltech101_dataset(download = TRUE)
#' first_item <- ds3[1]
#' first_item$x  # RGB image array
#' first_item$y$label          # e.g., "accordion"
#' first_item$y$box_coord      # Numeric vector: [x1, y1, x2, y2]
#' first_item$y$obj_contour    # Matrix of contour points: N x 2
#' }
#'
#' @name caltech101_dataset
#' @aliases caltech101_dataset
#' @title Caltech-101 Dataset
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

  initialize = function(
    root = tempdir(),
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$root <- fs::path(root, "caltech101")
    self$transform <- transform
    self$target_transform <- target_transform
    rlang::inform("Caltech101 Dataset (~131MB) will be downloaded and processed if not already available.")

    if (download) {
      self$download()
    }
    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to download.")

    all_dirs <- fs::dir_ls(fs::path(self$root, "caltech-101", "101_ObjectCategories"), type = "directory")
    self$classes <- sort(base::basename(all_dirs))
    self$classes <- self$classes[self$classes != "BACKGROUND_Google"]

    name_map <- list("Faces"="Faces_2", "Faces_easy"="Faces_3", "Motorbikes"="Motorbikes_16", "airplanes"="Airplanes_Side_2")
    self$annotation_classes <- vapply(self$classes, function(x) if (x %in% names(name_map)) name_map[[x]] else x, character(1))

    self$samples <- list()
    self$labels <- c()
    self$image_indices <- c()

    for (i in seq_along(self$classes)) {
      img_dir <- fs::path(self$root, "caltech-101", "101_ObjectCategories", self$classes[[i]])
      imgs <- fs::dir_ls(img_dir, glob = "*.jpg")
      imgs <- sort(imgs)
      self$samples <- append(self$samples, imgs)
      self$labels <- c(self$labels, rep(i, length(imgs)))
      self$image_indices <- c(self$image_indices, seq_along(imgs))
    }

    rlang::inform(glue::glue("Caltech101 dataset loaded with {length(self$samples)} images across {length(self$classes)} classes."))
  },

  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[[label_idx]]

    img <- jpeg::readJPEG(img_path)
    img <- img * 255
    img <- aperm(img, c(3, 1, 2))

    if (!is.null(self$transform))
      img <- self$transform(img)

    y <- switch(self$target_type,
      "category" = label,
      "annotation" = {
        ann_class <- self$annotation_classes[[label_idx]]
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
          obj_contour <- t(apply(as.matrix(mat_data[["obj.contour"]]), 2, as.numeric))
          list(box_coord = box_coord, obj_contour = obj_contour)
        }
      },
      "all" = {
        ann_class <- self$annotation_classes[[label_idx]]
        index_str <- formatC(self$image_indices[[index]], width = 4, flag = "0")
        ann_file <- fs::path(self$root, "caltech-101", "Annotations", ann_class, glue::glue("annotation_{index_str}.mat"))

        if (!fs::file_exists(ann_file)) {
          box_coord <- NULL
          obj_contour <- NULL
        } else {
          if (!requireNamespace("R.matlab", quietly = TRUE)) {
            runtime_error("Package 'R.matlab' is needed for this dataset. Please install it.")
          }
          mat_data <- R.matlab::readMat(as.character(ann_file))
          box_coord <- as.numeric(mat_data[["box.coord"]])
          obj_contour <- t(apply(as.matrix(mat_data[["obj.contour"]]), 2, as.numeric))
        }

        list(
          label = label,
          box_coord = box_coord,
          obj_contour = obj_contour
        )
      }
    )

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = img, y = y)
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

#' Caltech-256 Dataset
#'
#' Loads the Caltech-256 Object Category Dataset, which consists of 30,607 images from 256 distinct object categories.
#' Each category has at least 80 images, with significant variability in object position, scale, and background.
#' #'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/caltech256`.
#' @param transform Optional function to apply to each image after loading (e.g., resizing, normalization).
#' @param target_transform Optional function to transform the target label.
#' @param download Logical. If `TRUE`, downloads and extracts the dataset if it's not already present. Default is `FALSE`.
#'
#' @return An object of class \code{caltech256_dataset}, which behaves like a torch dataset.
#' Each element is a named list:
#' \describe{
#'   \item{x}{A 3 x W x H integer array representing an RGB image.}
#'   \item{y}{A character string representing the class label.}
#' }
#'
#' @examples
#' \dontrun{
#' caltech256 <- caltech256_dataset(download = TRUE)
#' 
#' first_item <- caltech256[1]
#' first_item$x  # Image array
#' first_item$y  # Class label, e.g., "ak47"
#' }
#'
#' @name caltech256_dataset
#' @aliases caltech256_dataset
#' @title Caltech-256 Object Category Dataset
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
      self$download()
    }
    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use `download = TRUE` to download.")
    }

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
    rlang::inform(glue::glue("Caltech256 dataset loaded with {length(self$samples)} images across {length(self$classes)} classes."))
  },
  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]
    label <- substring(label,5)

    img <- jpeg::readJPEG(img_path)
    img <- img * 255
    img <- aperm(img, c(3, 1, 2))

    if (!is.null(self$transform))
      img <- self$transform(img)
    if (!is.null(self$target_transform))
      label <- self$target_transform(label)

    list(x = img, y = label)
  },
  .length = function() {
    length(self$samples)
  },
  download = function() {
    if (self$check_exists()) return()
    fs::dir_create(self$root)
    lapply(self$resources, function(res) {
      tar_path <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- fs::path(self$root, fs::path_file(res$filename))
      fs::file_copy(tar_path, dest, overwrite = TRUE)
      md5_actual <- tools::md5sum(dest)[[1]]
      if (md5_actual != res$md5) {
        runtime_error(glue::glue("MD5 mismatch for file: {res$filename} (expected {res$md5}, got {md5_actual})"))
      }

      utils::untar(dest, exdir = self$root)
    })
    rlang::inform("Caltech256 dataset downloaded and extracted successfully!")
  },
  check_exists = function() {
    fs::dir_exists(fs::path(self$root, "256_ObjectCategories"))
  }
)