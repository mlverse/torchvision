#' Caltech-101 Detection Dataset
#'
#' Loads the Caltech-101 dataset with image-level labels, bounding boxes, and object contours.
#'
#' @inheritParams coco_detection_dataset
#' @param transform Optional transform function applied to the image.
#' @param download Logical. If TRUE, downloads and extracts the dataset if not already present in \code{root}.
#'
#' @return
#' A torch dataset. Each item is a list with two elements:
#' \describe{
#'   \item{x}{A 3D \code{torch_tensor} of shape \code{(3, H, W)} representing the image in RGB format.}
#'   \item{y}{A list with:
#'     \describe{
#'       \item{boxes}{A 2D \code{torch_tensor} of shape \code{(1, 4)} containing the bounding box in \code{xywh} format.}
#'       \item{labels}{A character scalar giving the class label for the image.}
#'       \item{contour}{A 2D \code{torch_tensor} of shape \code{(N, 2)} with the object contour coordinates.}
#'     }
#'   }
#' }
#'
#' @examples
#' \dontrun{
#' caltech101 <- caltech101_dataset(download = TRUE)
#' first_item <- caltech101[1]
#'
#' first_item$x <- torch_tensor(first_item$x, dtype = torch::torch_uint8())
#'
#' # Draw bounding box
#' bboxes <- draw_bounding_boxes(
#'   image = first_item$x,
#'   boxes = first_item$y$boxes,
#'   labels = caltech101$classes[first_item$y$labels],
#'   colors = "red"
#' )
#' tensor_image_browse(bboxes)
#'
#' # Draw contour points as keypoints
#' contour_points <- draw_keypoints(
#'   image = image,
#'   keypoints = first_item$y$contour,
#'   colors = "green",
#'   radius = 2
#' )
#' tensor_image_browse(contour_points)
#' }
#'
#' @name caltech101_dataset
#' @aliases caltech101_dataset
#' @title Caltech-101 Dataset
#' @export
caltech101_dataset <- dataset(
  name = "caltech101",
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

    cli::cli_inform("{.cls {class(self)[[1]]}} Dataset (~130MB) will be downloaded and processed if not already available.")

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    obj_dir <- fs::path(self$root, "caltech-101", "101_ObjectCategories")
    all_dirs <- fs::dir_ls(obj_dir, type = "directory")
    self$classes <- sort(base::basename(all_dirs))
    self$classes <- self$classes[self$classes != "BACKGROUND_Google"]

    name_map <- list("Faces" = "Faces_2", "Faces_easy" = "Faces_3", "Motorbikes" = "Motorbikes_16", "airplanes" = "Airplanes_Side_2")
    self$annotation_classes <- vapply(self$classes, function(x) if (x %in% names(name_map)) name_map[[x]] else x, character(1))

    self$samples <- list()
    self$labels <- c()
    self$image_indices <- c()

    for (i in seq_along(self$classes)) {
      img_dir <- fs::path(obj_dir, self$classes[[i]])
      imgs <- sort(fs::dir_ls(img_dir, glob = "*.jpg"))
      self$samples <- append(self$samples, imgs)
      self$labels <- c(self$labels, rep(i, length(imgs)))
      self$image_indices <- c(self$image_indices, seq_along(imgs))
    }

    cli::cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$samples)} images across {length(self$classes)} classes.")
  },

  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]

    img <- jpeg::readJPEG(img_path)
    img <- img * 255
    img <- aperm(img, c(3, 1, 2))

    ann_class <- self$annotation_classes[[label_idx]]
    index_str <- formatC(self$image_indices[[index]], width = 4, flag = "0")
    ann_file <- fs::path(self$root, "caltech-101", "Annotations", ann_class, glue::glue("annotation_{index_str}.mat"))

    if (!fs::file_exists(ann_file))
      runtime_error("Annotation file not found: {ann_file}")

    if (!requireNamespace("R.matlab", quietly = TRUE))
      runtime_error("Please install 'R.matlab' to read annotation files.")

    mat_data <- R.matlab::readMat(as.character(ann_file))
    coords <- as.numeric(mat_data[["box.coord"]])
    box_coord <- torch_tensor(c(coords[1], coords[3], coords[2], coords[4]), dtype = torch_float())$unsqueeze(1)
    box_coord <- box_convert(box_coord, in_fmt = "xywh", out_fmt = "xyxy")

    contour <- torch_tensor(t(apply(as.matrix(mat_data[["obj.contour"]]), 2, as.numeric)), dtype = torch_float())$unsqueeze(1)

    target <- list(
      boxes = box_coord,
      labels = label_idx,
      contour = contour
    )

    if (!is.null(self$transform))
      img <- self$transform(img)

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img, y = target)
  },

  .length = function() {
    length(self$samples)
  },

  check_exists = function() {
    fs::dir_exists(fs::path(self$root, "caltech-101", "101_ObjectCategories"))
  },

  download = function() {
    if (self$check_exists()) return()
    fs::dir_create(self$root)

    cli::cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    invisible(lapply(self$resources, function(res) {
      archive <- download_and_cache(res$url, prefix = "caltech101")
      dest <- fs::path(self$root, fs::path_file(res$filename))
      fs::file_copy(archive, dest, overwrite = TRUE)
      md5 <- tools::md5sum(dest)[[1]]
      if (md5 != res$md5)
        runtime_error("Corrupt file! Delete the file in {.file {archive}} and try again.")
      utils::unzip(dest, exdir = self$root)

      extracted <- fs::path(self$root, "caltech-101")
      if (fs::file_exists(fs::path(extracted, "101_ObjectCategories.tar.gz")))
        utils::untar(fs::path(extracted, "101_ObjectCategories.tar.gz"), exdir = extracted)
      if (fs::file_exists(fs::path(extracted, "Annotations.tar")))
        utils::untar(fs::path(extracted, "Annotations.tar"), exdir = extracted)
    }))

    cli::cli_inform("{.cls {class(self)[[1]]}} dataset downloaded and extracted successfully.")
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

    cli::cli_inform("Caltech256 Dataset (~1.2GB) will be downloaded and processed if not already cached.")

    if (download) {
      self$download()
    }
    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
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
    cli::cli_inform("Caltech256 dataset loaded with {length(self$samples)} images across {length(self$classes)} classes.")
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
      archive <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- fs::path(self$root, fs::path_file(res$filename))
      fs::file_copy(archive, dest, overwrite = TRUE)
      md5_actual <- tools::md5sum(dest)[[1]]
      if (md5_actual != res$md5) {
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")
      }

      utils::untar(dest, exdir = self$root)
    })
    cli::cli_inform("Caltech256 dataset downloaded and extracted successfully!")
  },
  check_exists = function() {
    fs::dir_exists(fs::path(self$root, "256_ObjectCategories"))
  }
)