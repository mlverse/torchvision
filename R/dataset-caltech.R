#' Caltech Datasets
#'
#' The Caltech-101 and Caltech-256 collections are **classification** datasets
#' made of color images with varying sizes. They cover 101 and 256 object
#' categories respectively and are commonly used for evaluating visual
#' recognition models.
#'
#' @inheritParams fgvc_aircraft_dataset
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/caltech-101`.
#'
#' @return An object of class \code{caltech101_dataset}, which behaves like a torch dataset.
#' Each element is a named list with:
#' - `x`: A H x W x 3 integer array representing an RGB image.
#' - `y`: An Integer representing the label.
#'
#' @details
#' The Caltech-101 dataset contains around 9,000 images
#' spread over 101 object categories plus a background class. Images
#' have varying sizes.
#'
#' Caltech-256 extends this to about 30,000 images
#' across 256 categories.
#'
#' @examples
#' \dontrun{
#' caltech101 <- caltech101_dataset(download = TRUE)
#'
#' first_item <- caltech101[1]
#' first_item$x  # Image array
#' first_item$y  # Integer label
#' }
#'
#' @name caltech_dataset
#' @title Caltech Datasets
#' @rdname caltech_dataset
#' @family classification_dataset
#' @export
caltech101_dataset <- torch::dataset(
  name = "caltech-101",
  subname = "101_ObjectCategories",
  archive_size = 0.13,
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
    self$root <- fs::path(root, class(self)[[1]])
    self$transform <- transform
    self$target_transform <- target_transform

    cli_inform("{.cls {class(self)[[1]]}} Dataset (~{.emph {self$archive_size}} GB) will be downloaded and processed if not already available.")

    if (download)
      self$download()

    if (!self$check_exists())
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")

    obj_dir <- fs::path(self$root, class(self)[[1]], self$subname)
    all_dirs <- fs::dir_ls(obj_dir, type = "directory")
    self$classes <- sort(base::basename(all_dirs))
    self$classes <- self$classes[self$classes != "BACKGROUND_Google"]

    name_map <- list("Faces" = "Faces_2", "Faces_easy" = "Faces_3", "Motorbikes" = "Motorbikes_16", "airplanes" = "Airplanes_Side_2")
    self$annotation_classes <- vapply(self$classes, function(x) if (x %in% names(name_map)) name_map[[x]] else x, character(1))

    for (i in seq_along(self$classes)) {
      img_dir <- fs::path(obj_dir, self$classes[[i]])
      imgs <- sort(fs::dir_ls(img_dir, glob = "*.jpg"))
      self$img_path <- append(self$img_path, imgs)
      self$labels <- c(self$labels, rep(i, length(imgs)))
      self$image_indices <- c(self$image_indices, seq_along(imgs))
    }

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$img_path)} images across {length(self$classes)} classes.")
  },

  .getitem = function(index) {
    img_path <- self$img_path[[index]]
    y <- self$labels[[index]]

    x <- jpeg::readJPEG(img_path)

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
  },

  .length = function() {
    length(self$img_path)
  },

  check_exists = function() {
    fs::dir_exists(fs::path(self$root, class(self)[[1]], self$subname))
  },

  download = function() {

    if (self$check_exists())
      return()

    fs::dir_create(self$root)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    invisible(lapply(self$resources, function(res) {
      archive <- download_and_cache(res$url, prefix = class(self)[1])
      dest <- fs::path(self$root, fs::path_file(res$filename))
      fs::file_move(archive, dest)
      md5 <- tools::md5sum(dest)[[1]]
      if (md5 != res$md5)
        cli_abort("Corrupt file! Delete the file in {.file {archive}} and try again.")
      if(class(self)[1] == "caltech-101")
        utils::unzip(dest, exdir = self$root)
      else
        utils::untar(dest, exdir = self$root)

      extracted <- fs::path(self$root, class(self)[[1]])
      if (fs::file_exists(fs::path(extracted, "101_ObjectCategories.tar.gz")))
        utils::untar(fs::path(extracted, "101_ObjectCategories.tar.gz"), exdir = extracted)
    }))

    cli_inform("{.cls {class(self)[[1]]}} dataset downloaded and extracted successfully.")
  }
)

#' Caltech-256 Dataset
#'
#' Loads the Caltech-256 Object Category Dataset for image classification. It consists of 30,607 images across 256 distinct object categories.
#' Each category has at least 80 images, with variability in image size.
#'
#' @inheritParams fgvc_aircraft_dataset
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/caltech256`.
#'
#' @return An object of class \code{caltech256_dataset}, which behaves like a torch dataset.
#' Each element is a named list with:
#' - `x`: A H x W x 3 integer array representing an RGB image.
#' - `y`: An Integer representing the label.
#'
#' @rdname caltech_dataset
#' @export
caltech256_dataset <- torch::dataset(
  name = "caltech256",
  subname = "256_ObjectCategories",
  inherit = caltech101_dataset,
  classes = NULL,
  archive_size = 1.12,
  resources = list(
    list(
      url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar",
      filename = "256_ObjectCategories.tar",
      md5 = "67b4f42ca05d46448c6bb8ecd2220f6d"
    )
  ),

  initialize = function(
    root = tempdir(),
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
  self$root <- fs::path(root, class(self)[[1]])
  self$transform <- transform
  self$target_transform <- target_transform

  cli_inform("{.cls {class(self)[[1]]}} Dataset (~{.emph {self$archive_size}} GB) will be downloaded and processed if not already cached.")

  if (download) {
    self$download()
  }
  if (!self$check_exists()) {
    cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
  }

  obj_dir <- fs::path(self$root, self$subname)
  all_dirs <- fs::dir_ls(obj_dir, type = "directory")
  self$classes <- sort(base::basename(all_dirs))
  self$classes <- self$classes[self$classes != "BACKGROUND_Google"]

  class_dirs <- fs::path(self$root, self$subname, self$classes)
  self$classes <- sub("^\\d+\\.", "", self$classes)
  images_per_class <- lapply(class_dirs, function(class_dir) {
    imgs <- fs::dir_ls(class_dir, glob = "*.jpg")
    sort(imgs)
  })
  self$img_path <- unlist(images_per_class, use.names = FALSE)
  self$labels <- unlist(
    mapply(function(i, imgs) {
      rep(i, length(imgs))
    }, seq_along(self$classes), images_per_class, SIMPLIFY = FALSE),
    use.names = FALSE
  )
  cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$img_path)} images across {length(self$classes)} classes.")
  },

  check_exists = function() {
    fs::dir_exists(fs::path(self$root, self$subname))
  }
)
