#' FGCV Aircraft Dataset
#'
#' Loads the FGCV Aircraft Dataset, a dataset of aircraft images with multiple annotation levels:
#' - "variant": fine-grained aircraft variants (e.g., specific model variants)
#' - "family": aircraft families
#' - "manufacturer": aircraft manufacturers
#' 
#' The dataset supports different splits: "train", "trainval", and "test".
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/fgvc-aircraft-2013b`.
#' @param split Character. Dataset split to use. One of `"train"`, `"trainval"`, or `"test"`. Default is `"train"`.
#' @param annotation_level Character. Level of annotation to use. One of `"variant"`, `"family"`, or `"manufacturer"`. Default is `"variant"`.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform labels.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @return A fgcv_aircraft_dataset object representing the dataset.
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' fgcv <- fgcv_aircraft_dataset(root = root_dir, split = "train", annotation_level = "variant", download = TRUE)
#' first_item <- fgcv[1]
#' # image tensor of first item
#' first_item$x
#' # label of first item
#' first_item$y
#' }
#'
#' @name fgcv_aircraft_dataset
#' @aliases fgcv_aircraft_dataset
#' @title FGCV Aircraft dataset
#' @export
fgcv_aircraft_dataset <- dataset(
  name = "fgcv_aircraft_dataset",

  initialize = function(root = rappdirs::user_cache_dir("torch"),
                        split = "train",
                        annotation_level = "variant",
                        transform = NULL,
                        target_transform = NULL,
                        download = FALSE) {
    self$root <- root
    self$split <- split
    self$annotation_level <- annotation_level
    self$transform <- transform
    self$target_transform <- target_transform

    self$data_path <- fs::path(root, "fgvc-aircraft-2013b")
    if (download) self$download()

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use download = TRUE.")
    }

    ann_file <- fs::path(self$data_path, "data", 
      switch(annotation_level,
             "variant" = "variants.txt",
             "family" = "families.txt",
             "manufacturer" = "manufacturers.txt"))

    self$classes <- readr::read_lines(ann_file)
    self$class_to_idx <- rlang::set_names(seq_along(self$classes), self$classes)

    img_folder  <- fs::path(self$data_path, "data", "images")
    labels_file <- fs::path(self$data_path, "data", 
                            sprintf("images_%s_%s.txt", annotation_level, split))
    lines <- readr::read_lines(labels_file)

    self$image_paths <- character(length(lines))
    self$labels <- integer(length(lines))

    for (i in seq_along(lines)) {
      parts <- strsplit(lines[i], " ", fixed = TRUE)[[1]]
      if (length(parts) < 2) {
        warning("Line does not contain expected format: ", lines[i])
        next
      }
      img_id <- parts[1]
      class_name <- trimws(paste(parts[-1], collapse = " "))


      if (is.null(self$class_to_idx[[class_name]])) {
        warning("Unknown class label: ", class_name)
        next
      }

      self$image_paths[i] <- fs::path(img_folder, paste0(img_id, ".jpg"))
      self$labels[i] <- self$class_to_idx[[class_name]]
    }
  },

  .getitem = function(index) {
    img <- torchvision::transform_to_tensor(magick::image_read(self$image_paths[index]))

    if (!is.null(self$transform)) img <- self$transform(img)
    label <- self$labels[index]
    if (!is.null(self$target_transform)) label <- self$target_transform(label)
    list(x = img, y = label)
  },

  .length = function() {
    length(self$image_paths)
  },

  download = function() {
    if (self$check_exists()) return()

    fs::dir_create(self$root)
    url <- "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    md5_expected <- "d4acdd33327262359767eeaa97a4f732"

    rlang::inform("Downloading and processing FGCV-Aircraft dataset...")

    archive <- withr::with_options(
      list(timeout = 6000),
      download_and_cache(url)
    )

    md5_actual <- digest::digest(archive, algo = "md5", file = TRUE)
    if (md5_actual != md5_expected) {
      fs::file_delete(archive)
      stop("MD5 checksum does not match. The file may be corrupted.")
    }

    untar(archive, exdir = self$root)

    rlang::inform("FGCV-Aircraft dataset processed successfully !")

  },

  check_exists = function() {
    fs::dir_exists(self$data_path)
  }
)