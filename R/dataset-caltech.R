#' Caltech-101 Dataset Loader
#'
#' Downloads and loads the official Caltech-101 dataset with optional annotations.
#' The dataset consists of 101 object categories with about 40 to 800 images per category.
#' Each image is roughly 300 × 200 pixels with available object annotations.
#'
#' @param root Character. Root directory where dataset will be stored.
#' @param download Logical. If TRUE, downloads the dataset.
#' @param download_annotations Logical. If TRUE, downloads annotation files.
#' @param include_annotations Logical. If TRUE, includes annotations in returned samples.
#' @param transform Function. Optional transformation for images.
#' @param target_transform Function. Optional transformation for labels.
#'
#' @return An R6 dataset object inheriting from `torch::dataset`.
#'
#' @examples
#' \dontrun{
#' # Basic usage
#' ds <- caltech101_dataset(root = "./data", download = TRUE)
#'
#' # With annotations
#' ds <- caltech101_dataset(
#'   root = "./data",
#'   download = TRUE,
#'   download_annotations = TRUE,
#'   include_annotations = TRUE
#' )
#' }
#' @export
caltech101_dataset <- torch::dataset(
  name = "caltech101",

  initialize = function(root,
                        download = FALSE,
                        download_annotations = FALSE,
                        include_annotations = FALSE,
                        transform = NULL,
                        target_transform = NULL) {
    self$root <- normalizePath(root, mustWork = FALSE)
    self$transform <- transform
    self$target_transform <- target_transform
    self$include_annotations <- include_annotations

    # Updated paths accounting for nested structure
    self$zip_file <- file.path(self$root, "caltech-101.zip")
    self$annotations_file <- file.path(self$root, "Annotations.tar")
    self$extracted_dir <- file.path(self$root, "caltech-101", "101_ObjectCategories")
    self$annotations_dir <- file.path(self$root, "Annotations")

    if (download) {
      self$download()
      if (download_annotations) {
        self$download_annotations()
      }
    }

    self$load_data()
  },

  download = function() {
    if (!dir.exists(self$root)) {
      dir.create(self$root, recursive = TRUE, showWarnings = FALSE)
    }

    if (!file.exists(self$zip_file)) {
      file_url <- "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
      message("Downloading Caltech-101 dataset (131 MB)...")
      options(timeout = 600)
      utils::download.file(url = file_url, destfile = self$zip_file, mode = "wb")
      message("Verifying checksum...")

      # Verify MD5 checksum
      md5 <- tools::md5sum(self$zip_file)
      expected_md5 <- "3138e1922a9193bfa496528edbbc45d0"
      if (md5 != expected_md5) {
        file.remove(self$zip_file)
        stop("Downloaded file checksum does not match. The download may be corrupted.")
      }

      message("Extracting zip file...")
      utils::unzip(self$zip_file, exdir = self$root)

      # Extract the inner tar.gz file
      inner_tar <- file.path(self$root, "caltech-101", "101_ObjectCategories.tar.gz")
      if (file.exists(inner_tar)) {
        message("Extracting inner tar.gz file...")
        utils::untar(inner_tar, exdir = file.path(self$root, "caltech-101"))
        file.remove(inner_tar)
      } else {
        warning("Inner tar.gz file not found at ", inner_tar)
      }
    }
  },

  download_annotations = function() {
    if (!file.exists(self$annotations_file)) {
      annotations_url <- "https://data.caltech.edu/records/mzrjq-6wc02/files/Annotations.tar?download=1"
      message("Downloading Annotations (3.3 MB)...")
      utils::download.file(url = annotations_url, destfile = self$annotations_file, mode = "wb")

      if (!dir.exists(self$annotations_dir)) {
        message("Extracting annotations...")
        utils::untar(self$annotations_file, exdir = self$root)
        if (!dir.exists(self$annotations_dir)) {
          stop("Failed to extract annotations to ", self$annotations_dir)
        }
      }
    }
  },

  load_data = function() {
    if (!dir.exists(self$extracted_dir)) {
      stop(sprintf(
        "Dataset directory not found at %s. Set download=TRUE to download it.",
        self$extracted_dir
      ))
    }

    message("Loading dataset from ", self$extracted_dir)
    category_dirs <- list.dirs(self$extracted_dir, recursive = FALSE, full.names = TRUE)
    if (length(category_dirs) == 0) {
      stop("No category directories found in ", self$extracted_dir)
    }

    self$classes <- basename(category_dirs)
    self$class_to_idx <- setNames(seq_along(self$classes) - 1, self$classes)

    self$image_paths <- unlist(lapply(category_dirs, function(dir) {
      list.files(dir, pattern = "\\.jpg$", full.names = TRUE)
    }))

    message("Loaded ", length(self$image_paths), " images from ", length(self$classes), " categories")

    self$labels <- factor(basename(dirname(self$image_paths)), levels = self$classes)

    if (self$include_annotations) {
      if (!dir.exists(self$annotations_dir)) {
        warning("Annotations directory not found at ", self$annotations_dir)
        self$annotation_paths <- rep(NA_character_, length(self$image_paths))
      } else {
        self$annotation_paths <- vapply(self$image_paths, function(img_path) {
          img_name <- tools::file_path_sans_ext(basename(img_path))
          category <- basename(dirname(img_path))
          annotation_file <- file.path(self$annotations_dir, category, paste0(img_name, ".mat"))
          ifelse(file.exists(annotation_file), annotation_file, NA_character_)
        }, character(1))
        message("Found ", sum(!is.na(self$annotation_paths)), " annotations")
      }
    }
  },

  .getitem = function(index) {
    img_path <- self$image_paths[index]
    if (!file.exists(img_path)) {
      stop(sprintf("Image file not found: %s", img_path))
    }

    label <- self$labels[index]
    img_array <- jpeg::readJPEG(img_path)

    if (!is.null(self$transform)) {
      img_array <- self$transform(img_array)
    }

    label_idx <- torch::torch_tensor(
      as.integer(self$class_to_idx[[as.character(label)]]) - 1L,
      dtype = torch::torch_long()
    )$squeeze()

    if (!is.null(self$target_transform)) {
      label_idx <- self$target_transform(label_idx)
    }

    result <- list(x = img_array, y = label_idx)

    if (self$include_annotations && !is.na(self$annotation_paths[index])) {
      annotation <- R.matlab::readMat(self$annotation_paths[index])
      result$annotation <- annotation
    }

    result
  },

  .length = function() {
    length(self$image_paths)
  }
)
