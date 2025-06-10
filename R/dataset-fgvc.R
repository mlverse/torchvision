#' FGVC Aircraft Dataset
#'
#' Loads the FGVC Aircraft Dataset, a dataset of aircraft images with multiple annotation levels:
#' - "variant": fine-grained aircraft variants (e.g., specific model variants)
#' - "family": aircraft families
#' - "manufacturer": aircraft manufacturers
#' 
#' The dataset supports the following splits:
#' - `"train"`: training subset (simulated from `"trainval"`)
#' - `"val"`: validation subset (simulated from `"trainval"`)
#' - `"trainval"`: full training + validation set (as provided by the official dataset)
#' - `"test"`: test set with labels
#'
#' Note: The original dataset provides `"trainval"` and `"test"` splits. The `"train"` and `"val"` subsets are simulated from `"trainval"` by an 80/20 split with a fixed random seed.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/fgvc-aircraft-2013b`.
#' @param split Character. One of `"train"`, `"val"`, `"trainval"`, or `"test"`. Default is `"train"`.
#' @param annotation_level Character. Level of annotation to use for classification.
#' One of:
#' - `"variant"`: the most fine-grained level, e.g., `"Boeing 737-700"`. There are 100 visually distinguishable aircraft variants.
#' - `"family"`: a mid-level grouping, e.g., `"Boeing 737"`, which includes multiple variants. There are 70 distinct families.
#' - `"manufacturer"`: the coarsest level, e.g., `"Boeing"`, grouping multiple families under a single aircraft manufacturer. There are 30 manufacturers.
#'
#' Note: These levels form a strict hierarchy: 
#' each `"manufacturer"` consists of multiple `"families"`, and each `"family"` contains several `"variants"`. 
#' Not all combinations of levels are valid — for example, a `"variant"` always belongs to exactly one `"family"`, 
#' and a `"family"` always belongs to exactly one `"manufacturer"`. You cannot mix or arbitrarily combine levels.
#'
#' Default is `"variant"`.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform labels.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @return A `fgvc_aircraft_dataset` object, which is a torch-style dataset.
#' Each element is a named list with:
#' - `x`: the image as a 3D torch tensor (C x H x W), already converted using `transform_to_tensor()`, unlike most torchvision-style datasets where images are returned as PIL objects.
#' - `y`: an integer class label corresponding to the selected `annotation_level`.
#'
#' The dataset supports standard dataset operations like indexing (`dataset[i]`) and length (`length(dataset)`).
#'
#' @examples
#' \dontrun{
#' fgvc <- fgvc_aircraft_dataset( split = "train", annotation_level = "variant", download = TRUE)
#' first_item <- fgvc[1]
#' # image tensor of first item
#' first_item$x
#' # label of first item
#' first_item$y
#' }
#' 
#' @name fgvc_aircraft_dataset
#' @aliases fgvc_aircraft_dataset
#' @title FGVC Aircraft dataset
#' @export
fgvc_aircraft_dataset <- dataset(
  name = "fgvc_aircraft_dataset",

  initialize = function(root = tempdir(),
                        split = "train",
                        annotation_level = "variant",
                        transform = NULL,
                        target_transform = NULL,
                        download = FALSE) {
    self$root <- root
    self$split <- split
    self$annotation_level <- annotation_level
    self$transform <- if (is.null(transform)) {
      function(img) {
        torchvision::transform_resize(img, c(224, 224))
      }
    } else {
      transform
    }
    self$target_transform <- target_transform

    self$data_path <- fs::path(root, "fgvc-aircraft-2013b")
    if (download) self$download()

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use download = TRUE.")
    }

    ann_file <- fs::path(
      self$data_path, "data",
      switch(annotation_level,
        "variant" = "variants.txt",
        "family" = "families.txt",
        "manufacturer" = "manufacturers.txt",
        runtime_error("Invalid annotation_level")
      )
    )

    cl <- readr::read_lines(ann_file)
    self$classes <- cl
    self$class_to_idx <- setNames(seq_along(cl), cl)

    img_folder <- fs::path(self$data_path, "data", "images")

    raw_split <- switch(split,
      "train" = "trainval",
      "val" = "trainval",
      "trainval" = "trainval",
      "test" = "test",
      runtime_error("Invalid split value")
    )

    labels_file <- fs::path(
      self$data_path, "data",
      sprintf("images_%s_%s.txt", annotation_level, raw_split)
    )

    lines <- readr::read_lines(labels_file)

    if (split %in% c("train", "val")) {
      set.seed(42)
      idx <- sample(length(lines))
      cutoff <- floor(0.8 * length(lines))
      if (split == "train") {
        lines <- lines[idx[1:cutoff]]
      } else {
        lines <- lines[idx[(cutoff + 1):length(lines)]]
      }
    }

    self$image_paths <- character()
    self$labels <- integer()

    for (line in lines) {
      parts <- strsplit(line, " ", fixed = TRUE)[[1]]
      if (length(parts) < 2) next

      img_id <- parts[1]
      class_name <- trimws(paste(parts[-1], collapse = " "))
      class_idx <- self$class_to_idx[[class_name]]

      if (is.null(class_idx)) next

      self$image_paths <- c(self$image_paths, fs::path(img_folder, paste0(img_id, ".jpg")))
      self$labels <- c(self$labels, class_idx)
    }
  },

  .getitem = function(index) {
    img <- magick::image_read(self$image_paths[index])

    if (!is.null(self$transform)) {
      img <- self$transform(img)
    }

    img <- torchvision::transform_to_tensor(img)

    label <- self$labels[index]
    if (!is.null(self$target_transform)) {
      label <- self$target_transform(label)
    }

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

    rlang::inform("Downloading FGVC-Aircraft dataset...")

    archive <- withr::with_options(
      list(timeout = 6000),
      download_and_cache(url)
    )

    md5_actual <- digest::digest(archive, algo = "md5", file = TRUE)
    if (!identical(md5_actual, md5_expected)) {
      fs::file_delete(archive)
      stop("MD5 checksum does not match. The file may be corrupted.")
    }

    untar(archive, exdir = self$root)
    rlang::inform("FGVC-Aircraft dataset successfully extracted.")
  },

  check_exists = function() {
    fs::dir_exists(self$data_path)
  }
)
