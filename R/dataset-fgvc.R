#' FGVC Aircraft Dataset
#'
#' Loads the FGVC Aircraft Dataset, a dataset of aircraft images with multiple annotation levels:
#' - "variant": fine-grained aircraft variants (e.g., specific model variants)
#' - "family": aircraft families
#' - "manufacturer": aircraft manufacturers
#'
#' The FGVC-Aircraft dataset supports the following official splits:
#' - `"train"`: training subset
#' - `"val"`: validation subset
#' - `"trainval"`: combined training and validation set
#' - `"test"`: test set with labels (used for evaluation)
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/fgvc-aircraft-2013b`.
#' @param split Character. One of `"train"`, `"val"`, `"trainval"`, or `"test"`. Default is `"train"`.
#' @param annotation_level Character. Level of annotation to use for classification. Default is `"variant"`.
#' One of `"variant"`, `"family"`, or `"manufacturer"`. See *Details*.
#'
#' @param transform Optional function to transform input images after loading. Default is [transform_to_tensor()].
#' @param target_transform Optional function to transform labels.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @details
#' The `annotation_level` determines the granularity of labels used for classification and supports three values:
#'
#' - `"variant"`: the most fine-grained level, e.g., `"Boeing 737-700"`. There are 100 visually distinguishable variants.
#' - `"family"`: a mid-level grouping, e.g., `"Boeing 737"`, which includes multiple variants. There are 70 distinct families.
#' - `"manufacturer"`: the coarsest level, e.g., `"Boeing"`, grouping multiple families under a single manufacturer. There are 30 manufacturers.
#'
#' These levels form a strict hierarchy: each `"manufacturer"` consists of multiple `"families"`, and each `"family"` contains several `"variants"`.
#' Not all combinations of levels are valid — for example, a `"variant"` always belongs to exactly one `"family"`, and a `"family"` to exactly one `"manufacturer"`.
#'
#' @return An object of class \code{fgvc_aircraft_dataset}, which behaves like a torch-style dataset.
#' Each element is a named list with:
#' - `x`: a torch tensor of the image with shape (C x H x W). Please note that images have varying sizes.
#' - `y`: an integer class label corresponding to the selected `annotation_level`.
#'
#' The dataset supports standard dataset operations like indexing (`dataset[i]`) and length (`length(dataset)`).
#'
#' @examples
#' \dontrun{
#' fgvc <- fgvc_aircraft_dataset( split = "train", annotation_level = "variant", download = TRUE )
#'
#' # Define a custom collate function to resize images in the batch
#' resize_collate_fn <- function(batch) {
#'   xs <- lapply(batch, function(sample) {
#'     torchvision::transform_resize(sample$x, c(768, 1024))
#'   })
#'   xs <- torch::torch_stack(xs)
#'   ys <- torch::torch_tensor(sapply(batch, function(sample) sample$y), dtype = torch::torch_long())
#'   list(x = xs, y = ys)
#' }
#'
#' dl <- torch::dataloader( dataset = fgvc, batch_size = 2, shuffle = TRUE, collate_fn = resize_collate_fn )
#' batch <- dataloader_next(dataloader_make_iter(dl))
#' batch$x  # batched image tensors resized to 768x1024
#' batch$y  # class labels
#' }
#'
#' @name fgvc_aircraft_dataset
#' @aliases fgvc_aircraft_dataset
#' @title FGVC Aircraft dataset
#' @export
fgvc_aircraft_dataset <- dataset(
  name = "fgvc_aircraft",

  initialize = function(
    root = tempdir(),
    split = "train",
    annotation_level = "variant",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    rlang::inform(glue::glue("Initializing FGVC-Aircraft dataset..."))

    self$root <- root
    self$split <- split
    self$annotation_level <- annotation_level
    self$transform <- transform
    self$target_transform <- target_transform

    self$base_dir <- file.path(root, "fgvc-aircraft-2013b")
    self$data_dir <- file.path(self$base_dir, "data")

    if (download) {
      rlang::inform(glue::glue(
        "Downloading and extracting FGVC-Aircraft dataset (Size: ~2.6 GB)..."
      ))
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use download = TRUE to fetch it.")
    }

    label_file <- file.path(
      self$data_dir,
      glue::glue("images_{annotation_level}_{split}.txt")
    )

    cls_file <- file.path(
      self$data_dir,
      switch(
        annotation_level,
        "variant" = "variants.txt",
        "family" = "families.txt",
        "manufacturer" = "manufacturers.txt",
        runtime_error("Invalid annotation_level")
      )
    )

    classes <- readLines(cls_file)
    self$classes <- classes
    self$class_to_idx <- setNames(seq_along(classes), classes)

    split_df <- read.fwf(
      label_file,
      widths = c(7, 100),
      colClasses = "character",
      stringsAsFactors = FALSE,
      col.names = c("img_id", "class_name"),
      strip.white = TRUE
    )
    class_idxs <- self$class_to_idx[split_df$class_name]
    known_mask <- !vapply(class_idxs, is.null, logical(1))
    self$image_paths <- file.path(
      self$data_dir,
      "images",
      glue::glue("{split_df$img_id[known_mask]}.jpg", .envir = environment())
    )
    self$labels <- class_idxs[known_mask]

    rlang::inform(glue::glue(
      "FGVC-Aircraft dataset loaded successfully with {length(self$image_paths)} samples ({split}, {annotation_level}-level)."
    ))
  },

  .getitem = function(index) {
    img <- magick::image_read(self$image_paths[index])
    if (!is.null(self$transform)) {
      img <- self$transform(img)
    } else {
      img <- torchvision::transform_to_tensor(img)
    }

    label <- self$labels[index]
    if (!is.null(self$target_transform)) {
      label <- self$target_transform(label)
    }

    list(x = img, y = label)
  },

  .length = function() {
    length(self$image_paths)
  },

  check_exists = function() {
    fs::dir_exists(self$base_dir)
  },

  download = function() {
    if (self$check_exists()) {
      return()
    }

    fs::dir_create(self$root)
    url <- "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    md5 <- "d4acdd33327262359767eeaa97a4f732"
    file <- download_and_cache(url)
    if (digest::digest(file, algo = "md5", file = TRUE) != md5) {
      fs::file_delete(file)
      stop("MD5 mismatch. Corrupted file.")
    }

    untar(file, exdir = self$root)
    rlang::inform(glue::glue(
      "FGVC-Aircraft dataset downloaded and extracted successfully."
    ))
  }
)
