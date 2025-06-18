#' FGVC Aircraft Dataset
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
#' One of `"variant"`, `"family"`, `"manufacturer"`, or `"all"`. See *Details*.
#'
#' @param transform Optional function to transform input images after loading. Default is [transform_to_tensor()].
#' @param target_transform Optional function to transform labels.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @details
#' The `annotation_level` determines the granularity of labels used for classification and supports four values:
#'
#' - `"variant"`: the most fine-grained level, e.g., `"Boeing 737-700"`. There are 100 visually distinguishable variants.
#' - `"family"`: a mid-level grouping, e.g., `"Boeing 737"`, which includes multiple variants. There are 70 distinct families.
#' - `"manufacturer"`: the coarsest level, e.g., `"Boeing"`, grouping multiple families under a single manufacturer. There are 30 manufacturers.
#' - `"all"`: multi-label format that returns all three levels as a vector of class indices `c(manufacturer_idx, family_idx, variant_idx)`.
#'
#' These levels form a strict hierarchy: each `"manufacturer"` consists of multiple `"families"`, and each `"family"` contains several `"variants"`.
#' Not all combinations of levels are valid — for example, a `"variant"` always belongs to exactly one `"family"`, and a `"family"` to exactly one `"manufacturer"`.
#'
#' When `annotation_level = "all"` is used, the `$classes` field is a named list with three components:
#' - `classes$manufacturer`: a character vector of manufacturer names
#' - `classes$family`: a character vector of family names
#' - `classes$variant`: a character vector of variant names
#'
#' @return An object of class \code{fgvc_aircraft_dataset}, which behaves like a torch-style dataset.
#' Each element is a named list with:
#' - `x`: a torch tensor of the image with shape (C x H x W). Please note that images have varying sizes.
#' - `y`: for single-level annotation (`"variant"`, `"family"`, `"manufacturer"`): an integer class label.
#'        for multi-level annotation (`"all"`): a vector of three integers `c(manufacturer_idx, family_idx, variant_idx)`.
#'
#' The dataset supports standard dataset operations like indexing (`dataset[i]`) and length (`length(dataset)`).
#'
#' @examples
#' \dontrun{
#' # Single-label classification
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
#' dl <- torch::dataloader( dataset = fgvc, batch_size = 2, collate_fn = resize_collate_fn )
#' batch <- dataloader_next(dataloader_make_iter(dl))
#' batch$x  # batched image tensors resized to 768x1024
#' batch$y  # class labels
#'
#' # Multi-label classification
#' fgvc_multi <- fgvc_aircraft_dataset( split = "train", annotation_level = "all", download = TRUE )
#' item <- fgvc_multi[1]
#' item$y  # Returns a named list with class indices
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
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (annotation_level == "all") {
      self$setup_multilabel()
    } else {
      self$setup_singlelabel()
    }

    rlang::inform(glue::glue(
      "FGVC-Aircraft dataset loaded successfully with {length(self$image_paths)} samples ({split}, {annotation_level}-level)."
    ))
  },

  setup_singlelabel = function() {
    label_file <- file.path(
      self$data_dir,
      glue::glue("images_{self$annotation_level}_{self$split}.txt")
    )

    cls_file <- file.path(
      self$data_dir,
      switch(
        self$annotation_level,
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
  },

  setup_multilabel = function() {
    self$classes <- list(
      manufacturer = readLines(file.path(self$data_dir, "manufacturers.txt")),
      family = readLines(file.path(self$data_dir, "families.txt")),
      variant = readLines(file.path(self$data_dir, "variants.txt"))
    )

    self$class_to_idx <- lapply(self$classes, function(cls) {
      setNames(seq_along(cls), cls)
    })

    levels <- names(self$classes)
    label_data <- lapply(levels, function(level) {
      label_file <- file.path(
        self$data_dir,
        glue::glue("images_{level}_{self$split}.txt")
      )
      read.fwf(
        label_file,
        widths = c(7, 100),
        colClasses = "character",
        stringsAsFactors = FALSE,
        col.names = c("img_id", "class_name"),
        strip.white = TRUE
      )
    })
    names(label_data) <- levels

    common_img_ids <- Reduce(intersect, lapply(label_data, function(df) df$img_id))

    idx_mat <- lapply(levels, function(level) {
      df <- label_data[[level]]
      fct <- factor(df$class_name, levels = names(self$class_to_idx[[level]]))
      idx <- as.integer(fct)
      names(idx) <- df$img_id
      idx
    })

    for (img_id in common_img_ids) {
      entry <- lapply(seq_along(levels), function(i) {
        idx_mat[[i]][[img_id]]
      })

      self$labels[[length(self$labels) + 1]] <- entry
      self$image_paths <- c(self$image_paths, file.path(
        self$data_dir, "images", glue::glue("{img_id}.jpg")
      ))
    }
  },

  .getitem = function(index) {
    img <- jpeg::readJPEG(self$image_paths[index])
    img <- img * 255L
    img <- as.integer(img)
    label <- self$labels[[index]]

    if (!is.null(self$transform)) {
      img <- self$transform(img)
    }

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

    archive <- withr::with_options(
      list(timeout = 1200),
      download_and_cache(url)
    )
    if (!tools::md5sum(archive) == md5) {
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }

    untar(archive, exdir = self$root)

    rlang::inform(glue::glue(
      "FGVC-Aircraft dataset downloaded and extracted successfully."
    ))
  }
)
