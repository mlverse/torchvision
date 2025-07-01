#' FGVC Aircraft Dataset
#'
#' The FGVC-Aircraft dataset supports the following official splits:
#' - `"train"`: training subset with labels.
#' - `"val"`: validation subset with labels.
#' - `"trainval"`: combined training and validation set with labels.
#' - `"test"`: test set with labels (used for evaluation).
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/fgvc-aircraft-2013b`.
#' @param split Character. One of `"train"`, `"val"`, `"trainval"`, or `"test"`. Default is `"train"`.
#' @param annotation_level Character. Level of annotation to use for classification. Default is `"variant"`.
#' One of `"variant"`, `"family"`, `"manufacturer"`, or `"all"`. See *Details*.
#' @param transform Optional function to transform input images after loading. Default is `NULL`.
#' @param target_transform Optional function to transform labels. Default is `NULL`.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @details
#' The `annotation_level` determines the granularity of labels used for classification and supports four values:
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
#' - `x`: an array of shape (H, W, C) with pixel values in the range (0, 255). Please note that images have varying sizes.
#' - `y`: for single-level annotation (`"variant"`, `"family"`, `"manufacturer"`): an integer class label.
#'        for multi-level annotation (`"all"`): a vector of three integers `c(manufacturer_idx, family_idx, variant_idx)`.
#'
#' @examples
#' \dontrun{
#' # Single-label classification
#' fgvc <- fgvc_aircraft_dataset(transform = transform_to_tensor, download = TRUE)
#'
#' # Create a custom collate function to resize images and prepare batches
#' resize_collate_fn <- function(batch) {
#'   xs <- lapply(batch, function(item) {
#'     torchvision::transform_resize(item$x, c(768, 1024))
#'   })
#'   xs <- torch::torch_stack(xs)
#'   ys <- torch::torch_tensor(sapply(batch, function(item) item$y), dtype = torch::torch_long())
#'   list(x = xs, y = ys)
#' }
#' dl <- torch::dataloader(dataset = fgvc, batch_size = 2, collate_fn = resize_collate_fn)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#' batch$x  # batched image tensors with shape (2, 3, 768, 1024)
#' batch$y  # class labels as integer tensor of shape 2
#'
#' # Multi-label classification
#' fgvc <- fgvc_aircraft_dataset(split = "test", annotation_level = "all")
#' item <- fgvc[1]
#' item$x  # a double vector representing the image
#' item$y  # an integer vector of length 3: manufacturer, family, and variant indices
#' fgvc$classes$manufacturer[item$y[1]]  # e.g., "Boeing"
#' fgvc$classes$family[item$y[2]]        # e.g., "Boeing 707"
#' fgvc$classes$variant[item$y[3]]       # e.g., "707-320"
#' }
#'
#' @family classification_dataset
#' @export
fgvc_aircraft_dataset <- dataset(
  name = "fgvc_aircraft",
  archive_size = "2.8 GB",

  initialize = function(
    root = tempdir(),
    split = "train",
    annotation_level = "variant",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    
    self$root <- root
    self$split <- split
    self$annotation_level <- annotation_level
    self$transform <- transform
    self$target_transform <- target_transform
    self$base_dir <- file.path(root, "fgvc-aircraft-2013b")
    self$data_dir <- file.path(self$base_dir, "data")

    cli_inform("{.cls {class(self)[[1]]}} Dataset (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")

    if (download){
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use `download = TRUE` to fetch it.")
    }

    self$classes <- list(
      manufacturer = readLines(file.path(self$data_dir, "manufacturers.txt")),
      family = readLines(file.path(self$data_dir, "families.txt")),
      variant = readLines(file.path(self$data_dir, "variants.txt"))
    )

    levels <- names(self$classes)
    label_data <- lapply(levels, function(level) {
      read.fwf(
        file.path(self$data_dir, glue::glue("images_{level}_{self$split}.txt")),
        widths = c(7, 100),
        colClasses = "character",
        stringsAsFactors = FALSE,
        col.names = c("img_id", level),
        strip.white = TRUE
      )
    })
    names(label_data) <- levels

    merged_df <- Reduce(function(x, y) merge(x, y, by = "img_id"), label_data)
    merged_df[levels] <- lapply(levels, function(level) {
      as.integer(factor(merged_df[[level]], levels = self$classes[[level]]))
    })

    merged_df <- merged_df[complete.cases(merged_df), ]
    self$image_paths <- file.path(self$data_dir, "images", glue::glue("{merged_df$img_id}.jpg"))
    self$labels_df <- merged_df[, levels]

    cli_inform(
      "{.cls {class(self)[[1]]}} dataset loaded with {length(self$image_paths)} images across {length(self$classes[[annotation_level]])} classes."
    )
  },

  .getitem = function(index) {
    img <- jpeg::readJPEG(self$image_paths[index]) * 255

    label <- if (self$annotation_level == "all") {
      as.integer(self$labels_df[index, ])
    } else {
      self$labels_df[[self$annotation_level]][index]
    }

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
    if (self$check_exists()){
      return()
    }

    fs::dir_create(self$root)
    url <- "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    md5 <- "d4acdd33327262359767eeaa97a4f732"

    cli_inform("{.cls {class(self)[[1]]}} Downloading...")

    archive <- withr::with_options(list(timeout = 1200), download_and_cache(url))
    if (!tools::md5sum(archive) == md5) {
      runtime_error("Corrupt file! Delete the file at {archive} and try again.")
    }

    untar(archive, exdir = self$root)

    cli_inform("{.cls {class(self)[[1]]}} dataset downloaded and extracted successfully.")
  }
)
