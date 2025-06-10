#' MS COCO Captions Dataset
#'
#' Loads the MS COCO Captions dataset.
#'
#' @param root Path to the root directory for data storage.
#' @param split "train" or "val".
#' @param download Whether to download the dataset if not present.
#' @param ... Additional arguments (currently unused).
#'
#' @export
ms_coco_captions_dataset <- dataset(
  name = "ms_coco_captions_dataset",
  initialize = function(root, split = "val", download = FALSE, ...) {
    self$root <- root
    self$split <- split
    self$image_dir <- file.path(root, "coco", paste0(split, "2014"))
    self$annotation_file <- file.path(root, "coco", "annotations", paste0("captions_", split, "2014.json"))

    if (download) {
      self$download()
    }

    # Check if required files exist before proceeding
    if (!file.exists(self$annotation_file) || !dir.exists(self$image_dir)) {
      if (!download) {
        stop("Dataset files not found. Set download=TRUE to download the dataset.")
      }
    }

    self$loader <- function(path) {
      if (!file.exists(path)) {
        stop("Image file does not exist: ", path)
      }
      img <- magick::image_read(path)
      # Convert to array and ensure proper dimensions [height, width, channels]
      img_array <- as.array(magick::image_data(img, channels = "rgb"))
      # magick returns [channels, width, height], we need [height, width, channels]
      aperm(img_array, c(3, 2, 1))
    }

    self$load_annotations()
  },

  .getitem = function(index) {
    if (index < 1 || index > nrow(self$annotations)) {
      stop("Index out of bounds: ", index)
    }

    annotation <- self$annotations[index, ]
    image_id <- annotation$image_id
    caption <- annotation$caption
    image_file <- file.path(self$image_dir, sprintf("%012d.jpg", image_id))

    tryCatch({
      image <- self$loader(image_file)
      list(
        image = image,
        caption = caption,
        image_id = image_id
      )
    }, error = function(e) {
      stop("Failed to load item at index ", index, ": ", e$message)
    })
  },

  .length = function() {
    if (is.null(self$annotations) || nrow(self$annotations) == 0) {
      return(0)
    }
    nrow(self$annotations)
  },

  download = function() {
    # Create directory structure
    coco_dir <- file.path(self$root, "coco")
    ann_dir <- file.path(coco_dir, "annotations")
    dir.create(coco_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(ann_dir, recursive = TRUE, showWarnings = FALSE)

    ann_url <- "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    img_url <- sprintf("http://images.cocodataset.org/zips/%s2014.zip", self$split)

    ann_zip <- file.path(tempdir(), "annotations.zip")
    img_zip <- file.path(tempdir(), "images.zip")

    # Download annotations if not already present
    if (!file.exists(self$annotation_file)) {
      message("Downloading annotations...")
      tryCatch({
        withr::with_options(list(timeout = 6000), {
          utils::download.file(ann_url, ann_zip, mode = "wb")
          utils::unzip(ann_zip, exdir = coco_dir)
        })
      }, error = function(e) {
        stop("Failed to download annotations: ", e$message)
      })
    }

    # Download images if directory doesn't exist or is empty
    if (!dir.exists(self$image_dir) || length(list.files(self$image_dir, pattern = "\\.jpg$")) == 0) {
      message("Downloading images for split: ", self$split)
      tryCatch({
        withr::with_options(list(timeout = 6000), {
          utils::download.file(img_url, img_zip, mode = "wb")
          utils::unzip(img_zip, exdir = coco_dir)
        })
      }, error = function(e) {
        stop("Failed to download images: ", e$message)
      })
    }

    # Clean up temporary files
    if (file.exists(ann_zip)) file.remove(ann_zip)
    if (file.exists(img_zip)) file.remove(img_zip)
  },

  load_annotations = function() {
    if (!file.exists(self$annotation_file)) {
      stop("Annotation file not found: ", self$annotation_file)
    }

    tryCatch({
      message("Loading annotations from: ", self$annotation_file)
      json_data <- jsonlite::fromJSON(self$annotation_file, flatten = TRUE)
      annotations <- json_data$annotations

      # Data validation
      if (is.null(annotations) || nrow(annotations) == 0) {
        stop("No annotations found in the file")
      }

      # Remove annotations with NA image_id
      annotations <- annotations[!is.na(annotations$image_id), , drop = FALSE]

      if (nrow(annotations) == 0) {
        stop("No valid annotations found after removing NA image_ids")
      }

      # Filter only those whose corresponding image file exists
      image_paths <- file.path(self$image_dir, sprintf("%012d.jpg", annotations$image_id))
      exists_flag <- file.exists(image_paths)

      missing_count <- sum(!exists_flag)
      if (missing_count > 0) {
        warning("Missing ", missing_count, " out of ", length(exists_flag), " images. These annotations will be skipped.")
      }

      self$annotations <- annotations[exists_flag, , drop = FALSE]

      if (nrow(self$annotations) == 0) {
        stop("No annotations remain after filtering for existing images")
      }

      message("Loaded ", nrow(self$annotations), " valid annotations")

    }, error = function(e) {
      stop("Failed to load annotations: ", e$message)
    })
  }
)
