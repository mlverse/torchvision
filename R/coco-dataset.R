#' MS COCO Detection Dataset
#'
#' Loads MS COCO Detection Dataset for object detection and segmentation.
#' This dataset contains images with bounding box annotations, segmentation masks,
#' and object class labels for 80 different object categories.
#'
#' @param root Root directory where data is downloaded or stored.
#' @param train If TRUE, loads training set, otherwise validation set.
#' @param year Dataset year. One of c("2017", "2016", "2014"). Default is "2017".
#' @param download Whether to download the dataset.
#' @param transforms Optional transform to be applied on images.
#' @param target_transform Optional transform to be applied on targets.
#' @param ... Additional arguments passed to the dataset.
#'
#' @return A dataset object that returns list with 'image' and 'target' elements.
#'   The target contains bounding boxes, labels, segmentation masks, and metadata.
#'
#' @export
coco_detection_dataset <- dataset(
  "coco_detection",
  initialize = function(root, train = TRUE, year = c("2017", "2016", "2014"), download = FALSE,
                        transforms = NULL, target_transform = NULL, ...) {
    # Validate year parameter
    year <- match.arg(year)

    # Convert train boolean to split string for internal use
    split <- if (train) "train" else "val"

    self$root <- root
    self$train <- train
    self$year <- year
    self$split <- split
    self$transforms <- transforms
    self$target_transform <- target_transform

    # Set up paths based on year
    if (year == "2016") {
      # COCO 2016 uses 2014 images but different annotation structure
      self$data_dir <- fs::path(root, "coco2016")
      self$image_dir <- fs::path(self$data_dir, glue::glue("{split}2014"))
      self$annotation_file <- fs::path(self$data_dir, "annotations",
                                       glue::glue("instances_{split}2016.json"))
    } else {
      self$data_dir <- fs::path(root, glue::glue("coco{year}"))
      self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{year}"))
      self$annotation_file <- fs::path(self$data_dir, "annotations",
                                       glue::glue("instances_{split}{year}.json"))
    }

    if (download) {
      self$download()
    }

    # Check if required files exist
    if (!fs::file_exists(self$annotation_file) || !fs::dir_exists(self$image_dir)) {
      if (!download) {
        stop("Dataset files not found. Set download=TRUE to download the dataset.")
      }
    }

    # Initialize image loader
    self$loader <- function(path) {
      if (!fs::file_exists(path)) {
        stop("Image file does not exist: ", path)
      }
      img <- magick::image_read(path)
      # Convert to array and ensure proper dimensions [height, width, channels]
      img_array <- as.array(magick::image_data(img, channels = "rgb"))
      # magick returns [channels, width, height], we need [height, width, channels]
      aperm(img_array, c(3, 2, 1))
    }

    # Load annotations and build index
    self$load_annotations()
  },

  .getitem = function(index) {
    if (index < 1 || index > length(self$image_ids)) {
      stop("Index out of bounds: ", index)
    }

    image_id <- self$image_ids[index]
    image_info <- self$images[[as.character(image_id)]]

    # Load image
    image_path <- fs::path(self$image_dir, image_info$file_name)
    image <- self$loader(image_path)

    # Get annotations for this image
    image_annotations <- self$annotations[self$annotations$image_id == image_id, ]

    # Prepare bounding boxes (convert from COCO format [x, y, width, height] to [x1, y1, x2, y2])
    if (nrow(image_annotations) > 0) {
      bboxes <- do.call(rbind, lapply(1:nrow(image_annotations), function(i) {
        bbox <- image_annotations$bbox[[i]]  # COCO format: [x, y, width, height]
        c(bbox[1], bbox[2], bbox[1] + bbox[3], bbox[2] + bbox[4])  # Convert to [x1, y1, x2, y2]
      }))
      colnames(bboxes) <- c("x1", "y1", "x2", "y2")

      # Extract other annotation data
      labels <- image_annotations$category_id
      areas <- sapply(image_annotations$area, function(x) if(is.null(x)) 0 else x)
      iscrowd <- sapply(image_annotations$iscrowd, function(x) if(is.null(x)) 0 else x)

      # Handle segmentation masks (if present)
      segmentation <- image_annotations$segmentation
    } else {
      # Empty annotations
      bboxes <- matrix(nrow = 0, ncol = 4)
      colnames(bboxes) <- c("x1", "y1", "x2", "y2")
      labels <- integer(0)
      areas <- numeric(0)
      iscrowd <- integer(0)
      segmentation <- list()
    }

    # Prepare target
    target <- list(
      image_id = image_id,
      boxes = bboxes,
      labels = labels,
      area = areas,
      iscrowd = iscrowd,
      segmentation = segmentation,
      height = image_info$height,
      width = image_info$width
    )

    # Apply transforms if provided
    if (!is.null(self$transforms)) {
      image <- self$transforms(image)
    }

    if (!is.null(self$target_transform)) {
      target <- self$target_transform(target)
    }

    list(
      image = image,
      target = target
    )
  },

  .length = function() {
    length(self$image_ids)
  },

  download = function() {
    # Create directory structure
    fs::dir_create(self$data_dir, recurse = TRUE)
    fs::dir_create(fs::path(self$data_dir, "annotations"), recurse = TRUE)

    # Define URLs based on year
    if (self$year == "2017") {
      ann_url <- "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
      img_url <- glue::glue("http://images.cocodataset.org/zips/{self$split}2017.zip")
    } else if (self$year == "2016") {
      # 2016 uses 2014 images but 2016 annotations
      ann_url <- "http://images.cocodataset.org/annotations/annotations_trainval2016.zip"
      img_url <- glue::glue("http://images.cocodataset.org/zips/{self$split}2014.zip")
    } else if (self$year == "2014") {
      ann_url <- "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
      img_url <- glue::glue("http://images.cocodataset.org/zips/{self$split}2014.zip")
    }

    # Download annotations if not present
    if (!fs::file_exists(self$annotation_file)) {
      rlang::inform(glue::glue("Downloading COCO {self$year} annotations..."))
      ann_zip <- fs::path(tempdir(), "annotations.zip")

      tryCatch({
        withr::with_options(list(timeout = 6000), {
          utils::download.file(ann_url, ann_zip, mode = "wb")
          utils::unzip(ann_zip, exdir = self$data_dir)
        })
      }, error = function(e) {
        stop("Failed to download annotations: ", e$message)
      })

      # Clean up
      if (fs::file_exists(ann_zip)) fs::file_delete(ann_zip)
    }

    # Download images if not present
    if (!fs::dir_exists(self$image_dir) ||
        length(fs::dir_ls(self$image_dir, glob = "*.jpg")) == 0) {
      rlang::inform(glue::glue("Downloading COCO {self$year} {self$split} images..."))
      img_zip <- fs::path(tempdir(), "images.zip")

      tryCatch({
        withr::with_options(list(timeout = 6000), {
          utils::download.file(img_url, img_zip, mode = "wb")
          utils::unzip(img_zip, exdir = self$data_dir)
        })
      }, error = function(e) {
        stop("Failed to download images: ", e$message)
      })

      # Clean up
      if (fs::file_exists(img_zip)) fs::file_delete(img_zip)
    }
  },

  load_annotations = function() {
    if (!fs::file_exists(self$annotation_file)) {
      stop("Annotation file not found: ", self$annotation_file)
    }

    tryCatch({
      rlang::inform(glue::glue("Loading annotations from: {self$annotation_file}"))
      json_data <- jsonlite::fromJSON(self$annotation_file, flatten = FALSE)

      # Store images info as named list for quick lookup
      image_list <- split(json_data$images, seq_len(nrow(json_data$images)))
      self$images <- stats::setNames(image_list, vapply(image_list, function(x) as.character(x$id), character(1)))

      # Store annotations as data frame
      self$annotations <- json_data$annotations

      # Store categories with mapping
      self$categories <- json_data$categories
      self$category_names <- stats::setNames(self$categories$name, self$categories$id)

      # Get list of image IDs that have corresponding image files
      all_image_ids <- as.numeric(names(self$images))
      image_files <- fs::path(self$image_dir,
                              sapply(all_image_ids, function(id) self$images[[as.character(id)]]$file_name))
      existing_files <- fs::file_exists(image_files)

      self$image_ids <- all_image_ids[existing_files]

      if (length(self$image_ids) == 0) {
        stop("No valid images found")
      }

      missing_count <- sum(!existing_files)
      if (missing_count > 0) {
        rlang::warn(glue::glue("Missing {missing_count} out of {length(all_image_ids)} images"))
      }

      rlang::inform(glue::glue("Loaded {length(self$image_ids)} images with {nrow(self$annotations)} annotations"))
      rlang::inform(glue::glue("Dataset contains {length(self$categories)} object categories"))

    }, error = function(e) {
      stop("Failed to load annotations: ", e$message)
    })
  },

  # Helper method to get category name from ID
  get_category_name = function(category_id) {
    self$category_names[as.character(category_id)]
  },

  # Helper method to get all category names
  get_categories = function() {
    self$categories
  }
)
