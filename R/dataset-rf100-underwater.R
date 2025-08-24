#' RF100 Underwater Dataset Collection
#'
#' Loads one of the RF100 underwater object detection datasets: "pipes",
#' "aquarium", "objects", or "coral". Images are provided with COCO-style
#' bounding box annotations for object detection tasks.
#'
#' @param dataset Character. One of "pipes", "aquarium", "objects", or "coral".
#' @param split Character. One of "train", "test", or "valid".
#' @param root Character. Root directory where the dataset will be stored.
#' @param download Logical. If TRUE, downloads the dataset if not present at `root`.
#' @param transform Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target.
#'
#' @return A torch dataset. Each element is a named list with:
#' - `x`: H x W x 3 array representing the image.
#' - `y`: a list containing:
#'     - `labels`: character vector with object class names.
#'     - `boxes`: a tensor of shape (N, 4) with bounding boxes in (xmin, ymin, xmax, ymax).
#'
#' The returned item inherits the class `image_with_bounding_box` so it can be
#' visualised with helper functions such as [draw_bounding_boxes()].
#'
#' @examples
#' \dontrun{
#' ds <- rf100_underwater_collection(
#'   dataset = "pipes",
#'   split = "train",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#'
#' # Retrieve a sample and inspect annotations
#' item <- ds[2]
#' item$y$labels
#' item$y$boxes
#'
#' # Draw bounding boxes and display the image
#' boxed_img <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed_img)
#' }
#'
#' @family detection_dataset
#' @export
rf100_underwater_collection <- torch::dataset(
  name = "rf100_underwater_collection",

  resources = data.frame(
    dataset = c("pipes", "aquarium", "objects", "coral"),
    url = c(
      "https://huggingface.co/datasets/Francesco/underwater-pipes-4ng4t/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/aquarium-qlnqy/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/underwater-objects-5v7p8/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/coral-lwptl/resolve/main/dataset.tar.gz?download=1"
    ),
    md5 = rep(NA_character_, 4),
    stringsAsFactors = FALSE
  ),

  initialize = function(
    dataset = c("pipes", "aquarium", "objects", "coral"),
    split = c("train", "test", "valid"),
    root = if (.Platform$OS.type == "windows") fs::path("C:/torchvision-datasets") else fs::path_temp("torchvision-datasets"),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- match.arg(dataset)
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    fs::dir_create(self$root, recurse = TRUE)
    self$dataset_dir <- fs::path(self$root, paste0("rf100_underwater_", self$dataset))

    resource <- self$resources[self$resources$dataset == self$dataset, , drop = FALSE]
    self$archive_url <- resource$url

    if (download) self$download()

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
  },

  download = function() {
    if (self$check_exists()) return(invisible(NULL))

    if (fs::dir_exists(self$dataset_dir)) fs::dir_delete(self$dataset_dir)
    fs::dir_create(self$dataset_dir, recurse = TRUE)

    # Download the archive
    dest <- fs::path(self$dataset_dir, "dataset.tar.gz")
    download.file(self$archive_url, dest, mode = "wb")

    if (!requireNamespace("archive", quietly = TRUE)) {
      runtime_error("Package 'archive' is required. Please install.packages('archive').")
    }

    # Inspect archive and filter files
    a <- archive::archive(dest)

    # Keep only regular files (skip hardlinks, dirs, symlinks)
    files <- a$path[a$type == "file"]

    # Strip the long 'home/zuppif/.../rf100' prefix (7 levels deep)
    strip_n <- 7L

    # Extract only the file entries, skipping hardlinks
    archive::archive_extract(
      dest,
      files = files,
      dir = self$dataset_dir,
      strip_components = strip_n
    )

    # Clean up the downloaded archive
    fs::file_delete(dest)

    invisible(NULL)
  },

  check_exists = function() {
    ann <- self$discover_annotation_file()
    if (is.na(ann) || !fs::file_exists(ann)) return(FALSE)

    self$annotation_file <- ann
    self$split_dir <- fs::path_dir(ann)
    candidate_img_dirs <- c(self$split_dir, fs::path(self$split_dir, "images"))
    self$image_dir <- candidate_img_dirs[fs::dir_exists(candidate_img_dirs)][1]

    fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)
  },

  discover_annotation_file = function() {
    if (!fs::dir_exists(self$dataset_dir)) return(NA_character_)

    jsons <- fs::dir_ls(self$dataset_dir, recurse = TRUE, type = "file", glob = "*_annotations.coco.json")

    if (!length(jsons)) {
      # Try alternative patterns
      jsons_alt <- fs::dir_ls(self$dataset_dir, recurse = TRUE, type = "file", glob = "*.json")
      return(NA_character_)
    }

    jsons_split <- jsons[grepl(paste0("[/\\\\]", self$split, "[/\\\\]_annotations\\.coco\\.json$"), jsons)]
    if (length(jsons_split)) return(jsons_split[[1]])
    jsons[[1]]
  },

  load_annotations = function() {
    ann <- jsonlite::fromJSON(self$annotation_file)
    self$categories  <- ann$categories
    self$images      <- ann$images
    self$annotations <- ann$annotations

    candidate_img_dirs <- c(self$split_dir, fs::path(self$split_dir, "images"))
    self$image_dir <- candidate_img_dirs[fs::dir_exists(candidate_img_dirs)][1]

    # Get all actual image files - search recursively in the entire dataset directory
    # since images might be scattered across different split directories
    all_img_files <- fs::dir_ls(self$dataset_dir, type = "file", recurse = TRUE,
                                regexp = "\\.(jpg|jpeg|png|bmp)$")

    # Also try the specific image directory if it exists
    if (!is.na(self$image_dir) && fs::dir_exists(self$image_dir)) {
      local_img_files <- fs::dir_ls(self$image_dir, type = "file",
                                    regexp = "\\.(jpg|jpeg|png|bmp)$")
      all_img_files <- unique(c(all_img_files, local_img_files))
    }

    actual_basenames <- fs::path_file(all_img_files)

    # Since the JSON filenames already include the RoboFlow format,
    # we can try direct matching first
    matched_paths <- character(nrow(self$images))
    exists <- logical(nrow(self$images))

    # Create a lookup table for faster matching
    filename_to_path <- setNames(all_img_files, actual_basenames)

    for (i in seq_len(nrow(self$images))) {
      json_filename <- self$images$file_name[i]

      # Direct filename match
      if (json_filename %in% names(filename_to_path)) {
        matched_paths[i] <- filename_to_path[[json_filename]]
        exists[i] <- TRUE
        next
      }

      # Try searching in specific image directory
      if (!is.na(self$image_dir)) {
        direct_path <- fs::path(self$image_dir, json_filename)
        if (fs::file_exists(direct_path)) {
          matched_paths[i] <- direct_path
          exists[i] <- TRUE
          next
        }
      }

      # Fuzzy matching - extract base name and look for partial matches
      json_base <- sub("_[lr]rgb_jpg\\.rf\\.[^.]+\\.jpg$", "", json_filename)
      json_base <- sub("^(empty_)?frame", "frame", json_base)

      potential_matches <- all_img_files[grepl(json_base, actual_basenames, fixed = TRUE)]

      if (length(potential_matches) > 0) {
        matched_paths[i] <- potential_matches[1]  # Take first match
        exists[i] <- TRUE
        next
      }

      exists[i] <- FALSE
    }

    # Filter to existing images
    self$images <- self$images[exists, , drop = FALSE]
    matched_paths <- matched_paths[exists]
    keep_ids <- self$images$id
    self$annotations <- self$annotations[self$annotations$image_id %in% keep_ids, , drop = FALSE]

    if (nrow(self$annotations) > 0) {
      self$annotations_by_image <- split(self$annotations, self$annotations$image_id)
    } else {
      self$annotations_by_image <- list()
    }

    self$image_paths <- matched_paths
  },

  .getitem = function(index) {
    img_path <- self$image_paths[index]
    img_info <- self$images[index, ]
    anns     <- self$annotations_by_image[[as.character(img_info$id)]]

    x <- tryCatch(jpeg::readJPEG(img_path), error = function(e) {
      tryCatch(png::readPNG(img_path), error = function(e) {
        if (requireNamespace("magick", quietly = TRUE)) {
          img <- magick::image_read(img_path)
          arr <- magick::image_data(img, channels = "rgb")
          aperm(as.integer(arr)/255, c(3,2,1))
        } else runtime_error("Failed to read image: ", img_path)
      })
    })

    if (length(dim(x)) == 2L) {
      x <- array(rep(x, each = 3), dim = c(dim(x), 3L))
    }

    if (is.null(anns) || nrow(anns) == 0) {
      boxes  <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
    } else {
      boxes_xywh <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float())
      boxes      <- box_xywh_to_xyxy(boxes_xywh)
      labels     <- as.character(self$categories$name[match(anns$category_id, self$categories$id)])
    }

    y <- list(labels = labels, boxes = boxes)

    if (!is.null(self$transform)) x <- self$transform(x)
    if (!is.null(self$target_transform)) y <- self$target_transform(y)

    structure(list(x = x, y = y), class = "image_with_bounding_box")
  },

  .length = function() {
    length(self$image_paths)
  }
)
