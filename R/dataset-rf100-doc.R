#' @include folder-dataset.R
NULL

#' RF100 Document Collection Dataset
#'
#' Loads one of the RF100 document object detection datasets with COCO-style
#' bounding box annotations for object detection tasks.
#'
#' @param dataset Character. One of "tweeter_post", "tweeter_profile", "document_part",
#'   "activity_diagram", "signature", "paper_part", "tabular_data", or "paragraph".
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
#' ds <- rf100_document_collection(
#'   dataset = "tweeter_post",
#'   split = "train",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#'
#' # Retrieve a sample and inspect annotations
#' item <- ds[1]
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
rf100_document_collection <- torch::dataset(
  name = "rf100_document_collection",

  resources = data.frame(
    dataset = c(
      "tweeter_post", "tweeter_profile", "document_part",
      "activity_diagram", "signature", "paper_part",
      "tabular_data", "paragraph"
    ),
    url = c(
      "https://huggingface.co/datasets/Francesco/tweeter-posts/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/tweeter-profile/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/document-parts/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/activity-diagrams-qdobr/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/signatures-xc8up/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/paper-parts/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/tabular-data-wf9uh/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/paragraphs-co84b/resolve/main/dataset.tar.gz?download=1"
    ),
    md5 = c(
      "0d2ce84f061dc186f4fb2ac44c3c0e5d",
      "60b449b28202bcbc3bdad9c833ff4a5e",
      "c8b34fa5a31be5557fe847e2ccf07eec",
      "04836401021613542718ab5d44880fd3",
      "96c49f5e2432abee7a9da602747e54f3",
      "7a933cf055ccb13c3309c748a38cd760",
      "0bfdc35e2eeb9c2d07d7b42bfe33e7ff",
      ""
    ),
    stringsAsFactors = FALSE
  ),

  initialize = function(
    dataset = c("tweeter_post", "tweeter_profile", "document_part",
                "activity_diagram", "signature", "paper_part",
                "tabular_data", "paragraph"),
    split = c("train", "test", "valid"),
    root = if (.Platform$OS.type == "windows") fs::path("C:/tv") else fs::path_temp("tv"),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- match.arg(dataset, self$resources$dataset)
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    fs::dir_create(self$root, recurse = TRUE)
    self$dataset_dir <- fs::path(self$root, self$dataset)

    resource <- self$resources[self$resources$dataset == self$dataset, , drop = FALSE]
    self$archive_url <- resource$url

    if (download) self$download()

    if (!self$check_exists()) {
      runtime_error(paste("Dataset not found. You can use `download = TRUE` to download it."))
    }

    self$load_annotations()
  },

  download = function() {
    if (self$check_exists()) return(invisible(NULL))

    if (fs::dir_exists(self$dataset_dir)) fs::dir_delete(self$dataset_dir)
    fs::dir_create(self$dataset_dir, recurse = TRUE)

    # Download the archive
    dest <- fs::path(self$dataset_dir, "dataset.tar.gz")

    if (requireNamespace("curl", quietly = TRUE)) {
      curl::curl_download(self$archive_url, dest, quiet = TRUE)
    } else {
      download.file(self$archive_url, dest, mode = "wb")
    }

    if (!requireNamespace("archive", quietly = TRUE)) {
      runtime_error("Archive package required. Install with: install.packages('archive')")
    }

    # Use short temp directory to avoid Windows path limits
    short_temp <- "C:/tmp"
    if (fs::dir_exists(short_temp)) fs::dir_delete(short_temp)
    fs::dir_create(short_temp, recurse = TRUE)

    tryCatch({
      # Get all archive contents at once
      contents <- archive::archive(dest)

      # Filter for needed files upfront
      ann_files <- contents[grepl("_annotations\\.coco\\.json", contents$path), ]
      img_files <- contents[grepl("\\.(jpg|jpeg|png|bmp)$", contents$path, ignore.case = TRUE), ]

      if (nrow(ann_files) == 0) {
        runtime_error("No annotation files found in archive.")
      }

      # Extract directly to final structure
      extract_dir <- self$dataset_dir
      fs::dir_create(extract_dir, recurse = TRUE)

      # Create split directories
      splits <- c("train", "test", "valid")
      for (split in splits) {
        fs::dir_create(fs::path(extract_dir, split), recurse = TRUE)
      }

      # Extract annotation files
      for (i in seq_len(nrow(ann_files))) {
        path <- ann_files$path[i]
        split <- if (grepl("/train/", path)) "train" else
          if (grepl("/test/", path)) "test" else
            if (grepl("/valid/", path)) "valid" else next

        target <- fs::path(extract_dir, split, "_annotations.coco.json")

        # Extract to short temp directory
        archive::archive_extract(dest, files = path, dir = short_temp)
        file.copy(fs::path(short_temp, path), target, overwrite = TRUE)
      }

      # Bulk extract images by split
      for (split in splits) {
        split_imgs <- img_files[grepl(paste0("/", split, "/"), img_files$path), ]
        if (nrow(split_imgs) == 0) next

        # Extract all images for this split at once
        archive::archive_extract(dest, files = split_imgs$path, dir = short_temp)

        # Copy images to final location
        for (j in seq_len(nrow(split_imgs))) {
          img_path <- split_imgs$path[j]
          img_name <- fs::path_file(img_path)
          src <- fs::path(short_temp, img_path)
          dst <- fs::path(extract_dir, split, img_name)

          if (fs::file_exists(src)) {
            file.copy(src, dst, overwrite = TRUE)
          }
        }
      }

      # Clean up temp directory
      if (fs::dir_exists(short_temp)) fs::dir_delete(short_temp)

    }, error = function(e) {
      # Clean up temp directory on error
      if (exists("short_temp") && fs::dir_exists(short_temp)) fs::dir_delete(short_temp)
      runtime_error("Failed to extract dataset: ", e$message)
    })

    # Cleanup
    fs::file_delete(dest)
    invisible(NULL)
  },

  check_exists = function() {
    ann <- self$discover_annotation_file()
    if (is.na(ann) || !fs::file_exists(ann)) return(FALSE)

    self$annotation_file <- ann
    self$split_dir <- fs::path_dir(ann)
    self$image_dir <- self$split_dir

    TRUE
  },

  discover_annotation_file = function() {
    data_dir <- fs::path(self$dataset_dir, self$split)
    ann_file <- fs::path(data_dir, "_annotations.coco.json")

    if (fs::file_exists(ann_file)) {
      return(ann_file)
    }

    # Fallback search
    if (!fs::dir_exists(self$dataset_dir)) return(NA_character_)

    jsons <- fs::dir_ls(self$dataset_dir, recurse = TRUE, type = "file",
                        regexp = "_annotations\\.coco\\.json$")

    if (!length(jsons)) return(NA_character_)

    # Find split-specific file
    jsons_split <- jsons[grepl(paste0("[/\\\\]", self$split, "[/\\\\]"), jsons)]
    if (length(jsons_split)) return(jsons_split[[1]])

    jsons[[1]]
  },

  load_annotations = function() {
    # Parse COCO annotations
    ann <- jsonlite::fromJSON(self$annotation_file)
    self$categories  <- ann$categories
    self$images      <- ann$images
    self$annotations <- ann$annotations

    # Build image paths efficiently
    self$image_paths <- fs::path(self$image_dir, self$images$file_name)

    # Filter to existing images
    exists <- fs::file_exists(self$image_paths)
    self$images <- self$images[exists, , drop = FALSE]
    self$image_paths <- self$image_paths[exists]

    # Filter annotations
    keep_ids <- self$images$id
    self$annotations <- self$annotations[self$annotations$image_id %in% keep_ids, , drop = FALSE]

    # Group annotations by image ID
    if (nrow(self$annotations) > 0) {
      self$annotations_by_image <- split(self$annotations, self$annotations$image_id)
    } else {
      self$annotations_by_image <- list()
    }
  },

  .getitem = function(index) {
    img_path <- self$image_paths[index]
    img_info <- self$images[index, ]
    anns     <- self$annotations_by_image[[as.character(img_info$id)]]

    # Load image using shared loader
    x <- tryCatch(
      base_loader(img_path),
      error = function(e) {
        runtime_error(paste("Failed to read image: ", img_path, " - ", e$message))
      }
    )

    if (length(dim(x)) == 3 && dim(x)[3] == 4) {
      x <- x[, , 1:3, drop = FALSE]
    }

    # Process annotations
    if (is.null(anns) || nrow(anns) == 0) {
      boxes  <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
    } else {
      # Convert COCO format [x, y, width, height] to [xmin, ymin, xmax, ymax]
      boxes_xywh <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float())
      boxes      <- box_xywh_to_xyxy(boxes_xywh)
      labels     <- as.character(self$categories$name[match(anns$category_id, self$categories$id)])
    }

    y <- list(labels = labels, boxes = boxes)

    # Apply transforms
    if (!is.null(self$transform)) x <- self$transform(x)
    if (!is.null(self$target_transform)) y <- self$target_transform(y)

    structure(list(x = x, y = y), class = "image_with_bounding_box")
  },

  .length = function() {
    length(self$image_paths)
  }
)
