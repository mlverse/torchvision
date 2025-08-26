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
    md5 = rep(NA_character_, 8),
    stringsAsFactors = FALSE
  ),

  initialize = function(
    dataset = c("tweeter_post", "tweeter_profile", "document_part",
                "activity_diagram", "signature", "paper_part",
                "tabular_data", "paragraph"),
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
    self$dataset_dir <- fs::path(self$root, paste0("rf100_document_", self$dataset))

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

    if (requireNamespace("curl", quietly = TRUE)) {
      curl::curl_download(self$archive_url, dest, quiet = FALSE)
    } else {
      download.file(self$archive_url, dest, mode = "wb")
    }

    # Create temporary simple extraction directory
    simple_extract <- fs::path(self$dataset_dir, "simple_extracted")
    if (fs::dir_exists(simple_extract)) fs::dir_delete(simple_extract)
    fs::dir_create(simple_extract, recurse = TRUE)

    if (requireNamespace("archive", quietly = TRUE)) {
      tryCatch({
        # Get archive contents
        contents <- archive::archive(dest)
        annotation_entries <- contents[grepl("_annotations\\.coco\\.json", contents$path), ]

        if (nrow(annotation_entries) > 0) {
          # Create split directories
          for (split_name in c("train", "test", "valid")) {
            fs::dir_create(fs::path(simple_extract, split_name), recurse = TRUE)
          }

          # Extract annotation files to simplified structure
          for (i in 1:nrow(annotation_entries)) {
            original_path <- annotation_entries$path[i]

            # Determine which split this belongs to
            split_name <- if (grepl("/train/", original_path)) "train" else
              if (grepl("/test/", original_path)) "test" else
                if (grepl("/valid/", original_path)) "valid" else "unknown"

            if (split_name != "unknown") {
              target_file <- fs::path(simple_extract, split_name, "_annotations.coco.json")

              # Extract specific file to temp directory
              temp_dir <- tempdir()
              archive::archive_extract(dest, files = original_path, dir = temp_dir)
              temp_extracted <- fs::path(temp_dir, original_path)

              if (fs::file_exists(temp_extracted)) {
                file.copy(temp_extracted, target_file, overwrite = TRUE)
              }
            }
          }

          for (split_name in c("train", "test", "valid")) {
            # Find image entries for this split
            image_pattern <- paste0("/", split_name, "/.*\\.(jpg|jpeg|png|bmp)$")
            image_entries <- contents[grepl(image_pattern, contents$path, ignore.case = TRUE), ]

            if (nrow(image_entries) > 0) {
              for (j in 1:nrow(image_entries)) {
                img_path <- image_entries$path[j]
                img_filename <- fs::path_file(img_path)
                target_img <- fs::path(simple_extract, split_name, img_filename)

                # Extract image to temp and copy
                tryCatch({
                  temp_dir <- tempdir()
                  archive::archive_extract(dest, files = img_path, dir = temp_dir)
                  temp_img <- fs::path(temp_dir, img_path)

                  if (fs::file_exists(temp_img)) {
                    file.copy(temp_img, target_img, overwrite = TRUE)
                  }
                }, error = function(e) {
                })
              }
            }
          }

          # Move simple extracted structure to be the main dataset directory
          final_dir <- fs::path(self$dataset_dir, "data")
          if (fs::dir_exists(final_dir)) fs::dir_delete(final_dir)
          fs::dir_copy(simple_extract, final_dir)
          fs::dir_delete(simple_extract)

        } else {
          runtime_error("No annotation files found in archive.")
        }

      }, error = function(e) {
        runtime_error("Failed to extract dataset: ", e$message)
      })
    } else {
      runtime_error("Archive package required for extraction. Install with: install.packages('archive')")
    }

    # Clean up
    gc()
    tryCatch(fs::file_delete(dest), error = function(e) invisible(NULL))

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

    # Look for annotation files in the dataset directory
    jsons <- fs::dir_ls(self$dataset_dir, recurse = TRUE, type = "file", regexp = "_annotations\\.coco\\.json$")

    if (!length(jsons)) {
      return(NA_character_)
    }

    # Try to find annotation file for the specific split
    jsons_split <- jsons[grepl(paste0("[/\\\\]", self$split, "[/\\\\]"), jsons)]
    if (length(jsons_split)) return(jsons_split[[1]])

    # Fallback to any annotation file
    jsons[[1]]
  },

  load_annotations = function() {
    # Parse COCO annotations
    ann <- jsonlite::fromJSON(self$annotation_file)
    self$categories  <- ann$categories
    self$images      <- ann$images
    self$annotations <- ann$annotations

    # Find all image files in the dataset directory
    all_img_files <- fs::dir_ls(
      self$dataset_dir,
      type = "file",
      recurse = TRUE,
      regexp = "(?i)\\.(jpg|jpeg|png|bmp)$"
    )

    # Match images from annotations to actual files
    actual_basenames <- tolower(fs::path_file(all_img_files))
    matched_paths <- character(nrow(self$images))
    exists <- logical(nrow(self$images))

    filename_to_path <- setNames(all_img_files, actual_basenames)

    for (i in seq_len(nrow(self$images))) {
      json_filename <- self$images$file_name[i]
      json_basename <- tolower(fs::path_file(json_filename))

      # Direct filename match
      if (json_basename %in% names(filename_to_path)) {
        matched_paths[i] <- filename_to_path[[json_basename]]
        exists[i] <- TRUE
        next
      }

      # Try full path from dataset directory
      full_path <- fs::path(self$dataset_dir, json_filename)
      if (fs::file_exists(full_path)) {
        matched_paths[i] <- full_path
        exists[i] <- TRUE
        next
      }

      # Fuzzy matching for files with modified names
      potential_matches <- all_img_files[grepl(sub("\\.[^.]*$", "", json_basename), actual_basenames, fixed = TRUE)]

      if (length(potential_matches) > 0) {
        matched_paths[i] <- potential_matches[1]
        exists[i] <- TRUE
      } else {
        exists[i] <- FALSE
      }
    }

    # Filter to existing images only
    self$images <- self$images[exists, , drop = FALSE]
    matched_paths <- matched_paths[exists]
    keep_ids <- self$images$id
    self$annotations <- self$annotations[self$annotations$image_id %in% keep_ids, , drop = FALSE]

    # Group annotations by image ID for efficient lookup
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

    # Load image using multiple fallback methods
    x <- tryCatch({
      if (grepl("\\.jpe?g$", img_path, ignore.case = TRUE)) {
        jpeg::readJPEG(img_path)
      } else if (grepl("\\.png$", img_path, ignore.case = TRUE)) {
        png::readPNG(img_path)
      } else {
        stop("Unsupported format")
      }
    }, error = function(e) {
      tryCatch({
        if (requireNamespace("magick", quietly = TRUE)) {
          img <- magick::image_read(img_path)
          arr <- magick::image_data(img, channels = "rgb")
          aperm(as.integer(arr)/255, c(3,2,1))
        } else {
          runtime_error("Failed to read image: ", img_path)
        }
      }, error = function(e2) {
        runtime_error("Failed to read image with all methods: ", img_path)
      })
    })

    # Convert grayscale to RGB if needed
    if (length(dim(x)) == 2L) {
      x <- array(rep(x, each = 3), dim = c(dim(x), 3L))
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
