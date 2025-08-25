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
#' @param debug Logical. If TRUE, prints diagnostic information during
#'   download and setup.
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
#'   dataset = "objects",  # Fixed: use "objects" not "object"
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
    target_transform = NULL,
    debug = FALSE
  ) {
    self$dataset <- match.arg(dataset)
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform
    self$debug <- debug

    fs::dir_create(self$root, recurse = TRUE)
    self$dataset_dir <- fs::path(self$root, paste0("rf100_underwater_", self$dataset))

    resource <- self$resources[self$resources$dataset == self$dataset, , drop = FALSE]
    self$archive_url <- resource$url

    if (download) self$download()

    if (!self$check_exists()) {
      if (self$debug) {
        contents <- try(fs::dir_ls(self$dataset_dir, recurse = TRUE), silent = TRUE)
        if (!inherits(contents, "try-error")) {
          message("Dataset directory contents:\n", paste(contents, collapse = "\n"))
        }
      }
      runtime_error("Dataset not found. You can use `download = TRUE` to download it. Checked directory: {self$dataset_dir}")
    }

    self$load_annotations()
  },

  download = function() {
    if (self$check_exists()) return(invisible(NULL))

    if (fs::dir_exists(self$dataset_dir)) fs::dir_delete(self$dataset_dir)
    fs::dir_create(self$dataset_dir, recurse = TRUE)

    # Download the archive with retries and curl fallback for large files
    dest <- fs::path(self$dataset_dir, "dataset.tar.gz")
    attempts <- 3L
    success <- FALSE
    last_err <- NULL

    if (self$debug) {
      message("Downloading dataset from: ", self$archive_url)
      message("Destination: ", dest)
    }

    for (i in seq_len(attempts)) {
      last_err <- tryCatch({
        if (requireNamespace("curl", quietly = TRUE)) {
          curl::curl_download(self$archive_url, dest, quiet = !self$debug)
        } else {
          download.file(self$archive_url, dest, mode = "wb", quiet = !self$debug)
        }
        if (fs::file_exists(dest) && fs::file_size(dest) > 0) {
          success <- TRUE
          break
        } else {
          stop("empty download")
        }
      }, error = function(e) e)
      if (!success) Sys.sleep(i)
    }
    if (!success) {
      runtime_error("Failed to download dataset archive from {self$archive_url}. {conditionMessage(last_err)}")
    }

    if (self$debug) {
      message("Downloaded archive size: ", fs::file_size(dest), " bytes")
    }

    # Extract the archive - use a shorter temp path on Windows to avoid path length issues
    if (.Platform$OS.type == "windows") {
      tmp_extract <- fs::path("C:/temp_rf100")
    } else {
      tmp_extract <- fs::path_temp("rf100_extract")
    }

    if (fs::dir_exists(tmp_extract)) fs::dir_delete(tmp_extract)
    fs::dir_create(tmp_extract)

    if (self$debug) {
      message("Extracting to temporary directory: ", tmp_extract)
    }

    extract_success <- FALSE

    # Try archive package first (better Windows support)
    if (requireNamespace("archive", quietly = TRUE)) {
      extract_success <- tryCatch({
        archive::archive_extract(dest, dir = tmp_extract)
        TRUE
      }, error = function(e) {
        if (self$debug) {
          message("archive extraction failed: ", conditionMessage(e))
        }
        FALSE
      })
    }

    # Fallback to untar if archive package failed or isn't available
    if (!extract_success) {
      extract_success <- tryCatch({
        # On Windows, try with additional options to handle long paths
        if (.Platform$OS.type == "windows") {
          # Try different tar options for Windows
          system2("tar", args = c("-xf", shQuote(dest), "-C", shQuote(tmp_extract)),
                  stdout = FALSE, stderr = FALSE)
          TRUE
        } else {
          utils::untar(dest, exdir = tmp_extract)
          TRUE
        }
      }, error = function(e) {
        if (self$debug) {
          message("untar failed: ", conditionMessage(e))
        }
        FALSE
      })
    }

    if (!extract_success) {
      runtime_error("Failed to extract archive. The downloaded file may be corrupted.")
    }

    # Find the extracted dataset directory
    extracted_dirs <- fs::dir_ls(tmp_extract, type = "directory")
    if (self$debug) {
      message("Extracted directories: ", paste(extracted_dirs, collapse = ", "))
    }

    # Look for annotation files with multiple possible patterns
    # Even if extraction had errors, some files may have been extracted
    all_files <- character(0)
    if (fs::dir_exists(tmp_extract)) {
      all_files <- fs::dir_ls(tmp_extract, recurse = TRUE)
    }

    if (self$debug) {
      message("Found ", length(all_files), " extracted files")
      if (length(all_files) > 0) {
        message("Sample extracted files:\n", paste(head(all_files, 10), collapse = "\n"))
        if (length(all_files) > 10) {
          message("... and ", length(all_files) - 10, " more files")
        }
      }
    }

    # Find annotation files with different possible patterns
    ann_patterns <- c(
      "_annotations\\.coco\\.json$",
      "_annotations\\.json$",
      "annotations\\.json$",
      "\\.json$"
    )

    ann_files <- character(0)
    for (pattern in ann_patterns) {
      if (fs::dir_exists(tmp_extract)) {
        potential_ann_files <- fs::dir_ls(tmp_extract, recurse = TRUE, regexp = pattern, type = "file")
        # Filter to only files that likely contain annotation data
        for (file in potential_ann_files) {
          if (grepl("annotation", tolower(fs::path_file(file))) ||
              grepl("(train|test|valid)", fs::path_dir(file))) {
            ann_files <- c(ann_files, file)
          }
        }
      }
      if (length(ann_files) > 0) {
        if (self$debug) {
          message("Found ", length(ann_files), " files matching pattern: ", pattern)
          message("Annotation files: ", paste(ann_files, collapse = ", "))
        }
        break
      }
    }

    # If no annotation files found, try to recreate them from the archive manually
    if (length(ann_files) == 0 && requireNamespace("archive", quietly = TRUE)) {
      if (self$debug) {
        message("No annotation files found, attempting to extract annotation files directly...")
      }

      # List archive contents to find annotation files
      archive_info <- tryCatch(archive::archive(dest), error = function(e) NULL)
      if (!is.null(archive_info)) {
        ann_entries <- archive_info[grepl("_annotations\\.coco\\.json$", archive_info$path), ]

        if (nrow(ann_entries) > 0) {
          # Extract just the annotation files to a simpler path structure
          for (i in seq_len(nrow(ann_entries))) {
            ann_path <- ann_entries$path[i]
            # Create a simplified path
            path_parts <- strsplit(ann_path, "/")[[1]]
            split_name <- path_parts[length(path_parts) - 1]  # Should be train/test/valid
            simple_path <- fs::path(tmp_extract, split_name, "_annotations.coco.json")

            fs::dir_create(fs::path_dir(simple_path), recurse = TRUE)

            tryCatch({
              archive::archive_extract(dest, files = ann_path, dir = fs::path_temp())
              temp_ann <- fs::path(fs::path_temp(), ann_path)
              if (fs::file_exists(temp_ann)) {
                fs::file_copy(temp_ann, simple_path, overwrite = TRUE)
                ann_files <- c(ann_files, simple_path)
              }
            }, error = function(e) {
              if (self$debug) {
                message("Failed to extract annotation file ", ann_path, ": ", conditionMessage(e))
              }
            })
          }
        }
      }
    }

    if (length(ann_files) == 0) {
      runtime_error("No annotation files found in extracted archive. The download may have failed or the archive structure is unexpected.")
    }

    # Find the root directory that contains the dataset
    dataset_source <- fs::path_dir(ann_files[1])

    # If annotation is in a split directory, go up one level
    if (tolower(fs::path_file(dataset_source)) %in% c("train", "test", "valid")) {
      dataset_source <- fs::path_dir(dataset_source)
    }

    if (self$debug) {
      message("Selected dataset source directory: ", dataset_source)
    }

    # Copy the dataset to the final location
    if (dataset_source == tmp_extract) {
      # Files are directly in temp directory
      fs::dir_copy(tmp_extract, self$dataset_dir, overwrite = TRUE)
    } else {
      # Files are in a subdirectory
      fs::dir_copy(dataset_source, self$dataset_dir, overwrite = TRUE)
    }

    # Clean up - close any file handles first
    gc()  # Force garbage collection to release file handles

    # Remove archive file with retry logic
    for (retry in 1:3) {
      remove_success <- tryCatch({
        if (fs::file_exists(dest)) {
          fs::file_delete(dest)
        }
        TRUE
      }, error = function(e) {
        if (self$debug) {
          message("Attempt ", retry, " to remove archive failed: ", conditionMessage(e))
        }
        FALSE
      })

      if (remove_success) break

      if (retry < 3) {
        Sys.sleep(1)  # Wait a second before retrying
        gc()  # Try garbage collection again
      } else if (self$debug) {
        message("Warning: Could not remove archive file. It may need to be manually deleted.")
      }
    }

    # Remove temp directory
    tryCatch({
      fs::dir_delete(tmp_extract)
    }, error = function(e) {
      if (self$debug) {
        message("Failed to remove temp directory: ", conditionMessage(e))
      }
    })

    if (self$debug) {
      extracted <- fs::dir_ls(self$dataset_dir, recurse = TRUE)
      message("Final extracted files:\n", paste(extracted, collapse = "\n"))
    }

    invisible(NULL)
  },

  check_exists = function() {
    ann <- self$discover_annotation_file()
    if (self$debug) {
      message("Annotation file discovered: ", ann)
    }
    if (is.na(ann) || !fs::file_exists(ann)) return(FALSE)

    self$annotation_file <- ann
    self$split_dir <- fs::path_dir(ann)
    candidate_img_dirs <- c(self$split_dir, fs::path(self$split_dir, "images"))
    self$image_dir <- candidate_img_dirs[fs::dir_exists(candidate_img_dirs)][1]
    if (self$debug) {
      message("Image directory selected: ", self$image_dir)
    }

    fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)
  },

  discover_annotation_file = function() {
    if (!fs::dir_exists(self$dataset_dir)) return(NA_character_)

    # Try multiple patterns for annotation files
    ann_patterns <- c(
      "_annotations\\.coco\\.json$",
      "_annotations\\.json$",
      "annotations\\.json$"
    )

    jsons <- character(0)
    for (pattern in ann_patterns) {
      jsons <- fs::dir_ls(self$dataset_dir, recurse = TRUE, type = "file", regexp = pattern)
      if (length(jsons) > 0) break
    }

    if (self$debug) {
      message("Annotation candidates:\n", paste(jsons, collapse = "\n"))
    }

    if (!length(jsons)) {
      # Try finding any JSON files as a last resort
      jsons_alt <- fs::dir_ls(self$dataset_dir, recurse = TRUE, type = "file", regexp = "\\.json$")
      if (self$debug) {
        message("No annotation files found with expected patterns under ", self$dataset_dir)
        message("All JSON candidates:\n", paste(jsons_alt, collapse = "\n"))
      }

      # Filter JSON files that might be annotations (contain "annotations" or split names)
      for (json_file in jsons_alt) {
        json_content <- tryCatch({
          jsonlite::fromJSON(json_file, simplifyVector = FALSE)
        }, error = function(e) NULL)

        if (!is.null(json_content) &&
            ("annotations" %in% names(json_content) ||
             "images" %in% names(json_content) ||
             "categories" %in% names(json_content))) {
          jsons <- c(jsons, json_file)
        }
      }

      if (!length(jsons)) {
        return(NA_character_)
      }
    }

    # Prefer files that match the split name
    jsons_split <- jsons[grepl(paste0("[/\\\\]", self$split, "[/\\\\]"), jsons) |
                           grepl(paste0(self$split, ".*\\.json$"), fs::path_file(jsons))]
    if (length(jsons_split)) return(jsons_split[[1]])

    # Return the first annotation file found
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
    all_img_files <- fs::dir_ls(
      self$dataset_dir,
      type = "file",
      recurse = TRUE,
      regexp = "(?i)\\.(jpg|jpeg|png|bmp)$"
    )

    # Also try the specific image directory if it exists
    if (!is.na(self$image_dir) && fs::dir_exists(self$image_dir)) {
      local_img_files <- fs::dir_ls(
        self$image_dir,
        type = "file",
        regexp = "(?i)\\.(jpg|jpeg|png|bmp)$"
      )
      all_img_files <- unique(c(all_img_files, local_img_files))
    }

    actual_basenames <- tolower(fs::path_file(all_img_files))

    # Since the JSON filenames already include the RoboFlow format,
    # we can try direct matching first
    matched_paths <- character(nrow(self$images))
    exists <- logical(nrow(self$images))

    # Create a lookup table for faster matching
    filename_to_path <- setNames(all_img_files, actual_basenames)

    for (i in seq_len(nrow(self$images))) {
      json_filename <- self$images$file_name[i]
      json_basename <- tolower(fs::path_file(json_filename))

      # Direct filename match using basename
      if (json_basename %in% names(filename_to_path)) {
        matched_paths[i] <- filename_to_path[[json_basename]]
        exists[i] <- TRUE
        next
      }

      # Try path relative to dataset root (the JSON may contain directories)
      full_path <- fs::path(self$dataset_dir, json_filename)
      if (fs::file_exists(full_path)) {
        matched_paths[i] <- full_path
        exists[i] <- TRUE
        next
      }

      # Try searching in the split's image directory
      if (!is.na(self$image_dir)) {
        direct_path <- fs::path(self$image_dir, json_basename)
        if (fs::file_exists(direct_path)) {
          matched_paths[i] <- direct_path
          exists[i] <- TRUE
          next
        }
      }

      # Fuzzy matching - extract base name and look for partial matches
      json_base <- sub("_[lr]gb_jpg\\.rf\\.[^.]+\\.jpg$", "", json_basename)
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
        } else runtime_error("Failed to read image: {img_path}")
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
