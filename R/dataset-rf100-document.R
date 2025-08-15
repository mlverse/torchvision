#' RF100 Document Collection Dataset
#'
#' Loads one of the document datasets from the RF100 collection for object detection.
#'
#' @param root Root directory where the dataset is stored or will be downloaded to.
#' @param dataset Character. Name of the dataset to load. One of
#'   c("tweeter_post", "tweeter_profile", "document_part", "activity_diagram",
#'     "signature", "paper_part", "tabular_data", "paragraph").
#' @param split Dataset split to load. One of \code{"train"}, \code{"valid"}, or
#'   \code{"test"}. Defaults to \code{"train"}.
#' @param download Logical. If TRUE, downloads the dataset from Hugging Face.
#' @param transform Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target.
#' @param debug Logical. If TRUE, prints additional debugging information about the
#'   download and extraction steps.
#'
#' @return A torch dataset where each item is a list with elements:
#' - `x`: a `(C, H, W)` float tensor representing the image.
#' - `y$boxes`: a `(N, 4)` float tensor of bounding boxes in `c(xmin, ymin, xmax, ymax)` format.
#' - `y$labels`: a character vector with class names for each box.
#'
#' @examples
#' \donttest{
#' ds <- rf100_document_collection(dataset = "tweeter_post", download = TRUE)
#' item <- ds[1]
#' boxed <- draw_bounding_boxes(item)
#' tensor_image_display(boxed)
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
    repo = c(
      "Francesco/tweeter-posts",
      "Francesco/tweeter-profile",
      "Francesco/document-parts",
      "Francesco/activity-diagrams-qdobr",
      "Francesco/signatures-xc8up",
      "Francesco/paper-parts",
      "Francesco/tabular-data-wf9uh",
      "Francesco/paragraphs-co84b"
    )
  ),

  initialize = function(
    root = tempdir(),
    dataset = c(
      "tweeter_post", "tweeter_profile", "document_part",
      "activity_diagram", "signature", "paper_part",
      "tabular_data", "paragraph"
    ),
    split = c("train", "valid", "test"),
    download = FALSE,
    transform = NULL,
    target_transform = NULL,
    debug = FALSE
  ) {
    dataset <- match.arg(dataset)
    split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$dataset <- dataset
    self$split <- split
    self$transform <- transform
    self$target_transform <- target_transform
    self$debug <- debug

    info <- self$resources[self$resources$dataset == dataset, ]
    self$repo <- info$repo
    self$folder <- fs::path_file(self$repo)
    self$data_dir <- fs::path(self$root, dataset, split)
    self$image_dir <- fs::path(self$data_dir, "images")
    self$annotation_file <- NULL

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
    cli_inform("{.cls {class(self)[[1]]}} dataset '{dataset}' loaded with {length(self$image_ids)} images.")
  },

  check_exists = function() {
    if (!fs::dir_exists(self$data_dir)) {
      if (self$debug) cli_inform("Debug: data directory {self$data_dir} missing")
      return(FALSE)
    }
    if (fs::dir_exists(fs::path(self$data_dir, "images")))
      self$image_dir <- fs::path(self$data_dir, "images")
    else
      self$image_dir <- self$data_dir
    jsons <- fs::dir_ls(self$data_dir, glob = "*.json", type = "file")
    if (length(jsons) == 0) {
      if (self$debug) cli_inform("Debug: no annotation files found in {self$data_dir}")
      return(FALSE)
    }
    self$annotation_file <- jsons[1]
    exists <- fs::dir_exists(self$image_dir) && fs::file_exists(self$annotation_file)
    if (self$debug) cli_inform("Debug: images dir {self$image_dir} exists={fs::dir_exists(self$image_dir)}, annotation file exists={fs::file_exists(self$annotation_file)}")
    exists
  },

  download = function() {
    if (self$check_exists()) {
      return()
    }

    if (!requireNamespace("curl", quietly = TRUE)) {
      install.packages("curl")
    }

    cli_inform("Downloading {.cls {class(self)[[1]]}} from Hugging Face ...")
    url <- sprintf(
      "https://huggingface.co/datasets/%s/resolve/main/dataset.tar.gz",
      self$repo
    )

    cache_dir <- file.path(Sys.getenv("HOME"), ".cache")
    if (!dir.exists(cache_dir))
      dir.create(cache_dir, recursive = TRUE)
    archive <- file.path(cache_dir, paste0(self$dataset, ".tar.gz"))
    if (!file.exists(archive)) {
      if (self$debug) cli_inform("Debug: downloading archive to {archive}")
      curl::curl_download(url, destfile = archive)
    } else if (self$debug) {
      cli_inform("Debug: using cached archive {archive}")
    }

    tmp_dir <- tempfile()
    dir.create(tmp_dir)
    if (self$debug) cli_inform("Debug: extraction temp dir {tmp_dir}")

    # Extract the archive using R's internal tar to handle absolute paths better
    tryCatch(
      utils::untar(archive, exdir = tmp_dir, tar = "internal"),
      error = function(e) {
        if (self$debug) cli_inform("Debug: internal tar failed, trying external tar")
        # Fallback to external tar without special options
        tryCatch(
          utils::untar(archive, exdir = tmp_dir),
          error = function(e2) {
            cli_abort("Failed to extract archive {archive}: {e2$message}")
          }
        )
      }
    )

    # Find the actual dataset folder structure
    extracted_files <- list.files(tmp_dir, recursive = TRUE, full.names = TRUE)
    if (self$debug) cli_inform("Debug: extracted {length(extracted_files)} files")

    # Look for the dataset folder pattern
    dataset_folders <- list.dirs(tmp_dir, recursive = TRUE, full.names = TRUE)
    target_folder <- NULL

    # Find folder containing the dataset splits (train, valid, test)
    for (folder in dataset_folders) {
      subfolders <- basename(list.dirs(folder, recursive = FALSE))
      if (any(c("train", "valid", "test") %in% subfolders)) {
        target_folder <- folder
        break
      }
    }

    if (is.null(target_folder)) {
      # Fallback: look for any folder with the expected name pattern
      pattern_matches <- grep(self$folder, dataset_folders, value = TRUE)
      if (length(pattern_matches) > 0) {
        target_folder <- pattern_matches[1]
      } else {
        # Last resort: use the first subdirectory that's not empty
        non_empty_dirs <- dataset_folders[sapply(dataset_folders, function(d) length(list.files(d)) > 0)]
        if (length(non_empty_dirs) > 0) {
          target_folder <- non_empty_dirs[1]
        }
      }
    }

    if (is.null(target_folder) || !dir.exists(target_folder)) {
      cli_abort("Failed to locate dataset folder in the extracted archive. Available folders: {paste(basename(dataset_folders), collapse=', ')}")
    }

    if (self$debug) cli_inform("Debug: found dataset folder at {target_folder}")

    # Copy the dataset to the final destination
    dest_dir <- fs::path(self$root, self$dataset)
    if (self$debug) cli_inform("Debug: copying dataset from {target_folder} to {dest_dir}")

    fs::dir_create(fs::path(self$root), recurse = TRUE)

    # If target_folder contains train/valid/test directly, copy its contents
    # Otherwise copy the folder itself and rename it
    subfolders <- basename(list.dirs(target_folder, recursive = FALSE))
    if (any(c("train", "valid", "test") %in% subfolders)) {
      fs::dir_copy(target_folder, dest_dir, overwrite = TRUE)
    } else {
      # Look one level deeper
      deeper_folders <- list.dirs(target_folder, recursive = FALSE, full.names = TRUE)
      found_dataset <- FALSE
      for (deeper in deeper_folders) {
        deeper_subfolders <- basename(list.dirs(deeper, recursive = FALSE))
        if (any(c("train", "valid", "test") %in% deeper_subfolders)) {
          fs::dir_copy(deeper, dest_dir, overwrite = TRUE)
          found_dataset <- TRUE
          break
        }
      }
      if (!found_dataset) {
        fs::dir_copy(target_folder, dest_dir, overwrite = TRUE)
      }
    }

    # Clean up temporary directory
    unlink(tmp_dir, recursive = TRUE)

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  load_annotations = function() {
    data <- jsonlite::fromJSON(self$annotation_file)
    self$image_metadata <- setNames(
      split(data$images, seq_len(nrow(data$images))),
      as.character(data$images$id)
    )
    self$annotations <- data$annotations
    self$categories <- data$categories
    self$category_names <- setNames(self$categories$name, self$categories$id)

    ids <- as.numeric(names(self$image_metadata))
    image_paths <- fs::path(self$image_dir, sapply(ids, function(id) self$image_metadata[[as.character(id)]]$file_name))
    exist <- fs::file_exists(image_paths)
    self$image_ids <- ids[exist]
  },

  .getitem = function(index) {
    image_id <- self$image_ids[index]
    image_info <- self$image_metadata[[as.character(image_id)]]
    img_path <- fs::path(self$image_dir, image_info$file_name)

    x <- jpeg::readJPEG(img_path)
    if (length(dim(x)) == 2) {
      x <- array(rep(x, 3), dim = c(dim(x), 3))
    }
    x <- aperm(x, c(3, 1, 2))
    x <- torch::torch_tensor(x, dtype = torch::torch_float())

    anns <- self$annotations[self$annotations$image_id == image_id, ]
    if (nrow(anns) > 0) {
      boxes_wh <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float())
      boxes <- box_xywh_to_xyxy(boxes_wh)
      labels <- as.character(self$category_names[match(anns$category_id, names(self$category_names))])
    } else {
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
    }
    y <- list(
      boxes = boxes,
      labels = labels
    )

    if (!is.null(self$transform))
      x <- self$transform(x)
    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    structure(list(x = x, y = y), class = "image_with_bounding_box")
  },

  .length = function() {
    length(self$image_ids)
  }
)
