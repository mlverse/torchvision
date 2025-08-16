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
      self$download()
    }

    if (!fs::dir_exists(self$data_dir)) {
      alt_dir <- fs::path(self$root, dataset)
      if (fs::dir_exists(alt_dir)) {
        self$data_dir <- alt_dir
        self$image_dir <- fs::path(self$data_dir, "images")
      }
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
  },

  check_exists = function() {
    if (!fs::dir_exists(self$data_dir)) {
      return(FALSE)
    }
    if (fs::dir_exists(fs::path(self$data_dir, "images")))
      self$image_dir <- fs::path(self$data_dir, "images")
    else
      self$image_dir <- self$data_dir
    jsons <- fs::dir_ls(self$data_dir, glob = "*.json", type = "file")
    if (length(jsons) == 0) {
      return(FALSE)
    }
    self$annotation_file <- jsons[1]
    fs::dir_exists(self$image_dir) && fs::file_exists(self$annotation_file)
  },

  download = function() {
    if (self$check_exists()) {
      return()
    }

    if (!requireNamespace("curl", quietly = TRUE)) {
      install.packages("curl")
    }

    url <- sprintf(
      "https://huggingface.co/datasets/%s/resolve/main/dataset.tar.gz",
      self$repo
    )

    cache_dir <- file.path(Sys.getenv("HOME"), ".cache")
    if (!dir.exists(cache_dir))
      dir.create(cache_dir, recursive = TRUE)
    archive <- file.path(cache_dir, paste0(self$dataset, ".tar.gz"))

    if (!file.exists(archive)) {
      tryCatch({
        curl::curl_fetch_memory(url, handle = curl::new_handle(nobody = TRUE))
      }, error = function(e) {
        cli_abort("Failed to access dataset URL {url}. Error: {conditionMessage(e)}")
      })

      curl::curl_download(url, destfile = archive)
    }

    dest_dir <- fs::path(self$root, self$dataset)

    if (fs::dir_exists(dest_dir)) {
      fs::dir_delete(dest_dir)
    }
    fs::dir_create(dest_dir, recurse = TRUE)

    tmp_dir <- file.path(tempdir(), paste0("rf", sample(1000:9999, 1)))
    dir.create(tmp_dir, recursive = TRUE)

    res <- try({
      utils::untar(archive, exdir = tmp_dir, tar = "internal")
    }, silent = TRUE)

    if (inherits(res, "try-error")) {
      res <- try({
        utils::untar(archive, exdir = tmp_dir)
      }, silent = TRUE)

      if (inherits(res, "try-error")) {
        msg <- conditionMessage(attr(res, "condition"))
        cli_abort("Failed to extract archive {archive}: {msg}")
      }
    }

    all_dirs <- list.dirs(tmp_dir, recursive = TRUE, full.names = TRUE)
    target_folder <- NULL

    for (folder in all_dirs) {
      subfolders <- basename(list.dirs(folder, recursive = FALSE))
      if (any(c("train", "valid", "test") %in% subfolders)) {
        target_folder <- folder
        break
      }
    }

    if (is.null(target_folder)) {
      pattern_matches <- grep(self$folder, all_dirs, value = TRUE, fixed = TRUE)
      if (length(pattern_matches) > 0) {
        target_folder <- pattern_matches[1]
      }
    }

    if (is.null(target_folder) || !dir.exists(target_folder)) {
      available_folders <- basename(all_dirs[nchar(basename(all_dirs)) > 0])
      cli_abort("Failed to locate dataset folder. Available folders: {paste(head(available_folders, 20), collapse=', ')}")
    }

    split_dirs <- c("train", "valid", "test")
    for (split in split_dirs) {
      src_split <- file.path(target_folder, split)
      if (dir.exists(src_split)) {
        dest_split <- file.path(dest_dir, split)

        dir.create(dest_split, recursive = TRUE, showWarnings = FALSE)

        src_files <- list.files(src_split, full.names = TRUE, recursive = FALSE)
        for (src_file in src_files) {
          if (dir.exists(src_file)) {
            file.copy(src_file, dest_split, recursive = TRUE, overwrite = TRUE)
          } else {
            file.copy(src_file, file.path(dest_split, basename(src_file)), overwrite = TRUE)
          }
        }
      }
    }

    unlink(tmp_dir, recursive = TRUE)
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
