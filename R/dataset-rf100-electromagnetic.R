#' @include dataset-rf100-underwater.R
NULL

#' RF100 Electromagnetic Dataset Collection
#'
#' Loads one of the RF100 electromagnetic object detection datasets (COCO format),
#' with per-dataset folders and train/valid/test splits.
#'
#' @param dataset One of "thermal_dog_and_people", "solar_panel", "radio_signal",
#'   "thermal_cheetah", "rheumatology", "knee", "abdomen_mri",
#'   "brain_axial_mri", "gynecology_mri", "brain_tumor", "fracture",
#'   or "ir_object".
#' @inheritParams rf100_underwater_collection
#' @inherit rf100_underwater_collection return
#'
#' @examples
#' \dontrun{
#' devtools::load_all()
#' ds <- rf100_electromagnetic_collection(
#'   dataset = "thermal_dog_and_people",
#'   split = "test",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#' item <- ds[1]
#' item$y$labels
#' item$y$boxes
#' boxed_img <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed_img)
#' }
#' @family detection_dataset
#' @export
rf100_electromagnetic_collection <- torch::dataset(
  name = "rf100_electromagnetic_collection",
  inherit = rf100_underwater_collection,

  resources = {
    base_url <- "https://huggingface.co/datasets/akankshakoshti/rf100-electromagnetic/resolve/main/"
    files <- c(
      "thermal-dogs-and-people-x6ejw.zip",
      "solar-panels-taxvb.zip",
      "radio-signal.zip",
      "thermal-cheetah-my4dp.zip",
      "x-ray-rheumatology.zip",
      "acl-x-ray.zip",
      "abdomen-mri.zip",
      "axial-mri.zip",
      "gynecology-mri.zip",
      "brain-tumor-m2pbp.zip",
      "bone-fracture-7fylg.zip",
      "flir-camera-objects.zip"
    )
    data.frame(
      dataset = c(
        "thermal_dog_and_people", "solar_panel", "radio_signal",
        "thermal_cheetah", "rheumatology", "knee",
        "abdomen_mri", "brain_axial_mri", "gynecology_mri",
        "brain_tumor", "fracture", "ir_object"
      ),
      url = paste0(base_url, files, "?download=1"),
      md5 = NA_character_,
      stringsAsFactors = FALSE
    )
  },

  initialize = function(
    dataset = c(
      "thermal_dog_and_people", "solar_panel", "radio_signal",
      "thermal_cheetah", "rheumatology", "knee",
      "abdomen_mri", "brain_axial_mri", "gynecology_mri",
      "brain_tumor", "fracture", "ir_object"
    ),
    split = c("train", "test", "valid"),
    root = tempdir(),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- match.arg(dataset)
    self$split   <- match.arg(split)
    self$root    <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    # Dataset-scoped dirs; tolerate optional double nesting
    self$dataset_dir <- fs::path(self$root, "rf100-electromagnetic", self$dataset)
    self$split_dir   <- fs::path(self$dataset_dir, self$split)
    if (!fs::dir_exists(self$split_dir)) {
      alt <- fs::path(self$dataset_dir, self$dataset, self$split)
      if (fs::dir_exists(alt)) self$split_dir <- alt
    }
    self$image_dir <- self$split_dir
    self$annotation_file <- fs::path(self$split_dir, "_annotations.coco.json")

    res <- self$resources[self$resources$dataset == self$dataset, , drop = FALSE]
    self$archive_url <- res$url

    if (download) {
      if (is.na(self$archive_url) || !nzchar(self$archive_url)) {
        runtime_error(paste0("No download URL for dataset '", self$dataset, "'."))
      }
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. Use download=TRUE or check paths.")
    self$load_annotations()
  },

  check_exists = function() {
    if (super$check_exists()) {
      return(TRUE)
    }

    if (!fs::dir_exists(self$dataset_dir)) {
      return(FALSE)
    }

    split_candidates <- if (identical(self$split, "valid")) c("valid", "val") else self$split
    split_pat <- if (length(split_candidates) > 1) {
      paste0("(", paste(split_candidates, collapse = "|"), ")")
    } else {
      split_candidates
    }

    hits <- fs::dir_ls(
      self$dataset_dir,
      recurse = 3,
      type = "file",
      regexp = paste0("[/\\]", split_pat, "[/\\]_annotations\\.coco\\.json$")
    )

    if (length(hits) >= 1) {
      self$annotation_file <- hits[[1]]
      self$split_dir <- fs::path_dir(self$annotation_file)
      self$image_dir <- self$split_dir
      return(super$check_exists())
    }

    FALSE
  },

  .getitem = function(index) {
    img_path <- self$image_paths[index]
    ext <- tolower(fs::path_ext(img_path))
    x <- if (ext %in% c("jpg","jpeg")) jpeg::readJPEG(img_path)
    else if (ext == "png")        png::readPNG(img_path)
    else                          jpeg::readJPEG(img_path)
    if (length(dim(x)) == 3 && dim(x)[3] == 4) x <- x[,,1:3, drop = FALSE]
    if (length(dim(x)) == 2) x <- array(rep(x, 3L), dim = c(dim(x), 3L))

    img_info <- self$images[index, ]
    anns <- self$annotations_by_image[[as.character(img_info$id)]]

    if (is.null(anns) || nrow(anns) == 0) {
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
    } else {
      boxes_xywh <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float())
      boxes <- box_xywh_to_xyxy(boxes_xywh)
      labels <- as.character(self$categories$name[match(anns$category_id, self$categories$id)])
    }

    y <- list(labels = labels, boxes = boxes)
    if (!is.null(self$transform))
      x <- self$transform(x)
    if (!is.null(self$target_transform))
      y <- self$target_transform(y)
    structure(list(x = x, y = y), class = "image_with_bounding_box")
  }
)
