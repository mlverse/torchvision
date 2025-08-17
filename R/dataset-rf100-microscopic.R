#' @include dataset-rf100-underwater.R
NULL

#' RF100 Microscopic Dataset Collection
#'
#' Loads one of the RF100 microscopic object detection datasets (COCO format),
#' with per-dataset folders and train/valid/test splits.
#'
#' @param dataset One of "stomata_cell", "blood_cell", "parasite",
#'   "cell", "liquid_crystals", "bacteria", "cotton_desease",
#'   "mitosis", "phage", "liver_desease", or "asbestos".
#' @inheritParams rf100_underwater_collection
#' @inherit rf100_underwater_collection return
#' @family detection_dataset
#' @export
rf100_microscopic_collection <- torch::dataset(
  name = "rf100_microscopic_collection",
  inherit = rf100_underwater_collection,

  resources = data.frame(
    dataset = c(
      "stomata_cell", "blood_cell", "parasite", "cell",
      "liquid_crystals", "bacteria", "cotton_desease",
      "mitosis", "phage", "liver_desease", "asbestos"
    ),
    url = c(
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/stomata-cells.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/bccd-ouzjz.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/parasites-1s07h.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/cells-uyemf.zip?download=1",
      NA_character_,  # liquid_crystals not present in repo
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/bacteria-ptywi.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/cotton-plant-disease.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/mitosis-gjs3g.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/phages.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/liver-disease.zip?download=1",
      "https://huggingface.co/datasets/akankshakoshti/rf100-microscopic/resolve/main/asbestos.zip?download=1"
    ),
    md5 = NA_character_
  ),

  initialize = function(
    dataset = c(
      "stomata_cell", "blood_cell", "parasite", "cell",
      "liquid_crystals", "bacteria", "cotton_desease",
      "mitosis", "phage", "liver_desease", "asbestos"
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
    self$dataset_dir <- fs::path(self$root, "rf100-microscopic", self$dataset)
    self$split_dir   <- fs::path(self$dataset_dir, self$split)
    if (!fs::dir_exists(self$split_dir)) {
      alt <- fs::path(self$dataset_dir, self$dataset, self$split)
      if (fs::dir_exists(alt)) self$split_dir <- alt
    }
    self$image_dir <- self$split_dir
    self$annotation_file <- fs::path(self$split_dir, "_annotations.coco.json")

    # URL for this dataset
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
    # If the dataset directory itself doesn't exist yet, nothing exists.
    if (!fs::dir_exists(self$dataset_dir)) {
      return(FALSE)
    }

    # Fast path: expected files/dirs already in place
    if (fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)) {
      return(TRUE)
    }

    # Accept both 'valid' and 'val'
    split_candidates <- if (identical(self$split, "valid")) c("valid", "val") else self$split
    split_pat <- if (length(split_candidates) > 1) {
      paste0("(", paste(split_candidates, collapse = "|"), ")")
    } else {
      split_candidates
    }

    # Search for '<split>/_annotations.coco.json' anywhere under dataset_dir
    hits <- fs::dir_ls(
      self$dataset_dir,
      recurse = 3,
      type = "file",
      regexp = paste0("[/\\\\]", split_pat, "[/\\\\]_annotations\\.coco\\.json$")
    )

    if (length(hits) >= 1) {
      self$annotation_file <- hits[[1]]
      self$split_dir <- fs::path_dir(self$annotation_file)
      self$image_dir <- self$split_dir
      return(TRUE)
    }

    FALSE
  },

  .getitem = function(index) {
    img_path <- self$image_paths[index]
    ext <- tolower(fs::path_ext(img_path))
    x <- if (ext %in% c("jpg","jpeg")) jpeg::readJPEG(img_path)
    else if (ext == "png")        png::readPNG(img_path)
    else                          jpeg::readJPEG(img_path) # fallback

    if (length(dim(x)) == 3 && dim(x)[3] == 4) x <- x[,,1:3, drop = FALSE] # drop alpha
    if (length(dim(x)) == 2) x <- array(rep(x, 3L), dim = c(dim(x), 3L))   # gray->RGB

    img_info <- self$images[index, ]
    anns <- self$annotations_by_image[[as.character(img_info$id)]]

    if (is.null(anns) || nrow(anns) == 0) {
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
    } else {
      boxes_xywh <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float())
      # xywh -> xyxy (same logic as parent)
      boxes <- torch::torch_stack(
        list(boxes_xywh[,1],
             boxes_xywh[,2],
             boxes_xywh[,1] + boxes_xywh[,3],
             boxes_xywh[,2] + boxes_xywh[,4]),
        dim = 2
      )
      labels <- as.character(self$categories$name[match(anns$category_id, self$categories$id)])
    }

    y <- list(labels = labels, boxes = boxes)
    if (!is.null(self$transform)) x <- self$transform(x)
    if (!is.null(self$target_transform)) y <- self$target_transform(y)
    structure(list(x = x, y = y), class = "image_with_bounding_box")
  }
)
