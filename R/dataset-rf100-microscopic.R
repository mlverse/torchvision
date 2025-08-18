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
#'
#' @examples
#' \dontrun{
#' devtools::load_all()
#' ds <- rf100_microscopic_collection(
#'   dataset = "stomata_cell",
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
    md5 = c(
      "86ad78350a356964db56311f22a254a4",
      "c3cf093a9b9ad04eae284f89486f04d3",
      "3c221d7a4c25ea59769b3f8f75d8ff55",
      "1750ed02223c6cc0febfb5217650fdf9",
      NA_character_,
      "cb71a6193026a9e0ebae4119a7b52cf1",
      "ad198c7b04ae2f42814a5c2976399308",
      "698d6f01605a80b4850973bb401930c4",
      "64f06f8a4b5ceb83456a3e8fd0c3f2e8",
      "804af8372e384086f6883b07994e50ce",
      "3a132ef91cd5a52e04d0f4115cc9898b"
    )
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
    # If parent check succeeds we're done
    if (super$check_exists()) {
      return(TRUE)
    }

    if (!fs::dir_exists(self$dataset_dir)) {
      return(FALSE)
    }

    # Accept both 'valid' and 'val'
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
      regexp = paste0("[/\\\\]", split_pat, "[/\\\\]_annotations\\.coco\\.json$")
    )

    if (length(hits) >= 1) {
      self$annotation_file <- hits[[1]]
      self$split_dir <- fs::path_dir(self$annotation_file)
      self$image_dir <- self$split_dir
      return(super$check_exists())
    }

    FALSE
  },

  read_image = function(img_path) {
    ext <- tolower(fs::path_ext(img_path))
    x <- if (ext %in% c("jpg","jpeg")) jpeg::readJPEG(img_path)
    else if (ext == "png")        png::readPNG(img_path)
    else                          jpeg::readJPEG(img_path)
    if (length(dim(x)) == 3 && dim(x)[3] == 4) x <- x[,,1:3, drop = FALSE]
    if (length(dim(x)) == 2) x <- array(rep(x, 3L), dim = c(dim(x), 3L))
    x
  }
)
