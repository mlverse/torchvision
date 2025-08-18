#' RF100 Peixos Segmentation Dataset
#'
#' Loads the Roboflow 100 "peixos" dataset for semantic segmentation. Each
#' split contains the raw images alongside an `_annotations.coco.json` file in
#' COCO format. Segmentation masks are generated on-the-fly from the polygon
#' annotations contained in this file.
#'
#' @param split Character. One of "train", "test", or "valid".
#' @param root Character. Root directory where the dataset will be stored.
#' @param download Logical. If TRUE, downloads the dataset if not present at
#'   `root`.
#' @param transform Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target
#'   (mask and labels).
#'
#' @return A torch dataset. Each element is a named list with:
#' - `x`: H x W x 3 array representing the image.
#' - `y`: a list containing:
#'     - `masks`: a boolean tensor of shape (1, H, W) with the segmentation mask.
#'     - `labels`: integer vector with the class index (always 1 for "fish").
#'
#' The returned item inherits the class `image_with_segmentation_mask` so it can
#' be visualised with helper functions such as [draw_segmentation_masks()].
#'
#' @examples
#' \dontrun{
#' ds <- rf100_peixos_segmentation_dataset(
#'   split = "train",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#' item <- ds[1]
#' overlay <- draw_segmentation_masks(item)
#' tensor_image_browse(overlay)
#' }
#' @family segmentation_dataset
#' @export
rf100_peixos_segmentation_dataset <- torch::dataset(
  name = "rf100_peixos_segmentation_dataset",
  resources = data.frame(
    url = "https://huggingface.co/datasets/akankshakoshti/rf100-peixos/resolve/main/peixos-fish.zip?download=1",
    md5 = NA_character_
  ),
  classes = c("fish"),
  initialize = function(
    split = c("train", "test", "valid"),
    root = tempdir(),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    self$dataset_dir <- fs::path(self$root, "rf100-peixos")
    self$split_dir <- fs::path(self$dataset_dir, self$split)
    self$annotation_file <- fs::path(self$split_dir, "_annotations.coco.json")
    self$archive_url <- self$resources$url[1]

    if (download) {
      self$download()
    }

    if (!self$check_exists()) {
      msg <- paste0(
        "Dataset not found. Expected annotations in '", self$annotation_file, "'."
      )
      if (fs::dir_exists(self$dataset_dir)) {
        sub_dirs <- fs::dir_ls(self$dataset_dir, type = "directory", recurse = 3)
        msg <- paste0(
          msg,
          "\nAvailable directories under '", self$dataset_dir, "':\n",
          paste(sub_dirs, collapse = "\n")
        )
      }
      runtime_error(msg)
    }

    self$load_annotations()

    self$image_paths <- fs::path(self$split_dir, self$images$file_name)
  },
  download = function() {
    if (self$check_exists())
      return(NULL)

    fs::dir_create(self$dataset_dir, recurse = TRUE)
    archive <- download_and_cache(self$archive_url, prefix = class(self)[1])
    utils::unzip(archive, exdir = self$dataset_dir)
  },
  check_exists = function() {
    if (fs::file_exists(self$annotation_file)) {
      return(TRUE)
    }

    if (!fs::dir_exists(self$dataset_dir)) {
      return(FALSE)
    }

    split_candidates <- c(self$split, if (self$split == "valid") "val")
    dirs <- fs::dir_ls(self$dataset_dir, recurse = TRUE, type = "directory")
    pat <- paste0("[/\\\\](", paste(split_candidates, collapse = "|"), ")$")
    candidates <- dirs[grepl(pat, tolower(dirs))]
    for (cand in candidates) {
      anno <- fs::path(cand, "_annotations.coco.json")
      if (fs::file_exists(anno)) {
        self$split_dir <- cand
        self$annotation_file <- anno
        return(TRUE)
      }
    }

    FALSE
  },
  load_annotations = function() {
    ann <- jsonlite::fromJSON(self$annotation_file, simplifyVector = TRUE)
    self$images <- ann$images
    self$annotations <- ann$annotations

    self$image_ids <- self$images$id
    self$ann_by_image <- split(self$annotations, self$annotations$image_id)
  },
  .getitem = function(index) {
    image_id <- self$image_ids[index]
    info <- self$images[self$images$id == image_id, ]
    img_path <- fs::path(self$split_dir, info$file_name)

    ext <- tolower(fs::path_ext(img_path))
    x <- if (ext %in% c("jpg", "jpeg")) jpeg::readJPEG(img_path)
    else if (ext == "png") png::readPNG(img_path)
    else jpeg::readJPEG(img_path)
    if (length(dim(x)) == 2) x <- array(rep(x, 3L), dim = c(dim(x), 3L))
    if (length(dim(x)) == 3 && dim(x)[3] == 4) x <- x[,,1:3, drop = FALSE]

    H <- dim(x)[1]
    W <- dim(x)[2]

    anns <- self$ann_by_image[[as.character(image_id)]]
    mask <- torch::torch_zeros(c(H, W), dtype = torch::torch_bool())

    if (!is.null(anns) && nrow(anns) > 0) {
      masks <- lapply(seq_len(nrow(anns)), function(i) {
        seg <- anns$segmentation[[i]]

        if (is.list(seg) && length(seg) > 0) {
          coco_polygon_to_mask(seg, height = H, width = W)
        } else if (is.numeric(seg) && length(seg) > 0) {
          coco_polygon_to_mask(list(seg), height = H, width = W)
        } else {
          NULL
        }
      })

      masks <- Filter(Negate(is.null), masks)
      if (length(masks) > 0) {
        for (m in masks) {
          mask <- mask | m
        }
      }
    }

    m <- mask$unsqueeze(1)

    y <- list(masks = m, labels = 1L)
    if (!is.null(self$transform))
      x <- self$transform(x)
    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    structure(list(x = x, y = y), class = "image_with_segmentation_mask")
  },
  .length = function() {
    length(self$image_ids)
  }
)
