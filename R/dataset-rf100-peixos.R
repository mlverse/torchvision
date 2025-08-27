#' @include dataset-rf100-doc.R
NULL

#' RF100 Peixos Segmentation Dataset
#'
#' Loads the Roboflow 100 "peixos" dataset for semantic segmentation. Each
#' split contains the raw images alongside an `_annotations.coco.json` file in
#' COCO format. Segmentation masks are generated on-the-fly from polygon
#' annotations (falling back to bounding boxes if necessary).
#'
#' @inheritParams rf100_document_collection
#' @return A torch dataset. Each element is a named list with:
#' - `x`: H × W × 3 array (use `transform_to_tensor()` in `transform` to get
#'   C × H × W tensor).
#' - `y`: a list with:
#'     - `masks`: boolean tensor of shape (1, H, W).
#'     - `labels`: integer vector with the class index (always 1 for "fish").
#'
#' The returned item is given class `image_with_segmentation_mask` so it can be
#' visualised with helpers like [draw_segmentation_masks()].
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
#'
#' @family segmentation_dataset
#' @export
rf100_peixos_segmentation_dataset <- torch::dataset(
  name = "rf100_peixos_segmentation_dataset",
  inherit = rf100_document_collection,

  resources = data.frame(
    dataset = "peixos",
    url = "https://huggingface.co/datasets/Francesco/peixos-fish/resolve/main/dataset.tar.gz?download=1",
    md5 = NA_character_,
    stringsAsFactors = FALSE
  ),
  classes = c("fish"),

  initialize = function(
    split = c("train", "test", "valid"),
    root = if (.Platform$OS.type == "windows") fs::path("C:/torchvision-datasets") else fs::path_temp("torchvision-datasets"),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- "peixos"
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    fs::dir_create(self$root, recurse = TRUE)
    self$dataset_dir <- fs::path(self$root, "rf100-peixos")

    resource <- self$resources[self$resources$dataset == self$dataset, , drop = FALSE]
    self$archive_url <- resource$url

    if (download) self$download()

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
  },

  .getitem = function(index) {
    img_path <- self$image_paths[index]
    img_info <- self$images[index, ]
    if (is.na(img_path) || !fs::file_exists(img_path)) {
      runtime_error(paste("Image file not found:", img_info$file_name))
    }
    anns <- self$annotations_by_image[[as.character(img_info$id)]]

    x <- base_loader(img_path)
    if (length(dim(x)) == 3 && dim(x)[3] == 4) {
      x <- x[, , 1:3, drop = FALSE]
    }

    H <- dim(x)[1]
    W <- dim(x)[2]
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

    if (!is.null(self$transform)) x <- self$transform(x)
    if (!is.null(self$target_transform)) y <- self$target_transform(y)

    structure(list(x = x, y = y), class = "image_with_segmentation_mask")
  }
)
