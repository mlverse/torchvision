#' @include collection-rf100-doc.R
NULL

#' RF100 Peixos Segmentation Dataset
#'
#' Loads the Roboflow 100 "peixos" dataset for semantic segmentation.
#' "peixos" contains 3 splits of respectively 821 / 118 / 251 color images of size 640 x 640.
#' Segmentation masks are generated on-the-fly from polygon
#' annotations of the unique "fish" category.
#'
#' @inheritParams rf100_document_collection
#' @inheritParams tiny_imagenet_dataset
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
  inherit = coco_detection_dataset,
  archive_size_table = list(train = "119 MB", test = "119 MB", valid = "119 MB"),
  resources = data.frame(
    dataset = "peixos",
    url = "https://huggingface.co/datasets/Francesco/peixos-fish/resolve/main/dataset.tar.gz",
    md5 = "0eb13ea40677178aed2fd47f153fabe2",
    stringsAsFactors = FALSE
  ),
  classes = c("fish"),

  initialize = function(
    split = c("train", "test", "valid"),
    root = tempdir(),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- "peixos"
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    self$data_dir <- fs::path(self$root, class(self)[[1]])
    self$image_dir <- fs::path(self$data_dir, self$split)
    fs::dir_create(self$image_dir, recurse = TRUE)
    self$annotation_file <- fs::path(self$image_dir, "_annotations.coco.json")

    if (download) self$download()

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
  },

  download = function() {
    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    archive <- download_and_cache(self$resources$url, prefix = class(self)[[1]])

    if (tools::md5sum(archive) != self$resources$md5) {
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }
    archive_gz <- fs::path(self$data_dir, basename(archive))
    fs::file_copy(archive, archive_gz, overwrite = TRUE)
    utils::untar(archive_gz, exdir = self$data_dir, extras = "--strip-components=8")

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  load_annotations = function() {
    data <- jsonlite::fromJSON(self$annotation_file)
    # Select relevant cols in each listed dataframe
    ann  <- data$annotations[, c("image_id", "category_id", "segmentation")]
    imgs <- data$images[, c("id", "file_name")]
    cats <- data$categories[, c("id", "name")]

    # Left join annotations with images
    ann_df <- merge(ann, imgs,
                 by.x = "image_id", by.y = "id",
                 all.x = TRUE)

    # shift image index and category index to be 1-indexed
    ann_df$image_id <- ann_df$image_id + 1
    ann_df$category_id <- ann_df$category_id + 1
    ann_df$file_name <- fs::path(self$data_dir, self$split, ann_df$file_name)

    # Filter on existing images and drop _id columns
    has_image <- fs::file_exists(ann_df$file_name)
    self$annotation <- ann_df[has_image, c("image_id", "segmentation", "file_name", "category_id")]
  },

  .length = function() {
    nlevels(as.factor(self$annotation$image_id))
  },

  .getitem = function(index) {
    index_annotation <- self$annotation[self$annotation$image_id == index, ]
    x <- base_loader(index_annotation$file_name[1])

    H <- dim(x)[1]
    W <- dim(x)[2]
    mask <- torch::torch_zeros(c(H, W), dtype = torch::torch_bool())

    mask_lst <- lapply(index_annotation$segmentation, function(seg) {
        if (is.list(seg) && length(seg) > 0) {
          coco_polygon_to_mask(seg, height = H, width = W)
        } else if (is.numeric(seg) && length(seg) > 0) {
          coco_polygon_to_mask(list(seg), height = H, width = W)
        }
      })

    y <- list(masks = torch_stack(mask_lst)$sum(dim = 1, keepdim = TRUE), labels = 1L)

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }
    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    item <- list(x = x, y = y)
    class(item) <- c("image_with_segmentation_mask", class(item))
    item
  }
)
