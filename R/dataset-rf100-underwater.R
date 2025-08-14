#' RF100 Underwater Dataset Collection
#'
#' Loads one of the RF100 underwater object detection datasets: "pipes",
#' "aquarium", "objects", or "coral". Images are provided with COCO-style
#' bounding box annotations for object detection tasks.
#'
#' @param dataset Character. One of "pipes", "aquarium", "objects", or "coral".
#' @param split Character. One of "train", "test", or "valid".
#' @param root Character. Root directory where the dataset will be stored.
#' @param download Logical. If TRUE, downloads the dataset if not present at
#'   `root`.
#' @param transform Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target
#'   (labels and boxes).
#'
#' @return A torch dataset. Each element is a named list with:
#' - `x`: H x W x 3 array representing the image.
#' - `y`: a list containing:
#'     - `labels`: character vector with object class names.
#'     - `boxes`: a tensor of shape (N, 4) with bounding boxes in
#'       `(xmin, ymin, xmax, ymax)` format.
#'
#' The returned item inherits the class `image_with_bounding_box` so it can be
#' visualised with helper functions such as [draw_bounding_boxes()].
#'
#' @examples
#' \dontrun{
#' # Load dataset
#' devtools::load_all()
#' ds <- rf100_underwater_collection(dataset = "pipes", split = "train", download = TRUE)
#'
#' # Find a sample with annotations
#' item <- ds[1]
#' # Convert array to tensor with proper dimensions (HWC -> CHW)
#' img_tensor <- torch::torch_tensor(item$x)$permute(c(3, 1, 2))
#' tensor_image_browse(img_tensor)
#' item$y$labels
#' item$y$boxes
#' }
#'
#' @family detection_dataset
#' @export
rf100_underwater_collection <- torch::dataset(
  name = "rf100_underwater_collection",
  resources = data.frame(
    dataset = c("pipes", "aquarium", "objects", "coral"),
    url = paste0(
      "https://huggingface.co/datasets/akankshakoshti/rf100-underwater-archives/resolve/main/",
      c(
        "underwater-pipes.zip",
        "aquarium.zip",
        "underwater-objects.zip",
        "coral.zip"
      )
    ),
    md5 = NA_character_
  ),
  initialize = function(
    dataset = c("pipes", "aquarium", "objects", "coral"),
    split = c("train", "test", "valid"),
    root = tempdir(),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- match.arg(dataset)
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    self$dataset_dir <- fs::path(self$root, paste0("rf100_underwater_", self$dataset))
    if (self$dataset %in% c("coral", "aquarium")) {
      self$split_dir <- fs::path(self$dataset_dir, self$dataset, self$split)
    } else {
      self$split_dir <- fs::path(self$dataset_dir, paste0("underwater-", self$dataset), self$split)
    }
    self$image_dir <- self$split_dir
    self$annotation_file <- fs::path(self$split_dir, "_annotations.coco.json")

    resource <- self$resources[self$resources$dataset == self$dataset, ]
    self$archive_url <- resource$url

    if (download) {
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
  },
  download = function() {
    if (self$check_exists())
      return(NULL)

    fs::dir_create(self$dataset_dir, recurse = TRUE)
    archive <- download_and_cache(self$archive_url, prefix = class(self)[1])
    utils::unzip(archive, exdir = self$dataset_dir)
  },
  check_exists = function() {
    fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)
  },
  load_annotations = function() {
    ann <- jsonlite::fromJSON(self$annotation_file)
    self$categories <- ann$categories
    self$images <- ann$images
    self$annotations <- ann$annotations

    if (nrow(self$annotations) > 0) {
      self$annotations_by_image <- split(self$annotations, self$annotations$image_id)
    } else {
      self$annotations_by_image <- list()
    }

    self$image_paths <- fs::path(self$image_dir, self$images$file_name)
  },
  .getitem = function(index) {
    img_path <- self$image_paths[index]
    x <- jpeg::readJPEG(img_path)
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
  },
  .length = function() {
    length(self$image_paths)
  }
)
