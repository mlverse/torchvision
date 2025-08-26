#' RF100 Underwater Dataset Collection
#'
#' Loads one of the RF100 underwater object detection datasets: "pipes",
#' "aquarium", "objects", or "coral". Images are provided with COCO-style
#' bounding box annotations for object detection tasks.
#'
#' @inheritParams rf100_document_collection
#' @inherit rf100_document_collection return
#'
#' @examples
#' \dontrun{
#' ds <- rf100_underwater_collection(
#'   dataset = "objects",
#'   split = "train",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#'
#' item <- ds[1]
#' boxed <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed)
#' }
#'
#' @family detection_dataset
#' @export
rf100_underwater_collection <- torch::dataset(
  name = "rf100_underwater_collection",
  inherit = rf100_document_collection,

  resources = data.frame(
    dataset = c("pipes", "aquarium", "objects", "coral"),
    url = c(
      "https://huggingface.co/datasets/Francesco/underwater-pipes-4ng4t/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/aquarium-qlnqy/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/underwater-objects-5v7p8/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/coral-lwptl/resolve/main/dataset.tar.gz?download=1"
    ),
    md5 = NA_character_,
    stringsAsFactors = FALSE
  ),

  initialize = function(
    dataset = c("pipes", "aquarium", "objects", "coral"),
    split = c("train", "test", "valid"),
    root = if (.Platform$OS.type == "windows") fs::path("C:/torchvision-datasets") else fs::path_temp("torchvision-datasets"),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    super$initialize(
      dataset = dataset,
      split = split,
      root = root,
      download = download,
      transform = transform,
      target_transform = target_transform
    )
  }
)
