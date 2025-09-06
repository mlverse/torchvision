#' @include collection-rf100_doc.R
NULL

#' RoboFlow 100 Damages dataset Collection
#'
#' Loads one of the [RoboFlow 100 Damage & Risk assesment](https://universe.roboflow.com/browse/damage-risk) datasets with COCO-style
#' bounding box annotations for object detection tasks.
#'
#' @inheritParams rf100_document_collection
#' @param dataset Dataset to select within \code{c("liquid_crystals", "solar_panel", "asbestos")}.
#' @inherit rf100_document_collection return
#'
#' @examples
#' \dontrun{
#' ds <- rf100_damage_collection(
#'   dataset = "solar_panel",
#'   split = "test",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#' item <- ds[1]
#' boxed <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed)
#' }
#'
#' @family detection_dataset
#' @export
rf100_damage_collection <- torch::dataset(
  name = "rf100_damage_collection",
  inherit = rf100_document_collection,

  resources = data.frame(
    dataset = rep(c("liquid_crystals", "solar_panel", "asbestos"),each = 3),
    split   = rep(c("train", "test", "valid"), times = 3),
    url = c(
      # liquid_crystals
      "https://huggingface.co/datasets/Francesco/4-fold-defect/resolve/main/data/train-00000-of-00001-b438dc70da70c5f3.parquet",
      "https://huggingface.co/datasets/Francesco/4-fold-defect/resolve/main/data/test-00000-of-00001-0c18bd0b95b38167.parquet",
      "https://huggingface.co/datasets/Francesco/4-fold-defect/resolve/main/data/validation-00000-of-00001-58a56f521b5aaffb.parquet",
      # solar_panel
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/resolve/main/data/train-00000-of-00001-70141d825b15d748.parquet",
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/resolve/main/data/test-00000-of-00001-6828d49996604be7.parquet",
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/resolve/main/data/validation-00000-of-00001-11cffae851c69d23.parquet",
      # asbestos
      "https://huggingface.co/datasets/Francesco/asbestos/resolve/main/data/train-00000-of-00001-4bdd9b41e9daaf91.parquet",
      "https://huggingface.co/datasets/Francesco/asbestos/resolve/main/data/test-00000-of-00001-283c6964a2943199.parquet",
      "https://huggingface.co/datasets/Francesco/asbestos/resolve/main/data/validation-00000-of-00001-6e5e04a76562dda8.parquet"
    ),
    md5 = c(
      # liquid_crystals
      "e25e75da04c0f8199a9b47215b6e3ac2",      "b9729b2776cee8704f1ddeff0597378d",      "5cbf6f189b2449969f47dfbb746ecca3",
      # solar_panel
      "a6078675c6c4fe50a74ed3fdbe781195",      "ff8983ca30ab4a5eac02ba516d0a5345",      "8f02bfbab9d2c59c1d7a57089c159e56",
      # asbestos
      "f53af703c4ce594c3950d7e732003f2d",      "8e99904ce49e7f0e830735fb22986868",      "4fef507d057690d1a55fa043696248cc"
    ),
    size = c(21.5, 5.7, 1.4,   8.5, 2.3, 1.5,  28, 7.8, 4) * 1e6
  )
)
