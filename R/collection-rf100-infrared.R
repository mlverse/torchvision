#' @include collection-rf100-doc.R
NULL

#' RoboFlow 100 Infrared dataset Collection
#'
#' Loads one of the [RoboFlow 100 Infrared](https://universe.roboflow.com/browse/infrared) datasets (COCO
#' format) with per-dataset folders and train/valid/test splits.
#'
#' @inheritParams rf100_document_collection
#' @param dataset Dataset to select within \code{c("thermal_dog_and_people", "solar_panel", "thermal_cheetah", "ir_object")}.
#' @inherit rf100_document_collection return
#'
#' @examples
#' \dontrun{
#' ds <- rf100_infrared_collection(
#'   dataset = "thermal_dog_and_people",
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
rf100_infrared_collection <- torch::dataset(
  name = "rf100_infrared_collection",
  inherit = rf100_document_collection,

  resources = data.frame(
    dataset = rep(
      c("thermal_dog_and_people", "solar_panel","thermal_cheetah", "ir_object"), each = 3
    ),
    split   = rep(c("train", "test", "valid"), times = 4),
    url = c(
      # thermal_dog_and_people
      "https://huggingface.co/datasets/Francesco/thermal-dogs-and-people-x6ejw/resolve/main/data/train-00000-of-00001-1610caffe8805fb2.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-dogs-and-people-x6ejw/resolve/main/data/test-00000-of-00001-d8b5e33cc7b3e0f0.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-dogs-and-people-x6ejw/resolve/main/data/validation-00000-of-00001-7c75a093271552c9.parquet",
      # solar_panel
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/resolve/main/data/train-00000-of-00001-70141d825b15d748.parquet",
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/resolve/main/data/test-00000-of-00001-6828d49996604be7.parquet",
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/resolve/main/data/validation-00000-of-00001-11cffae851c69d23.parquet",
      # thermal_cheetah
      "https://huggingface.co/datasets/Francesco/thermal-cheetah-my4dp/resolve/main/data/train-00000-of-00001-f25decaf4cece7b6.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-cheetah-my4dp/resolve/main/data/test-00000-of-00001-9a9c67c1034043ae.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-cheetah-my4dp/resolve/main/data/validation-00000-of-00001-80c7672b224b69ad.parquet",
      # ir_object
      "https://huggingface.co/datasets/Francesco/flir-camera-objects/resolve/main/data/train-00000-of-00001-f9046a472e12c19a.parquet",
      "https://huggingface.co/datasets/Francesco/flir-camera-objects/resolve/main/data/test-00000-of-00001-135a7244612639cc.parquet",
      "https://huggingface.co/datasets/Francesco/flir-camera-objects/resolve/main/data/validation-00000-of-00001-0e03b58b35fd7419.parquet"
    ),
    md5 = c(
      # thermal_dog_and_people
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # solar_panel
      "a6078675c6c4fe50a74ed3fdbe781195",      "ff8983ca30ab4a5eac02ba516d0a5345",      "8f02bfbab9d2c59c1d7a57089c159e56",
      # thermal_cheetah
      "cbbb821abbb4ebf802a5d21e352f68c6",      "2628129c62f797b719d5793af46b0e0f",      "0c5fe2b61ca6d3b0b519a57575182541",
      # ir_object
      "f263921d2671cad90ed13eb0d901df2c",      "29adfd94c41294823ea78754e65116ec",      "4879197eccf7ef0511e2f1c9b7e07d64"
    ),
    size = c(6, 1.7, 0.8,  8.5, 2.3, 1.5,  2.7, 0.7, 0.4,  411, 148, 74) * 1e6,
    stringsAsFactors = FALSE
  )
)
