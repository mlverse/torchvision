#' @include dataset-rf100-doc.R
NULL

#' RF100 Electromagnetic Dataset Collection
#'
#' Loads one of the RF100 electromagnetic object detection datasets (COCO
#' format) with per-dataset folders and train/valid/test splits.
#'
#' @inheritParams rf100_document_collection
#' @inherit rf100_document_collection return
#'
#' @examples
#' \dontrun{
#' ds <- rf100_electromagnetic_collection(
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
rf100_electromagnetic_collection <- torch::dataset(
  name = "rf100_electromagnetic_collection",
  inherit = rf100_document_collection,

  resources = data.frame(
    dataset = c(
      "thermal_dog_and_people", "solar_panel", "radio_signal",
      "thermal_cheetah", "rheumatology", "knee",
      "abdomen_mri", "brain_axial_mri", "gynecology_mri",
      "brain_tumor", "fracture", "ir_object"
    ),
    url = c(
      "https://huggingface.co/datasets/Francesco/thermal-dogs-and-people-x6ejw/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/radio-signal/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/thermal-cheetah-my4dp/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/x-ray-rheumatology/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/acl-x-ray/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/abdomen-mri/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/axial-mri/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/gynecology-mri/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/brain-tumor-m2pbp/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/bone-fracture-7fylg/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/flir-camera-objects/resolve/main/dataset.tar.gz?download=1"
    ),
    md5 = c(
      "9c22581d6a33efaa4673524a7a999c31",
      "ee6ee92a6ed67a7a3999feceb032a4bf",
      "a18a3139bcdc63fbb2441e6465869df3",
      "b3a9b673331d6b180d8a3397a4b3a8d2",
      "bc5dbf0118184f3c7fc29fd14f824370",
      "6417140ea24d31cc1796008f97aeec9e",
      "f08ca040a1512bff3dd18e4347c789eb",
      "3b4c34891d23815e3d1b334262ee987b",
      "6a05407e5bf48744d1bee74746898af9",
      "f4ad6d2cf7ed34a8a3d6cc96be06e91f",
      "5956bad4cca0c3bc0ad215d6e3d32638",
      "ebabf3b3c5d447f8a6070d503f9de476"
    ),
    stringsAsFactors = FALSE
  ),

  initialize = function(
    dataset = c(
      "thermal_dog_and_people", "solar_panel", "radio_signal",
      "thermal_cheetah", "rheumatology", "knee",
      "abdomen_mri", "brain_axial_mri", "gynecology_mri",
      "brain_tumor", "fracture", "ir_object"
    ),
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
