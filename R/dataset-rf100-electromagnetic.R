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
    md5 = NA_character_,
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
