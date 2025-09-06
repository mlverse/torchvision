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
    dataset = rep(c(
      "thermal_dog_and_people", "solar_panel", "radio_signal",
      "thermal_cheetah", "rheumatology", "knee",
      "abdomen_mri", "brain_axial_mri", "gynecology_mri",
      "brain_tumor", "fracture", "ir_object") each = 3),
    split   = rep(c("train", "test", "valid"), times = 6),
    url = c(
      # thermal_dog_and_people
      "https://huggingface.co/datasets/Francesco/thermal-dogs-and-people-x6ejw/blob/main/data/train-00000-of-00001-1610caffe8805fb2.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-dogs-and-people-x6ejw/blob/main/data/test-00000-of-00001-d8b5e33cc7b3e0f0.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-dogs-and-people-x6ejw/blob/main/data/validation-00000-of-00001-7c75a093271552c9.parquet",
      # solar_panel
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/blob/main/data/train-00000-of-00001-70141d825b15d748.parquet",
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/blob/main/data/test-00000-of-00001-6828d49996604be7.parquet",
      "https://huggingface.co/datasets/Francesco/solar-panels-taxvb/blob/main/data/validation-00000-of-00001-11cffae851c69d23.parquet",
      # radio_signal
      "https://huggingface.co/datasets/Francesco/radio-signal/blob/main/data/train-00000-of-00001-0e1bb7466d6c3ca3.parquet",
      "https://huggingface.co/datasets/Francesco/radio-signal/blob/main/data/test-00000-of-00001-cea246c736448c0f.parquet",
      "https://huggingface.co/datasets/Francesco/radio-signal/blob/main/data/validation-00000-of-00001-0e172ea211bc5c25.parquet",
      # thermal_cheetah
      "https://huggingface.co/datasets/Francesco/thermal-cheetah-my4dp/blob/main/data/train-00000-of-00001-f25decaf4cece7b6.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-cheetah-my4dp/blob/main/data/test-00000-of-00001-9a9c67c1034043ae.parquet",
      "https://huggingface.co/datasets/Francesco/thermal-cheetah-my4dp/blob/main/data/validation-00000-of-00001-80c7672b224b69ad.parquet",
      # rheumatology
      "https://huggingface.co/datasets/Francesco/x-ray-rheumatology/blob/main/data/train-00000-of-00001-3ca059ad007a94ea.parquet",
      "https://huggingface.co/datasets/Francesco/x-ray-rheumatology/blob/main/data/test-00000-of-00001-8ce1659e856c6713.parquet",
      "https://huggingface.co/datasets/Francesco/x-ray-rheumatology/blob/main/data/validation-00000-of-00001-0a5d512b2f7219ad.parquet",
      # knee
      "https://huggingface.co/datasets/Francesco/acl-x-ray/blob/main/data/train-00000-of-00001-297d76f4f8e3f0d1.parquet",
      "https://huggingface.co/datasets/Francesco/acl-x-ray/blob/main/data/test-00000-of-00001-771fc5699ba6259e.parquet",
      "https://huggingface.co/datasets/Francesco/acl-x-ray/blob/main/data/validation-00000-of-00001-d64bcf3a8b32ec7d.parquet",
      # abdomen_mri
      "https://huggingface.co/datasets/Francesco/abdomen-mri/blob/main/data/train-00000-of-00001-b5aa979424bb4685.parquet",
      "https://huggingface.co/datasets/Francesco/abdomen-mri/blob/main/data/test-00000-of-00001-8b677ef1cabf7f16.parquet",
      "https://huggingface.co/datasets/Francesco/abdomen-mri/blob/main/data/validation-00000-of-00001-73d9615650a3749b.parquet",
      # brain_axial_mri
      "https://huggingface.co/datasets/Francesco/axial-mri/blob/main/data/train-00000-of-00001-62cf6bf015fef032.parquet",
      "https://huggingface.co/datasets/Francesco/axial-mri/blob/main/data/test-00000-of-00001-7780878af8cf3e7b.parquet",
      "https://huggingface.co/datasets/Francesco/axial-mri/blob/main/data/validation-00000-of-00001-bcd8291312ff472b.parquet",
      # gynecology_mri
      "https://huggingface.co/datasets/Francesco/gynecology-mri/blob/main/data/train-00000-of-00001-ff598b0e3b7eb2c0.parquet",
      "https://huggingface.co/datasets/Francesco/gynecology-mri/blob/main/data/test-00000-of-00001-ef538477a1b308ab.parquet",
      "https://huggingface.co/datasets/Francesco/gynecology-mri/blob/main/data/validation-00000-of-00001-1d8f153a588ec0a9.parquet",
      # brain_tumor
      "https://huggingface.co/datasets/Francesco/brain-tumor-m2pbp/blob/main/data/train-00000-of-00001-92b37a681420e786.parquet",
      "https://huggingface.co/datasets/Francesco/brain-tumor-m2pbp/blob/main/data/test-00000-of-00001-bc5b44853d12ccfc.parquet",
      "https://huggingface.co/datasets/Francesco/brain-tumor-m2pbp/blob/main/data/validation-00000-of-00001-4b3513c4ec0a5e8f.parquet",
      # fracture
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
