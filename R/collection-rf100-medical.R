#' @include collection-rf100_doc.R
NULL

#' RoboFlow 100 Medical dataset Collection
#'
#' Loads one of the [RoboFlow 100 Medical](https://universe.roboflow.com/browse/medical) datasets (COCO
#' format) with per-dataset folders and train/valid/test splits.
#'
#' @inheritParams rf100_document_collection
#' @param dataset Dataset to select within \code{c("radio_signal",
#'  "rheumatology", "knee", "abdomen_mri", "brain_axial_mri",
#'  "gynecology_mri", "brain_tumor", "fracture")}.
#' @inherit rf100_document_collection return
#'
#' @examples
#' \dontrun{
#' ds <- rf100_medical_collection(
#'   dataset = "rheumatology",
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
rf100_medical_collection <- torch::dataset(
  name = "rf100_medical_collection",
  inherit = rf100_document_collection,

  resources = data.frame(
    dataset = rep(
      c("radio_signal", "rheumatology", "knee", "abdomen_mri", "brain_axial_mri",
        "gynecology_mri", "brain_tumor", "fracture"), each = 3
    ),
    split   = rep(c("train", "test", "valid"), times = 8),
    url = c(
      # radio_signal
      "https://huggingface.co/datasets/Francesco/radio-signal/resolve/main/data/train-00000-of-00001-0e1bb7466d6c3ca3.parquet",
      "https://huggingface.co/datasets/Francesco/radio-signal/resolve/main/data/test-00000-of-00001-cea246c736448c0f.parquet",
      "https://huggingface.co/datasets/Francesco/radio-signal/resolve/main/data/validation-00000-of-00001-0e172ea211bc5c25.parquet",
      # rheumatology
      "https://huggingface.co/datasets/Francesco/x-ray-rheumatology/resolve/main/data/train-00000-of-00001-3ca059ad007a94ea.parquet",
      "https://huggingface.co/datasets/Francesco/x-ray-rheumatology/resolve/main/data/test-00000-of-00001-8ce1659e856c6713.parquet",
      "https://huggingface.co/datasets/Francesco/x-ray-rheumatology/resolve/main/data/validation-00000-of-00001-0a5d512b2f7219ad.parquet",
      # knee
      "https://huggingface.co/datasets/Francesco/acl-x-ray/resolve/main/data/train-00000-of-00001-297d76f4f8e3f0d1.parquet",
      "https://huggingface.co/datasets/Francesco/acl-x-ray/resolve/main/data/test-00000-of-00001-771fc5699ba6259e.parquet",
      "https://huggingface.co/datasets/Francesco/acl-x-ray/resolve/main/data/validation-00000-of-00001-d64bcf3a8b32ec7d.parquet",
      # abdomen_mri
      "https://huggingface.co/datasets/Francesco/abdomen-mri/resolve/main/data/train-00000-of-00001-b5aa979424bb4685.parquet",
      "https://huggingface.co/datasets/Francesco/abdomen-mri/resolve/main/data/test-00000-of-00001-8b677ef1cabf7f16.parquet",
      "https://huggingface.co/datasets/Francesco/abdomen-mri/resolve/main/data/validation-00000-of-00001-73d9615650a3749b.parquet",
      # brain_axial_mri
      "https://huggingface.co/datasets/Francesco/axial-mri/resolve/main/data/train-00000-of-00001-62cf6bf015fef032.parquet",
      "https://huggingface.co/datasets/Francesco/axial-mri/resolve/main/data/test-00000-of-00001-7780878af8cf3e7b.parquet",
      "https://huggingface.co/datasets/Francesco/axial-mri/resolve/main/data/validation-00000-of-00001-bcd8291312ff472b.parquet",
      # gynecology_mri
      "https://huggingface.co/datasets/Francesco/gynecology-mri/resolve/main/data/train-00000-of-00001-ff598b0e3b7eb2c0.parquet",
      "https://huggingface.co/datasets/Francesco/gynecology-mri/resolve/main/data/test-00000-of-00001-ef538477a1b308ab.parquet",
      "https://huggingface.co/datasets/Francesco/gynecology-mri/resolve/main/data/validation-00000-of-00001-1d8f153a588ec0a9.parquet",
      # brain_tumor
      "https://huggingface.co/datasets/Francesco/brain-tumor-m2pbp/resolve/main/data/train-00000-of-00001-92b37a681420e786.parquet",
      "https://huggingface.co/datasets/Francesco/brain-tumor-m2pbp/resolve/main/data/test-00000-of-00001-bc5b44853d12ccfc.parquet",
      "https://huggingface.co/datasets/Francesco/brain-tumor-m2pbp/resolve/main/data/validation-00000-of-00001-4b3513c4ec0a5e8f.parquet",
      # fracture
      "https://huggingface.co/datasets/Francesco/bone-fracture-7fylg/resolve/main/data/train-00000-of-00001-26e4f0e80e263728.parquet",
      "https://huggingface.co/datasets/Francesco/bone-fracture-7fylg/resolve/main/data/test-00000-of-00001-9fe0a8c08ab79a8b.parquet",
      "https://huggingface.co/datasets/Francesco/bone-fracture-7fylg/resolve/main/data/validation-00000-of-00001-647af68e048aed16.parquet"
    ),
    md5 = c(
      # radio_signal
      "0e92cde0cae78019cdf736a0ec09cb6a",      "696458e584fb090a79790c46d8f0621d",      "08195336ff727c222fcd011d47164ec1",
     # rheumatology
      "3cdb356519def48577f4fbbd075c7328",      "0047b0c762e1434635a4ef78ff979e3d",      "bd6cbaa11f9d0e822881158e5f936e99",
      # knee
      "0e85f93fe793c25b8da3f245cd1968f6",      "7d0da6a924a3aabc21234da9423638a8",      "6afc61c17ad6da97f614ef558afca522",
      # abdomen_mri
      "6790fe6e214d5cf8ffef20bbd9a145bd",      "7b2985dbd628aaa2d6b226bbc13002ee",      "eab70beb5373212ba127c9dc9032214e",
      # brain_axial_mri
      "34d9fa3f0b86b75b98ddb3e111a78222",      "10df9ece0cfd4fd870cd6e77934f66c7",      "5e6a425dede86157a753cb8f62bb302f",
      # gynecology_mri
      "4705c18355707e921780c8c5c66f9233",      "a839a9242eeeed991623f376f83011b7",      "163cf422080c43db4a7d37fe7585d4f6",
      # brain_tumor
      "ac40f245c7c45f0eb9e4491e3452380f",      "0f1b174a26706f35d377dff81293f99b",      "f32abc88b31b06d77479f2aafc2a2062",
      # fracture
      "419080e7f1f400eb97f518c660083a32",      "5a2ee2edede004350478fc8feaa1458f",      "4a6c11900ff3f5b64fac71dba4d1f7f2"),
    size = c(51, 15, 7,  2.5, 0.6, 0.3,  46, 13, 6.5,  62, 15, 9,  3.4, 0.8, 0.5,
             50, 13, 7,  142, 40, 20,  9, 2, 1) * 1e6,
    stringsAsFactors = FALSE
  )
)
