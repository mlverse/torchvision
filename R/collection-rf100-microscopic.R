#' @include collection-rf100-doc.R
NULL

#' RF100 Microscopic Dataset Collection
#'
#' Loads one of the RF100 microscopic object detection datasets (COCO format)
#' with per-dataset folders and train/valid/test splits.
#'
#' @inheritParams rf100_document_collection
#' @inherit rf100_document_collection return
#'
#' @examples
#' \dontrun{
#' ds <- rf100_microscopic_collection(
#'   dataset = "stomata_cell",
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
rf100_microscopic_collection <- torch::dataset(
  name = "rf100_microscopic_collection",
  inherit = rf100_document_collection,

  resources = data.frame(
    dataset = rep(c(
      "stomata_cell", "blood_cell", "parasite", "cell",
      "liquid_crystals", "bacteria", "cotton_desease",
      "mitosis", "phage", "liver_desease", "asbestos"
    ),each = 3),
    url = c(
      # stomata_cell
      "https://huggingface.co/datasets/Francesco/stomata-cells/blob/main/data/train-00000-of-00001-f3d24fcd68c928f1.parquet",
      "https://huggingface.co/datasets/Francesco/stomata-cells/blob/main/data/test-00000-of-00001-32bdd663798cbfdc.parquet",
      "https://huggingface.co/datasets/Francesco/stomata-cells/blob/main/data/validation-00000-of-00001-e91f7b8ad813e041.parquet",
      # blood_cell
      "https://huggingface.co/datasets/Francesco/bccd-ouzjz/blob/main/data/train-00000-of-00001-f7bd2ef2fd6e29bf.parquet",
      "https://huggingface.co/datasets/Francesco/bccd-ouzjz/blob/main/data/test-00000-of-00001-8ea65561716eb43e.parquet",
      "https://huggingface.co/datasets/Francesco/bccd-ouzjz/blob/main/data/validation-00000-of-00001-135e61fc2bee97fd.parquet",
      # parasite
      "https://huggingface.co/datasets/Francesco/parasites-1s07h/blob/main/data/train-00000-of-00001-7e2c85bbcc1a45a2.parquet",
      "https://huggingface.co/datasets/Francesco/parasites-1s07h/blob/main/data/test-00000-of-00001-829beee109b299d9.parquet",
      "https://huggingface.co/datasets/Francesco/parasites-1s07h/blob/main/data/validation-00000-of-00001-e3aee1a46235b438.parquet",
      # cell
      "https://huggingface.co/datasets/Francesco/cells-uyemf/blob/main/data/train-00000-of-00001-ddb451e11ab01b6e.parquet",
      "https://huggingface.co/datasets/Francesco/cells-uyemf/blob/main/data/test-00000-of-00001-93af1cfd45e4b7aa.parquet",
      "https://huggingface.co/datasets/Francesco/cells-uyemf/blob/main/data/validation-00000-of-00001-948f71851dc45fa4.parquet",
      # liquid_crystals
      "https://huggingface.co/datasets/Francesco/4-fold-defect/blob/main/data/train-00000-of-00001-b438dc70da70c5f3.parquet",
      "https://huggingface.co/datasets/Francesco/4-fold-defect/blob/main/data/test-00000-of-00001-0c18bd0b95b38167.parquet",
      "https://huggingface.co/datasets/Francesco/4-fold-defect/blob/main/data/validation-00000-of-00001-58a56f521b5aaffb.parquet",
      # bacteria
      "https://huggingface.co/datasets/Francesco/bacteria-ptywi/blob/main/data/train-00000-of-00001-4874c525c9b5291f.parquet",
      "https://huggingface.co/datasets/Francesco/bacteria-ptywi/blob/main/data/test-00000-of-00001-515eb8a6a2a9bb07.parquet",
      "https://huggingface.co/datasets/Francesco/bacteria-ptywi/blob/main/data/validation-00000-of-00001-2d3417d16e44ab71.parquet",
      # cotton_desease
      "https://huggingface.co/datasets/Francesco/cotton-plant-disease/blob/main/data/train-00000-of-00001-fb83220158d0bab1.parquet",
      "https://huggingface.co/datasets/Francesco/cotton-plant-disease/blob/main/data/test-00000-of-00001-cb05eef9488873d3.parquet",
      "https://huggingface.co/datasets/Francesco/cotton-plant-disease/blob/main/data/validation-00000-of-00001-d9b9a7655deebe71.parquet",
      # mitosis
      "https://huggingface.co/datasets/Francesco/mitosis-gjs3g/blob/main/data/train-00000-of-00001-97a883514c06adf1.parquet",
      "https://huggingface.co/datasets/Francesco/mitosis-gjs3g/blob/main/data/test-00000-of-00001-e5d76ced0a07539a.parquet",
      "https://huggingface.co/datasets/Francesco/mitosis-gjs3g/blob/main/data/validation-00000-of-00001-54fb9dda67697025.parquet",
      # phage
      "https://huggingface.co/datasets/Francesco/phages/blob/main/data/train-00000-of-00001-b9eccf55a37a9c14.parquet",
      "https://huggingface.co/datasets/Francesco/phages/blob/main/data/test-00000-of-00001-4fee5dee9a0c8f17.parquet",
      "https://huggingface.co/datasets/Francesco/phages/blob/main/data/validation-00000-of-00001-5d783205edb8aef3.parquet",
      # liver_desease
      "https://huggingface.co/datasets/Francesco/liver-disease/blob/main/data/train-00000-of-00001-075b34404316815c.parquet",
      "https://huggingface.co/datasets/Francesco/liver-disease/blob/main/data/test-00000-of-00001-d6b1dd29852bde4e.parquet",
      "https://huggingface.co/datasets/Francesco/liver-disease/blob/main/data/validation-00000-of-00001-ba8e36e9bd143c60.parquet",
      # asbestos
      "https://huggingface.co/datasets/Francesco/asbestos/blob/main/data/train-00000-of-00001-4bdd9b41e9daaf91.parquet",
      "https://huggingface.co/datasets/Francesco/asbestos/blob/main/data/test-00000-of-00001-283c6964a2943199.parquet",
      "https://huggingface.co/datasets/Francesco/asbestos/blob/main/data/validation-00000-of-00001-6e5e04a76562dda8.parquet"
    ),
    md5 = c(
      # stomata_cell
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # blood_cell
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # parasite
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # cell
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # liquid_crystals
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # bacteria
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # cotton_desease
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # mitosis
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # phage
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # liver_desease
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4",
      # asbestos
      "eac1d66da9b5e92d1e9a21f3c99563e1",      "fe6317e4930f8be1c0c15a600c38e7a4",      "952d5adf6d969844527eb94b63656ce4"
    ),
    size = c(85, 24, 12, 6.7, 1.9, .9 ,68,19,9.5, .3, .1, .05,
             22.5, 5.9, 1.4, 2.5, .8, .8, 65, 17.6, 9,
             19, 5.5, 2.7, 142,72, 9.4, 6,205, 58, 29, 8.2, 4.1) * 1e6
  ),

  initialize = function(
    dataset = c(
      "stomata_cell", "blood_cell", "parasite", "cell",
      "liquid_crystals", "bacteria", "cotton_desease",
      "mitosis", "phage", "liver_desease", "asbestos"
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
