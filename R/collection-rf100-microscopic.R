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
    split   = rep(c("train", "test", "valid"), times = 11),
    url = c(
      # stomata_cell
      "https://huggingface.co/datasets/Francesco/stomata-cells/resolve/main/data/train-00000-of-00001-f3d24fcd68c928f1.parquet",
      "https://huggingface.co/datasets/Francesco/stomata-cells/resolve/main/data/test-00000-of-00001-32bdd663798cbfdc.parquet",
      "https://huggingface.co/datasets/Francesco/stomata-cells/resolve/main/data/validation-00000-of-00001-e91f7b8ad813e041.parquet",
      # blood_cell
      "https://huggingface.co/datasets/Francesco/bccd-ouzjz/resolve/main/data/train-00000-of-00001-f7bd2ef2fd6e29bf.parquet",
      "https://huggingface.co/datasets/Francesco/bccd-ouzjz/resolve/main/data/test-00000-of-00001-8ea65561716eb43e.parquet",
      "https://huggingface.co/datasets/Francesco/bccd-ouzjz/resolve/main/data/validation-00000-of-00001-135e61fc2bee97fd.parquet",
      # parasite
      "https://huggingface.co/datasets/Francesco/parasites-1s07h/resolve/main/data/train-00000-of-00001-7e2c85bbcc1a45a2.parquet",
      "https://huggingface.co/datasets/Francesco/parasites-1s07h/resolve/main/data/test-00000-of-00001-829beee109b299d9.parquet",
      "https://huggingface.co/datasets/Francesco/parasites-1s07h/resolve/main/data/validation-00000-of-00001-e3aee1a46235b438.parquet",
      # cell
      "https://huggingface.co/datasets/Francesco/cells-uyemf/resolve/main/data/train-00000-of-00001-ddb451e11ab01b6e.parquet",
      "https://huggingface.co/datasets/Francesco/cells-uyemf/resolve/main/data/test-00000-of-00001-93af1cfd45e4b7aa.parquet",
      "https://huggingface.co/datasets/Francesco/cells-uyemf/resolve/main/data/validation-00000-of-00001-948f71851dc45fa4.parquet",
      # liquid_crystals
      "https://huggingface.co/datasets/Francesco/4-fold-defect/resolve/main/data/train-00000-of-00001-b438dc70da70c5f3.parquet",
      "https://huggingface.co/datasets/Francesco/4-fold-defect/resolve/main/data/test-00000-of-00001-0c18bd0b95b38167.parquet",
      "https://huggingface.co/datasets/Francesco/4-fold-defect/resolve/main/data/validation-00000-of-00001-58a56f521b5aaffb.parquet",
      # bacteria
      "https://huggingface.co/datasets/Francesco/bacteria-ptywi/resolve/main/data/train-00000-of-00001-4874c525c9b5291f.parquet",
      "https://huggingface.co/datasets/Francesco/bacteria-ptywi/resolve/main/data/test-00000-of-00001-515eb8a6a2a9bb07.parquet",
      "https://huggingface.co/datasets/Francesco/bacteria-ptywi/resolve/main/data/validation-00000-of-00001-2d3417d16e44ab71.parquet",
      # cotton_desease
      "https://huggingface.co/datasets/Francesco/cotton-plant-disease/resolve/main/data/train-00000-of-00001-fb83220158d0bab1.parquet",
      "https://huggingface.co/datasets/Francesco/cotton-plant-disease/resolve/main/data/test-00000-of-00001-cb05eef9488873d3.parquet",
      "https://huggingface.co/datasets/Francesco/cotton-plant-disease/resolve/main/data/validation-00000-of-00001-d9b9a7655deebe71.parquet",
      # mitosis
      "https://huggingface.co/datasets/Francesco/mitosis-gjs3g/resolve/main/data/train-00000-of-00001-97a883514c06adf1.parquet",
      "https://huggingface.co/datasets/Francesco/mitosis-gjs3g/resolve/main/data/test-00000-of-00001-e5d76ced0a07539a.parquet",
      "https://huggingface.co/datasets/Francesco/mitosis-gjs3g/resolve/main/data/validation-00000-of-00001-54fb9dda67697025.parquet",
      # phage
      "https://huggingface.co/datasets/Francesco/phages/resolve/main/data/train-00000-of-00001-b9eccf55a37a9c14.parquet",
      "https://huggingface.co/datasets/Francesco/phages/resolve/main/data/test-00000-of-00001-4fee5dee9a0c8f17.parquet",
      "https://huggingface.co/datasets/Francesco/phages/resolve/main/data/validation-00000-of-00001-5d783205edb8aef3.parquet",
      # liver_desease
      "https://huggingface.co/datasets/Francesco/liver-disease/resolve/main/data/train-00000-of-00001-075b34404316815c.parquet",
      "https://huggingface.co/datasets/Francesco/liver-disease/resolve/main/data/test-00000-of-00001-d6b1dd29852bde4e.parquet",
      "https://huggingface.co/datasets/Francesco/liver-disease/resolve/main/data/validation-00000-of-00001-ba8e36e9bd143c60.parquet",
      # asbestos
      "https://huggingface.co/datasets/Francesco/asbestos/resolve/main/data/train-00000-of-00001-4bdd9b41e9daaf91.parquet",
      "https://huggingface.co/datasets/Francesco/asbestos/resolve/main/data/test-00000-of-00001-283c6964a2943199.parquet",
      "https://huggingface.co/datasets/Francesco/asbestos/resolve/main/data/validation-00000-of-00001-6e5e04a76562dda8.parquet"
    ),
    md5 = c(
      # stomata_cell
      "ba27c9cee8476a814a2869a622b9c2e3",      "95e3713caf397aa4182c2ed72651032a",      "c7c50c24fb1c11abc8da96eda21d8055",
      # blood_cell
      "40a57c143f70aa7b9898011669d5c500",      "601b0b184be5ff7b134edae5619ba852",      "4429c295861719418d038819241b40b9",
      # parasite
      "83d03d6eeee66f9b0bb32e59b9168a2f",      "c25500e4e96d60d5f6ae9ec4c40cb479",      "d74f6367de2cf8c9ef7c1d2199881362",
      # cell
      "6b23b8be679a0eff434147adec24c30a",      "24340cf1332e32241ca453f7277e4b43",      "2d9d19a6f8280bc72a7377f94a0d745d",
      # liquid_crystals
      "e25e75da04c0f8199a9b47215b6e3ac2",      "b9729b2776cee8704f1ddeff0597378d",      "5cbf6f189b2449969f47dfbb746ecca3",
      # bacteria
      "c6641ee845c254e398d746c4286ae9af",      "7468dc6ae120a82d2f750a0bd051d338",      "ed9d0b920cb96cfa707b8f648675c07f",
      # cotton_desease
      "475743a07c9ffb2ad106c112f83fd110",      "8773a3dba89c3bf86879d12c1bb93007",      "45082d626c19fd3e5c3d3d955ae83f7b",
      # mitosis
      "6d7a45051cbb8bda5203004e04bb6639",      "64c4e2470e47556d3736dd2279f6457e",      "9f915191ff92498d1c9250bd1994afdf",
      # phage
      "c7622fed8a37b697b9dd1c3ab0d0708f",      "4623461f1526e50c6badae08e28b1692",      "9610c62631166e65636d25d541b6911b",
      # liver_desease
      "9b5839ae524277eb1702a0db33030e9a",      "c97b80332d710b1378a9f3ceb5abc197",      "2f255ba7ed3c1ce0948d2fc06305ce54",
      # asbestos
      "f53af703c4ce594c3950d7e732003f2d",      "8e99904ce49e7f0e830735fb22986868",      "4fef507d057690d1a55fa043696248cc"
    ),
    size = c(81, 24, 12, 6.4, 1.8, .9 ,65.1,17.9,9, .3, .1, .05,
             21.5, 5.7, 1.4, 2.5, .8, .8, 62, 16.8, 9,
             19, 5.3, 2.7,  69,9.0, 5.7,   192,55.6, 28, 28, 7.8, 4) * 1e6
  )
)
