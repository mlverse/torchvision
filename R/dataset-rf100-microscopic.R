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
    dataset = c(
      "stomata_cell", "blood_cell", "parasite", "cell",
      "liquid_crystals", "bacteria", "cotton_desease",
      "mitosis", "phage", "liver_desease", "asbestos"
    ),
    url = c(
      "https://huggingface.co/datasets/Francesco/stomata-cells/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/bccd-ouzjz/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/parasites-1s07h/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/cells-uyemf/resolve/main/dataset.tar.gz?download=1",
      NA_character_,
      "https://huggingface.co/datasets/Francesco/bacteria-ptywi/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/cotton-plant-disease/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/mitosis-gjs3g/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/phages/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/liver-disease/resolve/main/dataset.tar.gz?download=1",
      "https://huggingface.co/datasets/Francesco/asbestos/resolve/main/dataset.tar.gz?download=1"
    ),
    md5 = NA_character_,
    stringsAsFactors = FALSE
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
