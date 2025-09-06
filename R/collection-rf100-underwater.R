#' @include collection-rf100_doc.R
NULL

#' RoboFlow 100 Underwater dataset Collection
#'
#' Loads one of the [RoboFlow 100 Underwater](https://universe.roboflow.com/browse/documents) datasets: "pipes",
#' "aquarium", "objects", or "coral". Images are provided with COCO-style
#' bounding box annotations for object detection tasks.
#'
#' @inheritParams rf100_document_collection
#' @param dataset Dataset to select within \code{c("pipes", "aquarium", "objects", "coral")}.
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
    dataset = rep(c("pipes", "aquarium", "objects", "coral"),each = 3),
    split   = rep(c("train", "test", "valid"), times = 4),
    url = c(
      # pipes
      "https://huggingface.co/datasets/Francesco/underwater-pipes-4ng4t/resolve/main/data/train-00000-of-00001-229f86f63d040d1c.parquet",
      "https://huggingface.co/datasets/Francesco/underwater-pipes-4ng4t/resolve/main/data/test-00000-of-00001-6f4640f0fece4cfe.parquet",
      "https://huggingface.co/datasets/Francesco/underwater-pipes-4ng4t/resolve/main/data/validation-00000-of-00001-be63221743f11a18.parquet",
      # aquarium
      "https://huggingface.co/datasets/Francesco/aquarium-qlnqy/resolve/main/data/train-00000-of-00001-f7b9135b8ae646d4.parquet",
      "https://huggingface.co/datasets/Francesco/aquarium-qlnqy/resolve/main/data/test-00000-of-00001-a436c40b7f9121c8.parquet",
      "https://huggingface.co/datasets/Francesco/aquarium-qlnqy/resolve/main/data/validation-00000-of-00001-8a7f5f5ee0144783.parquet",
      # objects
      "https://huggingface.co/datasets/Francesco/underwater-objects-5v7p8/resolve/main/data/train-00000-of-00001-2c9335408bd5503d.parquet",
      "https://huggingface.co/datasets/Francesco/underwater-objects-5v7p8/resolve/main/data/test-00000-of-00001-7670918d21b7b1b6.parquet",
      "https://huggingface.co/datasets/Francesco/underwater-objects-5v7p8/resolve/main/data/validation-00000-of-00001-51a32a9b21a2097c.parquet",
      # coral
      "https://huggingface.co/datasets/Francesco/coral-lwptl/resolve/main/data/train-00000-of-00001-05c58869565f0bfb.parquet",
      "https://huggingface.co/datasets/Francesco/coral-lwptl/resolve/main/data/test-00000-of-00001-6d38ab2deab119d4.parquet",
      "https://huggingface.co/datasets/Francesco/coral-lwptl/resolve/main/data/validation-00000-of-00001-52a8f5261f53bb7c.parquet"
    ),
    md5 = c(
      # pipes
      "521c3781e2e078d6ff2a1d47800bade1",      "8ed57e5b2c81e4cf881f0d6ac1e8ce78",      "3ed01d2935747e8f4f36bb6f9c77da68",
      # aquarium
      "f271749dff54a654c6c337f255744b50",      "961b0588671b682bad4a55b18bf0264f",      "bddf0ef97f146fb7f2411e833caa713c",
      # objects
      "2c97febd495f514d96b8acd2ac293b90",      "95e3713caf397aa4182c2ed72651032a",      "c7c50c24fb1c11abc8da96eda21d8055",
      # coral
      "97bbf0dfdf902952619940493d29f9de",      "09ce58473adf04f1c38071ea86fbfab7",      "3097061e42de960c8746bcf188c5eeb0"
    ),
    size = c(223, 60, 30, 26, 7.5, 3.5 ,290,85,42, 33,7,5) *1e6,
    stringsAsFactors = FALSE
  )
)
