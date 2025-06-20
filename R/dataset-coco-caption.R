#' COCO Caption Dataset
#'
#' Loads the Microsoft COCO dataset for image captioning.
#'
#' @param root Root directory for the dataset.
#' @param train Logical. Whether to load the training split (`TRUE`) or validation split (`FALSE`). Default is `TRUE`.
#' @param year Year of the dataset. Only `"2014"` is currently supported.
#' @param download If `TRUE`, downloads the dataset if it's not already present.
#'
#' @return
#' A `coco_caption_dataset` object. Each item is a list with the following elements:
#' \describe{
#'   \item{image}{A 3D array of shape `(H, W, C)` in RGB format.}
#'   \item{caption}{A character string containing the caption for the image.}
#'   \item{image_id}{An integer ID associated with the image.}
#' }
#'
#' @details
#' This dataset inherits from `coco_detection_dataset` and reuses most of its logic,
#' except it loads image-caption pairs instead of bounding boxes and object labels.
#'
#' @examples
#' \dontrun{
#' library(torch)
#' library(torchvision)
#'
#' ds <- coco_caption_dataset(
#'   root = "~/data",
#'   train = FALSE,
#'   year = "2014",
#'   download = TRUE
#' )
#'
#' item <- ds[1]
#' image <- item$image
#' caption <- item$caption
#'
#' image_array <- as.numeric(image)
#' dim(image_array) <- dim(image)
#'
#' plot(as.raster(image_array))
#' title(main = caption, col.main = "black", font.main = 1)
#' }
#'
#' @importFrom torchvision coco_detection_dataset
#' @export
coco_caption_dataset  <- torch::dataset(
  name = "coco_caption_dataset",
  inherit = coco_detection_dataset,

  initialize = function(root, train = TRUE, year = c("2014"), download = FALSE) {
    year <- match.arg(year)
    split <- if (train) "train" else "val"

    root <- fs::path_expand(root)
    self$root <- root
    self$split <- split
    self$year <- year
    self$data_dir <- fs::path(root, glue::glue("coco{year}"))
    self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{year}"))
    self$ann_file <- fs::path(self$data_dir, "annotations", glue::glue("captions_{split}{year}.json"))

    if (download)
      self$download()

    if (!self$check_files())
      rlang::abort("Dataset files not found. Use download = TRUE to fetch them.")

    self$load_annotations()
  },

  check_files = function() {
    fs::file_exists(self$ann_file) && fs::dir_exists(self$image_dir)
  },

  load_annotations = function() {
    annotations <- jsonlite::fromJSON(self$ann_file)
    self$samples <- annotations$annotations
  },

  .getitem = function(index) {
    if (index < 1 || index > length(self))
      rlang::abort("Index out of bounds")

    ann <- self$samples[index, ]
    image_id <- ann$image_id
    caption <- ann$caption

    prefix <- if (self$split == "train") "COCO_train2014_" else "COCO_val2014_"
    filename <- paste0(prefix, sprintf("%012d", image_id), ".jpg")
    image_path <- fs::path(self$image_dir, filename)

    image <- jpeg::readJPEG(image_path)

    list(image = image, caption = caption, image_id = image_id)
  },

  .length = function() {
    nrow(self$samples)
  }
)
