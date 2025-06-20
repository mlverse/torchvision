#' MS COCO Captions Dataset
#'
#' Loads the Microsoft COCO dataset for image captioning.
#'
#' @param root Root directory for the dataset.
#' @param train Whether to load training data (TRUE) or validation data (FALSE).
#' @param year One of "2014" (only available year for captions).
#' @param download If TRUE, downloads the dataset if needed.
#'
#' @export
ms_coco_captions_dataset <- torch::dataset(
  name = "ms_coco_captions_dataset",

  initialize = function(root, train = TRUE, year = c("2014"), download = FALSE) {
    year <- match.arg(year)
    split <- if (train) "train" else "val"

    root <- fs::path_expand(root)
    self$root <- root
    self$year <- year
    self$split <- split
    self$train <- train

    self$image_dir <- fs::path(root, glue::glue("{split}{year}"))
    self$ann_file <- fs::path(root, "annotations", glue::glue("captions_{split}{year}.json"))

    if (download)
      self$download()

    if (!self$check_files())
      rlang::abort("Dataset files not found. Use download = TRUE to fetch them.")

    self$load_annotations()
  },

  download = function() {
    ann_url <- "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    img_url <- if (self$train) {
      "http://images.cocodataset.org/zips/train2014.zip"
    } else {
      "http://images.cocodataset.org/zips/val2014.zip"
    }

    ann_md5 <- "0a379cfc70b0e71301e0f377548639bd"
    img_md5 <- if (self$train) {
      "0da8c0bd3d6c671303cd997d35d63247"
    } else {
      "a3d79f5ed8d289b7a7554ce06a5782b3"
    }

    ann_zip <- torchvision:::download_and_cache(ann_url)
    img_zip <- torchvision:::download_and_cache(img_url)


    utils::unzip(ann_zip, exdir = self$root)
    utils::unzip(img_zip, exdir = self$root)
  },

  check_files = function() {
    fs::file_exists(self$ann_file) && fs::dir_exists(self$image_dir)
  },

  load_annotations = function() {
    annotations <- jsonlite::fromJSON(self$ann_file)
    self$samples <- annotations$annotations
  },

  .getitem = function(index) {
    if (index < 1 || index > length(self)) {
      rlang::abort("Index out of bounds")
    }

    ann <- self$samples[index, ]
    image_id <- ann$image_id
    caption <- ann$caption

    prefix <- if (self$train) "COCO_train2014_" else "COCO_val2014_"
    filename <- paste0(prefix, sprintf("%012d", image_id), ".jpg")
    image_path <- fs::path(self$image_dir, filename)

    image <- jpeg::readJPEG(image_path)

    list(image = image, caption = caption, image_id = image_id)
  },

  .length = function() {
    nrow(self$samples)
  }
)
