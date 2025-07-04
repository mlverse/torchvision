#' Oxford-IIIT Pet Dataset
#'
#' The Oxford-IIIT Pet Dataset is a **segmentation** dataset consisting of color images
#' of 37 pet breeds (cats and dogs). Each image is annotated with a pixel-level
#' trimap segmentation mask, identifying pet, background, and ambiguous regions.
#' It is commonly used for evaluating models on object segmentation tasks.
#'
#' @inheritParams fgvc_aircraft_dataset
#' @param root Character. Root directory where the dataset is stored or will be downloaded to. Files are placed under `root/oxfordiiitpet`.
#' @param target_type Character. One of \code{"category"}, \code{"binary-category"}, or \code{"segmentation"} (default: \code{"category"}).
#'
#' @return A dataset object of class \code{oxfordiiitpet_dataset}, where each item is a named list:
#' \describe{
#'   \item{\code{x}}{a H x W x 3 integer array representing an RGB image.}
#'   \item{\code{y}}{
#'     The target value, depending on \code{target_type}:
#'     \code{"category"} – class index (1–37);
#'     \code{"binary-category"} – 1 for Cat, 2 for Dog;
#'     \code{"segmentation"} – integer mask with values: 1 (pet), 2 (background), 3 (outline).
#'   }
#' }
#'
#' @examples
#' \dontrun{
#' # Load training data with class labels (1 to 37)
#' oxfordiiitpet <- oxfordiiitpet_dataset(train = TRUE, download = TRUE)
#' first_item <- oxfordiiitpet[1]
#' first_item$x  # image tensor
#' first_item$y  # class index
#'
#' # Load with segmentation masks
#' oxfordiiitpet <- oxfordiiitpet_dataset(root = root_dir, target_type = "segmentation")
#' first_item <- oxfordiiitpet[1]
#' first_item$x  # image tensor
#' first_item$y  # segmentation mask tensor (1 = foreground, 2 = background, 3 = outline)
#' }
#'
#' @family segmentation_dataset
#' @export
oxfordiiitpet_segmentation_dataset <- dataset(
  name = "oxfordiiitpet",
  resources = list(
    c("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
    c("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f")
  ),
  training_file = "trainval.rds",
  test_file = "test.rds",
  initialize = function(
    root = tempdir(),
    train = TRUE,
    target_type = "category",
    transform = NULL,
    target_transform = NULL,
    download = FALSE) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$target_type <- target_type
    self$split <- if (train) "train" else "test"

    cli_inform("Oxford-IIIT Pet Dataset (~811MB) will be downloaded and processed if not already available.")

    if (download)
      self$download()

    if (!self$check_exists())
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")

    data_file <- if (train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, data_file))

    self$image_paths <- data$image_paths
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
    self$classes <- names(self$class_to_idx)

    cli_inform("Loaded {length(self$labels)} valid samples for split: '{self$split}'")
  },

  download = function() {

    if (self$check_exists())
      return()

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      url <- r[1]
      p <- download_and_cache(url)
      actual_md5 <- tools::md5sum(p)

      if (actual_md5 != r[2]) {
        cli_abort(glue::glue("Corrupt file! Delete the file in {p} and try again."))
      }

      utils::untar(p, exdir = self$raw_folder)
    }

    for (split in c("trainval", "test")) {
      ann_file <- file.path(self$raw_folder, "annotations", glue::glue("{split}.txt"))
      lines <- readLines(ann_file)
      parts <- strsplit(lines, " ")
      img_ids <- vapply(parts, `[[`, character(1), 1)
      labels <- vapply(parts, function(x) as.integer(x[2]), integer(1))
      bin_labels <- vapply(parts, function(x) as.integer(x[3]), integer(1))
      
      img_paths <- file.path(self$raw_folder, "images", glue::glue("{img_ids}.jpg"))
      seg_paths <- file.path(self$raw_folder, "annotations", "trimaps", glue::glue("{img_ids}.png"))
      
      valid <- file.exists(img_paths) & file.exists(seg_paths)
      
      if (any(!valid)) {
        rlang::warn(glue::glue("Some files are missing in {split} split and will be skipped."))
        for (id in img_ids[!valid]) {
          rlang::warn(glue::glue("Missing files for: {id}"))
        }
      }
      
      image_paths <- img_paths[valid]
      labels <- labels[valid]
      
      raw_classes <- sub("_\\d+$", "", img_ids[valid])
      self$classes <- unique(raw_classes)
      self$classes <- gsub("_", " ", self$classes, fixed = TRUE)
      class_to_idx <- setNames(seq_along(self$classes), self$classes)
      
      saveRDS(
        list(
          image_paths = image_paths,
          labels = labels,
          class_to_idx = class_to_idx
        ),
        file.path(self$processed_folder, glue::glue("{split}.rds"))
      )
    }
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) && fs::file_exists(file.path(self$processed_folder, self$test_file))
  },

  .getitem = function(index) {
    img <- magick::image_read(self$image_paths[index])
    img <- magick::image_data(img, channels = "rgb")
    img <- as.integer(img)
    label <- self$labels[index]

    if (!is.null(self$transform))
      img <- self$transform(img)

    if (self$target_type == "segmentation") {
      seg_name <- basename(self$image_paths[index])
      mask_path <- file.path(self$raw_folder, "annotations", "trimaps", sub("\\.jpg$", ".png", seg_name))
      mask <- magick::image_read(mask_path)
      mask <- magick::image_data(mask, channels = "gray")
      mask <- as.integer(mask)
      label <- mask
    } else if (self$target_type == "binary-category") {
      self$classes <- names(self$class_to_idx)[label]
      if (grepl("^[A-Z]", self$classes)) {
        label <- 1
        self$classes <- "Cat"
      } else {
        label <- 2
        self$classes <- "Dog"
      }
    }

    if (!is.null(self$target_transform))
      label <- self$target_transform(label)

    list(x = img, y = label)
  },

  .length = function() {
    length(self$image_paths)
  },

  active = list(
    raw_folder = function() {
      file.path(self$root_path, "oxfordiiitpet", "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, "oxfordiiitpet", "processed")
    }
  )
)
