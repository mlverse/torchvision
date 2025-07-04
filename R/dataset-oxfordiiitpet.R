#' Oxford-IIIT Pet Dataset
#'
#' The Oxford-IIIT Pet Dataset is a **segmentation** dataset consisting of color images
#' of 37 pet breeds (cats and dogs). Each image is annotated with a pixel-level
#' trimap segmentation mask, identifying pet, background, and outline regions.
#' It is commonly used for evaluating models on object segmentation tasks.
#'
#' @inheritParams fgvc_aircraft_dataset
#' @param root Character. Root directory where the dataset is stored or will be downloaded to. Files are placed under `root/oxfordiiitpet`.
#' @param target_type Character. One of \code{"category"} or \code{"binary-category"} (default: \code{"category"}).
#'
#' @return A torch dataset object \code{oxfordiiitpet_dataset}. Each item is a named list:
#' - \code{x}: a H x W x 3 integer array representing an RGB image.
#' - \code{y$mask}: an integer array with the same height and width as \code{x}, representing
#'   the segmentation trimap.
#' - \code{y$label}: an integer representing the class label, depending on the \code{target_type}:
#'   - \code{"category"}: an integer in 1–37 indicating the pet breed.
#'   - \code{"binary-category"}: 1 for Cat, 2 for Dog.
#'
#' @examples
#' \dontrun{
#' # Load the Oxford-IIIT Pet segmentation dataset with image and mask transforms
#' oxfordiiitpet <- oxfordiiitpet_segmentation_dataset(
#'   transform = function(x) {
#'     x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
#'   },
#'   target_transform = function(y) {
#'     y$mask <- y$mask %>% transform_to_tensor() %>% transform_resize(c(224, 224))
#'     y
#'   }
#' )
#'
#' # Create a dataloader and fetch one batch
#' dl <- dataloader(oxfordiiitpet, batch_size = 4)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#'
#' # Access batch data
#' batch$x             # Tensor of shape (4, 3, 224, 224)
#' batch$y$mask        # Tensor of shape (4, 1, 224, 224)
#' batch$y$label       # Tensor of shape (4,) with class labels
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
  archive_size = "800 MB",
  initialize = function(
    root = tempdir(),
    train = TRUE,
    target_type = "category",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$target_type <- target_type
    self$split <- if (train) "train" else "test"

    if (download){
      cli_inform("{.cls {class(self)[[1]]}} Dataset (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    data_file <- if (train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, data_file))
    self$image_paths <- data$image_paths
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
    self$classes <- if (self$target_type == "category") names(self$class_to_idx) else c("Cat", "Dog")

    if (!self$check_exists())
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$image_paths)} images across {length(self$classes)} classes.")
  },

  download = function() {

    if (self$check_exists())
      return()

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    for (r in self$resources) {
      url <- r[1]
      archive <- download_and_cache(url)
      actual_md5 <- tools::md5sum(archive)

      if (actual_md5 != r[2]) {
        cli_abort("Corrupt file! Delete the file in {.file {archive}} and try again.")
      }

      utils::untar(archive, exdir = self$raw_folder)
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
        cli_warn("Some files are missing in {split} split and will be skipped.")
        for (id in img_ids[!valid]) {
          cli_warn("Missing files for: {id}")
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

    cli_inform("{.cls {class(self)[[1]]}} dataset downloaded and extracted successfully.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) && fs::file_exists(file.path(self$processed_folder, self$test_file))
  },

  .getitem = function(index) {
    x <- jpeg::readJPEG(self$image_paths[index])

    label <- self$labels[index]

    seg_name <- basename(self$image_paths[index])
    mask_path <- file.path(self$raw_folder, "annotations", "trimaps", sub("\\.jpg$", ".png", seg_name))
    mask <- png::readPNG(mask_path)

    if (self$target_type == "binary-category") {
      self$classes <- names(self$class_to_idx)[label]
      if (substr(self$classes, 1, 1) %in% LETTERS) {
        label <- as.integer(1)
      } else {
        label <- as.integer(2)
      }
      self$classes <- c("Cat","Dog")
    }

    y <- list(
      mask = mask,
      label = label
    )

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
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
