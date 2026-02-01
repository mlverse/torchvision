#' Oxford-IIIT Pet Segmentation Dataset
#'
#' The Oxford-IIIT Pet Dataset is a **segmentation** dataset consisting of color images
#' of 37 pet breeds (cats and dogs). Each image is annotated with a pixel-level
#' trimap segmentation mask, identifying pet, background, and outline regions.
#' It is commonly used for evaluating models on object segmentation tasks.
#'
#' @inheritParams mnist_dataset
#' @param root Character. Root directory where the dataset is stored or will be downloaded to. Files are placed under `root/oxfordiiitpet`.
#' @param target_type Character. One of \code{"category"} or \code{"binary-category"} (default: \code{"category"}).
#'
#' @return A torch dataset object \code{oxfordiiitpet_dataset}. Each item is a named list:
#' - \code{x}: a H x W x 3 integer array representing an RGB image.
#' - \code{y$masks}: a boolean tensor of shape (3, H, W), representing the segmentation trimap as one-hot masks.
#' - \code{y$label}: an integer representing the class label, depending on the \code{target_type}:
#'   - \code{"category"}: an integer in 1–37 indicating the pet breed.
#'   - \code{"binary-category"}: 1 for Cat, 2 for Dog.
#'
#' @examples
#' \dontrun{
#' # Load the Oxford-IIIT Pet dataset with basic tensor transform
#' oxfordiiitpet <- oxfordiiitpet_segmentation_dataset(
#'    transform = transform_to_tensor,
#'    download = TRUE
#' )
#'
#' # Retrieve the image tensor, segmentation mask and label
#' first_item <- oxfordiiitpet[1]
#' first_item$x  # RGB image tensor of shape (3, H, W)
#' first_item$y$masks   # (3, H, W) bool tensor: pet, background, outline
#' first_item$y$label  # Integer label (1–37 or 1–2 depending on target_type)
#' oxfordiiitpet$classes[first_item$y$label] # Class name of the label
#'
#' # Visualize
#' overlay <- draw_segmentation_masks(first_item)
#' tensor_image_browse(overlay)
#' }
#'
#' @family segmentation_dataset
#' @export
oxfordiiitpet_segmentation_dataset <- torch::dataset(
  name = "oxfordiiitpet segmentation",
  resources = list(
    c("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
    c("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f")
  ),
  training_file = "trainval.rds",
  test_file = "test.rds",
  archive_size = "750 MB",
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
    if (train) {
      self$split <- "train"
    } else {
      self$split <- "test"
    }

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (train) {
      data_file <- self$training_file
    } else {
      data_file <- self$test_file
    }

    data <- readRDS(file.path(self$processed_folder, data_file))
    self$img_path <- data$img_path
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
    if (self$target_type == "category") {
      self$classes <- names(self$class_to_idx)
    } else {
      self$classes <- c("Cat", "Dog")
    }

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  download = function() {

    if (self$check_exists()) {
      return()
    }

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    for (r in self$resources) {
      url <- r[1]
      archive <- download_and_cache(url, prefix = class(self)[1])
      actual_md5 <- tools::md5sum(archive)

      if (actual_md5 != r[2]) {
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")
      }

      utils::untar(archive, exdir = self$raw_folder)
    }

    for (split in c("trainval", "test")) {
      ann_file <- file.path(self$raw_folder, "annotations", glue::glue("{split}.txt"))

      ann <- read.delim(
        ann_file,
        sep = " ",
        header = FALSE,
        col.names = c("img_id", "label", "bin_label", "other")
      )

      img_paths <- file.path(self$raw_folder, "images", glue::glue("{ann$img_id}.jpg"))
      seg_paths <- file.path(self$raw_folder, "annotations", "trimaps", glue::glue("{ann$img_id}.png"))

      valid <- file.exists(img_paths) & file.exists(seg_paths)

      img_path <- img_paths[valid]
      labels <- ann$label[valid]

      raw_classes <- sub("_\\d+$", "", ann$img_id[valid])
      self$classes <- unique(raw_classes)
      self$classes <- gsub("_", " ", self$classes, fixed = TRUE)
      class_to_idx <- setNames(seq_along(self$classes), self$classes)

      saveRDS(
        list(
          img_path = img_path,
          labels = labels,
          class_to_idx = class_to_idx
        ),
        file.path(self$processed_folder, glue::glue("{split}.rds"))
      )
    }

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) && fs::file_exists(file.path(self$processed_folder, self$test_file))
  },

  .getitem = function(index) {
    x <- jpeg::readJPEG(self$img_path[index])

    label <- self$labels[index]

    seg_name <- basename(self$img_path[index])
    mask_path <- file.path(self$raw_folder, "annotations", "trimaps", sub("\\.jpg$", ".png", seg_name))
    mask_int <- torch_tensor(png::readPNG(mask_path) * 255)
    masks <- torch_stack(list(mask_int == 1, mask_int == 2, mask_int == 3))

    if (self$target_type == "binary-category") {
      class_name <- names(self$class_to_idx)[label]
      if (substr(class_name, 1, 1) %in% LETTERS) {
        label <- 1L
      } else {
        label <- 2L
      }

      self$classes <- c("Cat","Dog")
    }

    y <- list(
      masks = masks,
      label = label
    )

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }

    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    result <- list(x = x, y = y)
    class(result) <- c("image_with_segmentation_mask", class(result))
    result
  },

  .length = function() {
    length(self$img_path)
  },

  active = list(
    raw_folder = function() {
      fs::dir_create(self$root_path, "oxfordiiitpet", "raw")
      file.path(self$root_path, "oxfordiiitpet", "raw")
    },
    processed_folder = function() {
      fs::dir_create(self$root_path, "oxfordiiitpet", "processed")
      file.path(self$root_path, "oxfordiiitpet", "processed")
    }
  )
)

#' Oxford-IIIT Pet Datasets
#'
#' The Oxford-IIIT Pet collection is a **classification** dataset consisting of high-quality
#' images of 37 cat and dog breeds. It includes two variants:
#' - `oxfordiiitpet_dataset`: Multi-class classification across 37 pet breeds.
#' - `oxfordiiitpet_binary_dataset`: Binary classification distinguishing cats vs dogs.
#'
#' @inheritParams oxfordiiitpet_segmentation_dataset
#'
#' @return A torch dataset object \code{oxfordiiitpet_dataset} or \code{oxfordiiitpet_binary_dataset}.
#' Each element is a named list with:
#' - `x`: A H x W x 3 integer array representing an RGB image.
#' - `y`: An integer label:
#'   - For `oxfordiiitpet_dataset`: a value from 1–37 representing the breed.
#'   - For `oxfordiiitpet_binary_dataset`: 1 for Cat, 2 for Dog.
#'
#' @details
#' The Oxford-IIIT Pet dataset contains over 7,000 images across 37 categories,
#' with roughly 200 images per class. Each image is labeled with its breed and species (cat/dog).
#'
#' @examples
#' \dontrun{
#' # Multi-class version
#' oxford <- oxfordiiitpet_dataset(download = TRUE)
#' first_item <- oxford[1]
#' first_item$x  # RGB image
#' first_item$y  # Label in 1–37
#' oxford$classes[first_item$y]  # Breed name
#'
#' # Binary version
#' oxford_bin <- oxfordiiitpet_binary_dataset(download = TRUE)
#' first_item <- oxford_bin[1]
#' first_item$x  # RGB image
#' first_item$y  # 1 for Cat, 2 for Dog
#' oxford_bin$classes[first_item$y]  # "Cat" or "Dog"
#' }
#'
#' @name oxfordiiitpet_dataset
#' @title Oxford-IIIT Pet Classification Datasets
#' @rdname oxfordiiitpet_dataset
#' @family classification_dataset
#' @export
oxfordiiitpet_dataset <- dataset(
  inherit = oxfordiiitpet_segmentation_dataset,
  name = "oxfordiiitpet",

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    if (train) {
      self$split <- "train"
    } else {
      self$split <- "test"
    }

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (train) {
      data_file <- self$training_file
    } else {
      data_file <- self$test_file
    }
    data <- readRDS(file.path(self$processed_folder, data_file))
    self$img_path <- data$img_path
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
    self$classes <- names(self$class_to_idx)

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  .getitem = function(index) {
    x <- jpeg::readJPEG(self$img_path[index])

    y <- self$labels[index]

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }

    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    list(x = x, y = y)
  }
)

#' @rdname oxfordiiitpet_dataset
#' @export
oxfordiiitpet_binary_dataset <- dataset(
  inherit = oxfordiiitpet_segmentation_dataset,
  name = "oxfordiiitpet binary",

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    if (train) {
      self$split <- "train"
    } else {
      self$split <- "test"
    }

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (train) {
      data_file <- self$training_file
    } else {
      data_file <- self$test_file
    }
    data <- readRDS(file.path(self$processed_folder, data_file))
    self$img_path <- data$img_path
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
    self$classes <- c("Cat", "Dog")

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  .getitem = function(index) {
    x <- jpeg::readJPEG(self$img_path[index])

    label <- self$labels[index]
    class_name <- names(self$class_to_idx)[label]
    if (substr(class_name, 1, 1) %in% LETTERS) {
      y <- 1L
    } else {
      y <- 2L
    }

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }

    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    list(x = x, y = y)
  }
)
