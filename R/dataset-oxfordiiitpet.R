#' Oxford-IIIT Pet Dataset
#'
#' Loads the Oxford-IIIT Pet Dataset, which contains images of 37 pet breeds (cats and dogs),
#' each annotated with a class label and optional segmentation mask.
#' Supports category labels, binary-category labels (cat/dog), and pixel-level segmentation maps.
#' Dataset is automatically split into train and test subsets.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/oxfordiiitpet`.
#' @param train Logical. Whether to load the training split. Default is `TRUE`.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform labels.
#' @param target_type Character. Type of target to load: one of `"category"`, `"binary-category"`, or `"segmentation"`.
#' Default is `"category"`.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @return An `oxfordiiitpet_dataset` object representing the dataset, with fields:
#' \itemize{
#'   \item \code{x}: image tensor.
#'   \item \code{y}: class index or segmentation mask.
#'   \item \code{class_name}: (if applicable) class label as a string.
#' }
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' oxford <- oxfordiiitpet_dataset(root = root_dir, train = TRUE, download = TRUE)
#' first_item <- oxford[1]
#' # image tensor of first item
#' first_item$x
#' # class index or segmentation mask
#' first_item$y
#' # class name (if applicable)
#' first_item$class_name
#' }
#'
#' @name oxfordiiitpet_dataset
#' @aliases oxfordiiitpet_dataset
#' @title Oxford-IIIT Pet Dataset
#' @export
oxfordiiitpet_dataset <- dataset(
  name = "oxfordiiitpet",
  resources = list(
    c("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
    c("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f")
  ),
  training_file = "trainval.rds",
  test_file = "test.rds",
  initialize = function(root, train = TRUE, transform = NULL, target_transform = NULL,
                        target_type = "category", download = FALSE) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$target_type <- target_type

    if (download)
      rlang::inform("Downloading the Oxford-IIIT Pet Dataset...")
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    data_file <- if (train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, data_file))

    self$image_paths <- data$image_paths
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
  },
  download = function() {
    if (self$check_exists())
      return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      url <- r[1]
      md5 <- r[2]
      filename <- basename(url)
      destpath <- file.path(self$raw_folder, filename)

      p <- download_and_cache(url, prefix = class(self)[1])
      fs::file_copy(p, destpath, overwrite = TRUE)

      if (!tools::md5sum(destpath) == md5)
        runtime_error(paste("MD5 mismatch for:", url))

      utils::untar(destpath, exdir = self$raw_folder)
    }

    rlang::inform("Processing Oxford-IIIT Pet dataset...")

    for (split in c("trainval", "test")) {
      ann_file <- file.path(self$raw_folder, "annotations", paste0(split, ".txt"))
      lines <- readLines(ann_file)

      image_paths <- character()
      labels <- integer()
      raw_classes <- character()

      for (line in lines) {
        parts <- strsplit(line, " ")[[1]]
        img_id <- parts[1]
        label <- as.integer(parts[2])
        bin_label <- as.integer(parts[3])

        img_path <- file.path(self$raw_folder, "images", paste0(img_id, ".jpg"))
        seg_path <- file.path(self$raw_folder, "annotations", "trimaps", paste0(img_id, ".png"))

        image_paths <- c(image_paths, if (self$target_type == "segmentation") seg_path else img_path)
        labels <- c(labels, switch(
          self$target_type,
          "category" = label,
          "binary-category" = bin_label,
          "segmentation" = NA,
          label
        ))

        raw_classes <- c(raw_classes, sub("_\\d+$", "", img_id))
      }

      class_names <- unique(raw_classes)
      class_names <- sapply(class_names, function(x) gsub("_", " ", tools::toTitleCase(x)))
      class_to_idx <- setNames(seq_along(class_names), class_names)

      saveRDS(list(
        image_paths = image_paths,
        labels = labels,
        class_to_idx = class_to_idx
      ), file.path(self$processed_folder, paste0(split, ".rds")))
    }

    rlang::inform("Processed Oxford-IIIT Pet dataset Successfully !")
  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img <- torchvision::transform_to_tensor(magick::image_read(self$image_paths[index]))

    if (!is.null(self$transform))
        img <- self$transform(img)

    label <- self$labels[index]

    if (!is.null(self$target_transform))
        label <- self$target_transform(label)

    if (self$target_type == "segmentation") {
        class_name <- NA
    } else {
        class_name <- names(self$class_to_idx)[label]
    }

    list(x = img, y = label, class_name = class_name)
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
