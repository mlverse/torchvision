#' Oxford-IIIT Pet Dataset
#'
#' Loads the Oxford-IIIT Pet Dataset, which contains images of 37 pet breeds (cats and dogs),
#' each annotated with a class label and optional segmentation mask.  
#' Supports three types of targets: class category labels, binary-category labels (Cat vs Dog),
#' and pixel-level segmentation masks.
#'
#' The dataset is automatically split into `"train"` and `"test"` subsets based on the official annotation files.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/oxfordiiitpet`.
#' @param train Logical. Whether to load the training split. Default is `TRUE`.
#' @param transform Optional function to transform the input image tensors.
#' @param target_transform Optional function to transform target labels or segmentation masks.
#' @param target_type Character. One of:
#' \describe{
#'   \item{\code{"category"}}{(default) Integer class index from 1 to 37.}
#'   \item{\code{"binary-category"}}{Returns 1 for Cat and 2 for Dog. Based on whether the class name starts with an uppercase letter.}
#'   \item{\code{"segmentation"}}{Returns a single-channel pixel-level segmentation mask as a torch tensor.}
#' }
#' @param download Logical. Whether to download the dataset if not already present. Default is `FALSE`.
#'
#' @return A dataset object of class `oxfordiiitpet_dataset`, where each indexed item returns a named list:
#' \describe{
#'   \item{\code{x}}{An image tensor of shape (3, H, W).}
#'   \item{\code{y}}{The label — an integer index, a binary category, or a segmentation mask tensor depending on \code{target_type}.}
#' }
#'
#' @examples
#' \dontrun{
#' # Load training data with class indices
#' ds <- oxfordiiitpet_dataset(train = TRUE, download = TRUE)
#' item <- ds[1]
#' item$x  # image tensor
#' item$y  # class index
#'
#' # Load binary-category labels: 1 = Cat, 2 = Dog
#' ds_bin <- oxfordiiitpet_dataset(root = root_dir, train = TRUE, target_type = "binary-category")
#'
#' # Load segmentation masks
#' ds_seg <- oxfordiiitpet_dataset(root = root_dir, train = TRUE, target_type = "segmentation")
#'
#' # Use custom collate function for variable image sizes
#' dl <- dataloader(ds, batch_size = 4, collate_fn = collate_fn_variable_size)
#' batch <- dataloader_next(dataloader_make_iter(dl))
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
  initialize = function(root = tempdir(), train = TRUE, transform = NULL, target_transform = NULL,
                        target_type = "category", download = FALSE) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$target_type <- target_type

    if (download)
      rlang::inform("Oxford-IIIT Pet Dataset (~811MB) will be downloaded and processed if not already available.")
      rlang::inform("Downloading and preparing the Oxford-IIIT Pet dataset...")
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    data_file <- if (train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, data_file))

    self$image_paths <- data$image_paths
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
    self$classes <- names(self$class_to_idx)
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

      utils::untar(destpath, exdir = self$raw_folder)  # Ensures 'annotations' folder is inside self$raw_folder
    }

    rlang::inform("Preparing image paths and labels from annotations...")

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

        seg_path <- file.path(self$raw_folder, "annotations", "trimaps", paste0(img_id, ".png"))
        if (!file.exists(seg_path)) {
          warning("Missing segmentation file: ", seg_path)
        }
        image_paths <- c(image_paths, seg_path)
        labels <- c(labels, NA)

        raw_classes <- c(raw_classes, sub("_\\d+$", "", img_id))
      }

      class_names <- unique(raw_classes)
      class_names <- sapply(class_names, function(x) gsub("_", " ", x))
      class_to_idx <- setNames(seq_along(class_names), class_names)

      saveRDS(list(
        image_paths = image_paths,
        labels = labels,
        class_to_idx = class_to_idx
      ), file.path(self$processed_folder, paste0(split, ".rds")))
      rlang::inform(glue::glue("Loaded {length(labels)} samples for split: {split}"))
    }

    rlang::inform("Oxford-IIIT Pet dataset downloaded and processed successfully.")
  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img <- torchvision::transform_to_tensor(magick::image_read(self$image_paths[index]))

    if (!is.null(self$transform))
      img <- self$transform(img)
    if (self$target_type == "segmentation") {
      mask <- magick::image_read(self$image_paths[index])
      mask_tensor <- torchvision::transform_to_tensor(mask)[1, , ]  # Trimap is single-channel
      label <- mask_tensor
    } else if (self$target_type == "binary-category") {
      label_index <- self$labels[index]
      original_class <- names(self$class_to_idx)[label_index]
      if (!is.na(original_class) && substr(original_class, 1, 1) == toupper(substr(original_class, 1, 1))) {
        label <- 1
      } else {
        label <- 2
      }
      if (!is.null(self$target_transform))
        label <- self$target_transform(label)
    } else {  
      label <- self$labels[index]
      if (!is.null(self$target_transform))
        label <- self$target_transform(label)
    }
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
