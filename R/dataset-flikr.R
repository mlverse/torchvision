#' Flickr8k Dataset
#'
#' Loads the Flickr8k dataset consisting of 8,000 images with five captions each.
#' The dataset is split into training and test sets with 6,000 and 1,000 images respectively.
#' Images are resized to a fixed size and captions are stored as lists of strings.
#'
#' @param root Character. Root directory where the dataset will be stored under `root/flickr8k`.
#' @param train Logical. If `TRUE`, loads the training split, otherwise the test split. Default is `TRUE`.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform captions.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @return A flickr8k_dataset object representing the dataset.
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' flickr8k <- flickr8k_dataset(root = root_dir, train = TRUE, download = TRUE)
#' first_item <- flickr8k[1]
#' # image tensor of first item
#' first_item$x
#' # list of captions of first item
#' first_item$y
#' }
#'
#' @name flickr8k_dataset
#' @aliases flickr8k_dataset
#' @title Flickr8k Dataset
#' @export
flickr8k_dataset <- dataset(
  name = "flickr8k",
  training_file = "train.rds",
  test_file = "test.rds",
  resources = list(
    c("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
      "bf6c1abcb8e4a833b7f922104de18627"),
    c("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
      "f18a1e2920de5bd84dae7cf08ec78978")
  ),
  initialize = function(root, train = TRUE, transform = NULL, target_transform = NULL, download = FALSE) {
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to fetch it.")

    if (self$train)
      data <- readRDS(file.path(self$processed_folder, self$training_file))
    else
      data <- readRDS(file.path(self$processed_folder, self$test_file))

    self$images <- data$images
    self$captions <- data$captions
  },
  download = function() {
    if (self$check_exists())
      return(invisible(NULL))

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
  zip_path <- download_and_cache(r[1], prefix = class(self)[1])
  
  md5_actual <- tools::md5sum(zip_path)
  if (md5_actual != r[2]) {
    runtime_error(glue::glue("MD5 sums do not match for: {r[1]}. Expected {r[2]}, got {md5_actual}"))
  }
  dest_zip <- file.path(self$raw_folder, basename(zip_path))
  fs::file_copy(zip_path, dest_zip, overwrite = TRUE)
  utils::unzip(dest_zip, exdir = self$raw_folder)
}



    rlang::inform("Processing Flickr8k captions...")

    captions_file <- file.path(self$raw_folder, "Flickr8k.token.txt")
    captions_lines <- readLines(captions_file)
    captions_map <- list()

    for (line in captions_lines) {
      parts <- strsplit(line, "\t")[[1]]
      img_id <- strsplit(parts[1], "#")[[1]][1]
      caption <- parts[2]

      if (!img_id %in% names(captions_map)) {
        captions_map[[img_id]] <- list()
      }
      captions_map[[img_id]] <- c(captions_map[[img_id]], caption)
    }

    train_ids <- readLines(file.path(self$raw_folder, "Flickr_8k.trainImages.txt"))
    test_ids <- readLines(file.path(self$raw_folder, "Flickr_8k.testImages.txt"))

    process_split <- function(ids) {
      img_paths <- file.path(self$raw_folder, "Flicker8k_Dataset", ids)
      img_paths <- img_paths[file.exists(img_paths)]
      captions <- lapply(ids, function(id) captions_map[[id]])
      list(images = img_paths, captions = captions)
    }

    train_data <- process_split(train_ids)
    test_data <- process_split(test_ids)

    saveRDS(train_data, file.path(self$processed_folder, self$training_file))
    saveRDS(test_data, file.path(self$processed_folder, self$test_file))

    rlang::inform("Done processing Flickr8k.")
  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img_path <- self$images[[index]]
    img <- magick::image_read(img_path)
    img <- magick::image_resize(img, "224x224!")
    img_tensor <- torchvision::transform_to_tensor(img)
    if (!is.null(self$transform))
      img_tensor <- self$transform(img_tensor)
    target <- self$captions[[index]]
    if (!is.null(self$target_transform))
      target <- self$target_transform(target)
    list(x = img_tensor, y = target)
  },

  .length = function() {
    length(self$images)
  },
  active = list(
    raw_folder = function() {
      file.path(self$root, "flickr8k", "raw")
    },
    processed_folder = function() {
      file.path(self$root, "flickr8k", "processed")
    }
  )
)
