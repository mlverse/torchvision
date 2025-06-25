#' Flickr8k Dataset
#'
#' Loads the Flickr8k dataset consisting of 8,000 images with five human-annotated captions per image.
#'
#' The dataset is split into:
#' - `"train"`: training subset with captions.
#' - `"test"`: test subset with captions.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/flickr8k`.
#' @param train Logical. If `TRUE`, loads the training set. If `FALSE`, loads the test set. Default is `TRUE`.
#' @param transform Optional function to apply to each image (e.g., resize, normalization). Images are RGB of varied dimensions.
#' @param target_transform Optional function to transform the captions. Default is `NULL`.
#' @param download Logical. Whether to download and process the dataset if it's not already available. Default is `FALSE`.
#'
#' @return An object of class \code{flickr8k_dataset}, which behaves like a torch dataset.
#' Each element is a named list:
#' - `x`: a H x W x 3 integer array representing an RGB image.
#' - `y`: a character vector of captions for the image.
#'
#' @examples
#' \dontrun{
#' flickr <- flickr8k_dataset(train = TRUE, download = TRUE)
#'
#' # Define a custom collate function to resize images in the batch
#' resize_collate_fn <- function(batch) {
#'   xs <- lapply(batch, function(sample) {
#'     torchvision::transform_resize(sample$x, c(224, 224))
#'   })
#'   xs <- torch::torch_stack(xs)
#'   ys <- sapply(batch, function(sample) sample$y)
#'   list(x = xs, y = ys)
#' }
#'
#' dl <- torch::dataloader(dataset = flickr, batch_size = 4, collate_fn = resize_collate_fn)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#' batch$x  # batched image tensors resized to 224x224
#' batch$y  # list of caption vectors
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
    c("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip","bf6c1abcb8e4a833b7f922104de18627"),
    c("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip","f18a1e2920de5bd84dae7cf08ec78978")
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE) {

    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$split <- if (train) "train" else "test"

    cli_inform("Flickr8k Dataset (~1GB) will be downloaded and processed if not already cached.")
    
    if (download) {
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. Use `download = TRUE` to fetch it.")
    }

    file <- if (self$train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, file))

    self$images <- data$images
    self$captions <- data$captions
    cli_inform("Split '{self$split}' loaded with {length(self$images)} samples.")
  },

  download = function() {
    if (self$check_exists()){
      return()
    }

    cli_inform("Downloading Flickr8k split: '{self$split}' (~1GB)")
    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      zip_path <- download_and_cache(r[1], prefix = class(self)[1])

      if (tools::md5sum(zip_path) != r[2]) {
        cli_abort("Corrupt file! Delete the file at {zip_path} and try again.")
      }
      dest_zip <- file.path(self$raw_folder, basename(zip_path))
      fs::file_copy(zip_path, dest_zip, overwrite = TRUE)
      utils::unzip(dest_zip, exdir = self$raw_folder)
    }

    captions_file <- file.path(self$raw_folder, "Flickr8k.token.txt")
    captions_lines <- readLines(captions_file)
    captions_map <- list()

    for (line in captions_lines) {
      parts <- strsplit(line, "\t")[[1]]
      img_id <- strsplit(parts[1], "#")[[1]][1]
      caption <- parts[2]
      captions_map[[img_id]] <- c(captions_map[[img_id]], caption)
    }

    train_ids <- readLines(file.path(self$raw_folder, "Flickr_8k.trainImages.txt"))
    test_ids <- readLines(file.path(self$raw_folder, "Flickr_8k.testImages.txt"))

    process_split <- function(ids, split_name) {
      img_paths <- file.path(self$raw_folder, "Flicker8k_Dataset", ids)
      img_paths <- img_paths[file.exists(img_paths)]
      captions <- lapply(ids, function(id) captions_map[[id]])
      list(images = img_paths, captions = captions)
    }

    train_data <- process_split(train_ids, "train")
    test_data <- process_split(test_ids, "test")

    saveRDS(train_data, file.path(self$processed_folder, self$training_file))
    saveRDS(test_data, file.path(self$processed_folder, self$test_file))
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) && fs::file_exists(file.path(self$processed_folder, self$test_file))
  },

  .getitem = function(index) {
    img_path <- self$images[[index]]
    img <- magick::image_read(img_path)
    img <- magick::image_data(img, channels = "rgb")
    img <- as.integer(img)
    target <- self$captions[[index]]

    if (!is.null(self$transform)) {
      img <- self$transform(img)
    }

    if (!is.null(self$target_transform)) {
      target <- self$target_transform(target)
    }
    list(x = img, y = target)
  },

  .length = function() {
    length(self$images)
  },

  active = list(
    raw_folder = function() file.path(self$root, "flickr8k", "raw"),
    processed_folder = function() file.path(self$root, "flickr8k", "processed")
  )
)

#' Flickr30k Dataset
#'
#' Loads the Flickr30k dataset consisting of 30,000 images with five human-annotated captions per image.
#'
#' The dataset is split into:
#' - `"train"`: training subset with captions.
#' - `"test"`: test subset with captions.
#'
#' @inheritParams flickr8k_dataset
#' @param root Character. Root directory where the dataset will be stored under `root/flickr30k`.
#'
#' @return An object of class \code{flickr30k_dataset}, which behaves like a torch dataset.
#' Each element is a named list:
#' - `x`: a H x W x 3 integer array representing an RGB image.
#' - `y`: a character vector of captions for the image.
#'
#' @examples
#' \dontrun{
#' flickr <- flickr30k_dataset(train = TRUE, download = TRUE)
#'
#' # Define a custom collate function to resize images in the batch
#' resize_collate_fn <- function(batch) {
#'   xs <- lapply(batch, function(sample) {
#'     torchvision::transform_resize(sample$x, c(224, 224))
#'   })
#'   xs <- torch::torch_stack(xs)
#'   ys <- sapply(batch, function(sample) sample$y)
#'   list(x = xs, y = ys)
#' }
#'
#' dl <- torch::dataloader(dataset = flickr, batch_size = 4, collate_fn = resize_collate_fn)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#' batch$x  # batched image tensors resized to 224x224
#' batch$y  # list of caption vectors
#' }
#'
#' @name flickr30k_dataset
#' @aliases flickr30k_dataset
#' @title Flickr30k Dataset
#' @export
flickr30k_dataset <- dataset(
  name = "flickr30k",
  resources = list(
    c("https://uofi.app.box.com/shared/static/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl.gz","985ac761bbb52ca49e0c474ae806c07c"),
    c("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip","4fa8c08369d22fe16e41dc124bd1adc2")
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE) {

    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train

    self$split <- if (self$train) "train" else "test"
    cli_inform("Flickr30k Dataset (~4.1GB) will be downloaded and processed if not already cached.")

    if (download) {
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. Use `download = TRUE` to fetch it.")
    }

    captions_path <- file.path(self$raw_folder, "dataset_flickr30k.json")
    captions_json <- jsonlite::fromJSON(captions_path)

    split_name <- if (self$train) "train" else "test"
    imgs_df <- captions_json$images
    filtered_images <- imgs_df[imgs_df$split == split_name, ]
    self$filenames <- filtered_images$filename

    self$captions_map <- list()
    for (i in seq_len(nrow(filtered_images))) {
      img_entry <- filtered_images[i, ]
      filename <- img_entry$filename
      sentences_list <- img_entry$sentences[[1]]

      captions <- sentences_list$raw
      captions <- captions[!is.na(captions) & nzchar(captions)]

      self$captions_map[[filename]] <- captions
    }

    cli_inform("Split '{self$split}' loaded with {length(self$filenames)} samples.")
  },

  download = function() {
    if (self$check_exists()){
      return()
    }

    cli_inform("Downloading Flickr30k split: '{self$split}' (~4.1GB)")
    fs::dir_create(self$raw_folder)

    for (r in self$resources) {
      archive_path <- download_and_cache(r[1], prefix = class(self)[1])

      if (tools::md5sum(archive_path) != r[2]) {
        cli_abort("Corrupt file! Delete the file at {archive_path} and try again.")
      }

      dest_path <- file.path(self$raw_folder, basename(archive_path))
      fs::file_copy(archive_path, dest_path, overwrite = TRUE)

      if (grepl("\\.zip$", dest_path)) {
        utils::unzip(dest_path, exdir = self$raw_folder)
      } else if (grepl("\\.tar\\.gz$", dest_path)) {
        utils::untar(dest_path, exdir = self$raw_folder)
      } else if (grepl("\\.gz$", dest_path)) {
        tar_path <- sub("\\.gz$", "", dest_path)
        gunzip_base(dest_path, tar_path)
        utils::untar(tar_path, exdir = self$raw_folder)
      }
    }
  },

  check_exists = function() {
    fs::file_exists(file.path(self$raw_folder, "dataset_flickr30k.json")) && fs::dir_exists(file.path(self$raw_folder, "flickr30k-images"))
  },

  .getitem = function(index) {
    fname <- self$filenames[[index]]
    img_path <- file.path(self$raw_folder, "flickr30k-images", fname)
    img <- magick::image_read(img_path)
    img <- magick::image_data(img, channels = "rgb")
    img <- as.integer(img)
    target <- self$captions_map[[fname]]

    if (!is.null(self$transform))
      img <- self$transform(img)

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img, y = target)
  },

  .length = function() {
    length(self$filenames)
  },

  active = list(
    raw_folder = function() {
      file.path(self$root, "flickr30k", "raw")
    }
  )
)

gunzip_base <- function(src, dest) {
  in_con <- gzfile(src, "rb")
  out_con <- file(dest, "wb")
  on.exit({ close(in_con); close(out_con) }, add = TRUE)

  repeat {
    bytes <- readBin(in_con, what = raw(), n = 65536)
    if (length(bytes) == 0) break
    writeBin(bytes, out_con)
  }

  invisible(dest)
}