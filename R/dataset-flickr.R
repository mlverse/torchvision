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
#' flickr8k <- flickr8k_dataset(download = TRUE)
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
  initialize = function(root = tempdir(), train = TRUE, transform = NULL, target_transform = NULL, download = FALSE) {
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train

    split <- if (train) "train" else "test"
    rlang::inform(glue::glue("Flickr8k Dataset (~1GB) will be downloaded and processed if not already cached."))
    
    if (download) {
      rlang::inform(glue::glue("Downloading Flickr8k split: '{split}' (~1GB)"))
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use `download = TRUE` to fetch it.")
    }

    rlang::inform(glue::glue("Loading processed split: '{split}'"))

    file <- if (self$train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, file))

    self$images <- data$images
    self$captions <- data$captions
    rlang::inform(glue::glue("Split '{split}' loaded with {length(self$images)} samples."))
  },
  download = function() {
    if (self$check_exists()) return(invisible(NULL))

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      zip_path <- download_and_cache(r[1], prefix = class(self)[1])
      md5_actual <- tools::md5sum(zip_path)
      if (md5_actual != r[2]) {
        runtime_error(glue::glue("MD5 mismatch for {r[1]}.\nExpected: {r[2]}, Got: {md5_actual}"))
      }
      dest_zip <- file.path(self$raw_folder, basename(zip_path))
      fs::file_copy(zip_path, dest_zip, overwrite = TRUE)
      utils::unzip(dest_zip, exdir = self$raw_folder)
    }

    rlang::inform("Extracting images and processing captions (may take a minute)...")

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
      rlang::inform(glue::glue("Done processing split: '{split_name}' ({length(img_paths)} samples saved)"))
      list(images = img_paths, captions = captions)
    }

    train_data <- process_split(train_ids, "train")
    test_data <- process_split(test_ids, "test")

    saveRDS(train_data, file.path(self$processed_folder, self$training_file))
    saveRDS(test_data, file.path(self$processed_folder, self$test_file))
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
    raw_folder = function() file.path(self$root, "flickr8k", "raw"),
    processed_folder = function() file.path(self$root, "flickr8k", "processed")
  )
)

#' Flickr30k Dataset
#'
#' Loads the Flickr30k dataset consisting of 31,014 images, each annotated with five captions.
#' The dataset is split into training and test sets using the official Karpathy splits.
#' Images are resized to a fixed 224x224 resolution, and captions are extracted as lists of raw strings.
#'
#' @inheritParams flickr8k_dataset
#' @param root Character. Root directory where the dataset will be stored under `root/flickr30k`.
#'
#' @return A flickr30k_dataset object representing the dataset with images and their corresponding captions.
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' flickr30k <- flickr30k_dataset(download = TRUE)
#' first_item <- flickr30k[1]
#' # Tensor representing the image
#' first_item$x
#' # List of captions for the image
#' first_item$y
#' }
#'
#' @name flickr30k_dataset
#' @aliases flickr30k_dataset
#' @title Flickr30k Dataset
#' @export
flickr30k_dataset <- dataset(
  name = "flickr30k",
  resources = list(
    c(
      "https://uofi.app.box.com/shared/static/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl.gz",
      "985ac761bbb52ca49e0c474ae806c07c"
    ),
    c(
      "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
      "4fa8c08369d22fe16e41dc124bd1adc2"
    )
  ),

  initialize = function(root = tempdir(), train = TRUE, transform = NULL, target_transform = NULL, download = FALSE) {
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train

    split <- if (self$train) "train" else "test"
    rlang::inform("Flickr30k Dataset (~4.1GB) will be downloaded and processed if not already cached.")

    if (download) {
      rlang::inform(glue::glue("Downloading Flickr30k split: '{split}' (~4.1GB)"))
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use `download = TRUE` to fetch it.")
    }

    rlang::inform("Extracting images and processing metadata (may take a minute)...")

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

    rlang::inform(glue::glue("Done processing split: '{split}' ({length(self$filenames)} samples saved)"))
    rlang::inform(glue::glue("Loading processed split: '{split}'"))
    rlang::inform(glue::glue("Split '{split}' loaded with {length(self$filenames)} samples."))
  },

  download = function() {
    if (self$check_exists()) return(invisible(NULL))

    fs::dir_create(self$raw_folder)

    for (r in self$resources) {
      archive_path <- download_and_cache(r[1], prefix = class(self)[1])
      md5_actual <- tools::md5sum(archive_path)

      if (md5_actual != r[2]) {
        runtime_error(glue::glue("MD5 mismatch: expected {r[2]}, got {md5_actual}"))
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
    fs::file_exists(file.path(self$raw_folder, "dataset_flickr30k.json")) &&
      fs::dir_exists(file.path(self$raw_folder, "flickr30k-images"))
  },

  .getitem = function(index) {
    fname <- self$filenames[[index]]
    img_path <- file.path(self$raw_folder, "flickr30k-images", fname)
    img <- magick::image_read(img_path)
    img <- magick::image_resize(img, "224x224!")
    img_tensor <- torchvision::transform_to_tensor(img)

    if (!is.null(self$transform))
      img_tensor <- self$transform(img_tensor)

    target <- self$captions_map[[fname]]

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img_tensor, y = target)
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