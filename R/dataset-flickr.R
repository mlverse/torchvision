#' Flickr8k Dataset
#'
#' Loads the Flickr8k dataset consisting of 8,000 images with five human-annotated captions per image. 
#' The images in this dataset are in RGB format and vary in spatial resolution.
#'
#' The dataset is split into:
#' - `"train"`: training subset with captions.
#' - `"test"`: test subset with captions.
#'
#' @inheritParams fgvc_aircraft_dataset
#' @param root : Root directory for dataset storage. The dataset will be stored under `root/flickr8k`.
#' @param train : If `TRUE`, loads the training set. If `FALSE`, loads the test set. Default is `TRUE`.
#'
#' @return A torch dataset of class \code{flickr8k_caption_dataset}.
#' Each element is a named list:
#' - `x`: a H x W x 3 integer array representing an RGB image.
#' - `y`: a character vector containing all five captions associated with the image.
#'
#' @examples
#' \dontrun{
#' # Load the Flickr8k caption dataset with inline transformation
#' flickr8k <- flickr8k_caption_dataset(
#'   transform = function(x) {
#'     x %>%
#'       transform_to_tensor() %>%
#'       transform_resize(c(224, 224))
#'   },
#'   target_transform = function(y) glue::glue_collapse(y, sep = " "),
#'   download = TRUE
#' )
#'
#' # Create a dataloader and retrieve a batch
#' dl <- dataloader(flickr8k, batch_size = 4)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#'
#' # Access images and captions
#' batch$x  # batched image tensors with shape (4, 3, 224, 224)
#' batch$y  # batched character vector containing all five captions concatenated.
#' }
#'
#' @name flickr8k_caption_dataset
#' @aliases flickr8k_caption_dataset
#' @title Flickr8k Caption Dataset
#' @export
flickr8k_caption_dataset <- torch::dataset(
  name = "flickr8k",
  training_file = "train.rds",
  test_file = "test.rds",
  class_index_file = "classes.rds",

  resources = list(
    c("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip", "bf6c1abcb8e4a833b7f922104de18627"),
    c("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", "f18a1e2920de5bd84dae7cf08ec78978")
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$split <- if (train) "train" else "test"

    cli_inform("{.cls {class(self)[[1]]}} Dataset (~1GB) will be downloaded and processed if not already cached.")
    
    if (download)
      self$download()

    if (!self$check_exists())
      cli_abort("Dataset not found. Use `download = TRUE` to fetch it.")

    if (!self$check_processed_exists()) {
      fs::dir_create(self$processed_folder)

      caption_df <- read.delim(
        file.path(self$raw_folder, "Flickr8k.token.txt"),
        sep = "\t",
        header = FALSE,
        col.names = c("file_id", "caption"),
        stringsAsFactors = FALSE
      )
      caption_df[c("file", "id")] <- do.call(rbind, strsplit(caption_df$file_id, "#"))
      caption_df$file_id <- NULL
      caption_df$file <- trimws(caption_df$file)
      captions_map <- split(caption_df$caption, caption_df$file)
      all_ids <- sort(unique(caption_df$file))

      caption_to_index <- setNames(seq_along(all_ids), all_ids)
      unique_merged_captions <- captions_map[all_ids]

      saveRDS(unique_merged_captions, file.path(self$processed_folder, self$class_index_file))

      train_ids <- readLines(file.path(self$raw_folder, "Flickr_8k.trainImages.txt"))
      test_ids <- readLines(file.path(self$raw_folder, "Flickr_8k.testImages.txt"))

      process_split <- function(ids) {
        ids <- trimws(ids)
        img_paths <- file.path(self$raw_folder, "Flicker8k_Dataset", ids)
        img_paths <- img_paths[file.exists(img_paths)]

        missing <- setdiff(ids, names(caption_to_index))
        if (length(missing) > 0) {
          cli_abort("The following IDs are missing captions: {paste(missing, collapse=', ')}")
        }

        caption_indices <- vapply(ids, function(id) caption_to_index[[id]], integer(1))
        list(images = img_paths, captions = caption_indices)
      }

      saveRDS(process_split(train_ids), file.path(self$processed_folder, self$training_file))
      saveRDS(process_split(test_ids), file.path(self$processed_folder, self$test_file))
    }

    file <- if (train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, file))
    self$images <- data$images
    self$captions <- data$captions
    self$classes <- readRDS(file.path(self$processed_folder, self$class_index_file))

    cli_inform("Split '{self$split}' loaded with {length(self$images)} samples.")
  },

  download = function() {

    if (self$check_exists()) 
      return()

    cli_inform("Downloading {.cls {class(self)[[1]]}} split: '{self$split}'")
    fs::dir_create(self$raw_folder)

    for (r in self$resources) {
      archive <- download_and_cache(r[1], prefix = class(self)[1])
      if (tools::md5sum(archive) != r[2]) {
        cli_abort("Corrupt file! Delete the file at {archive} and try again.")
      }
      dest_zip <- file.path(self$raw_folder, basename(archive))
      fs::file_move(archive, dest_zip)

      if (tools::file_ext(dest_zip) == "zip") {
        utils::unzip(dest_zip, exdir = self$raw_folder)
      } else if (tools::file_ext(dest_zip) == "gz") {
        tar_path <- sub("\\.gz$", "", dest_zip)
        R.utils::gunzip(dest_zip, tar_path)
        utils::untar(tar_path, exdir = self$raw_folder)
      }
    }
  },

  check_processed_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
    fs::file_exists(file.path(self$processed_folder, self$test_file)) &&
    fs::file_exists(file.path(self$processed_folder, self$class_index_file))
  },

  check_exists = function() {
    fs::file_exists(file.path(self$raw_folder, "Flickr8k.token.txt")) &&
    fs::file_exists(file.path(self$raw_folder, "Flickr_8k.trainImages.txt")) &&
    fs::file_exists(file.path(self$raw_folder, "Flickr_8k.testImages.txt")) &&
    fs::dir_exists(file.path(self$raw_folder, "Flicker8k_Dataset"))
  },

  .getitem = function(index) {
    x <- jpeg::readJPEG(self$images[[index]])
    caption_index <- self$captions[[index]]
    y <- self$classes[[caption_index]]

    if (!is.null(self$transform)) 
      x <- self$transform(x)

    if (!is.null(self$target_transform)) 
      y <- self$target_transform(y)

    list(x = x, y = y)
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
#' The images in this dataset are in RGB format and vary in spatial resolution.
#'
#' The dataset is split into:
#' - `"train"`: training subset with captions.
#' - `"test"`: test subset with captions.
#'
#' @inheritParams flickr8k_caption_dataset
#' @param root Character. Root directory where the dataset will be stored under `root/flickr30k`.
#'
#' @return A torch dataset of class \code{flickr30k_caption_dataset}.
#' Each element is a named list:
#' - `x`: a H x W x 3 integer array representing an RGB image.
#' - `y`: a character vector containing all five captions associated with the image.
#'
#' @examples
#' \dontrun{
#' # Load the Flickr30k caption dataset with transformation
#' flickr30k <- flickr30k_caption_dataset(
#'   transform = function(x) {
#'     x %>%
#'       transform_to_tensor() %>%
#'       transform_resize(c(224, 224))
#'   },
#'   target_transform = function(y) glue::glue_collapse(y, sep = " "),
#'   download = TRUE
#' )
#'
#' # Create a dataloader and retrieve a batch
#' dl <- dataloader(flickr30k, batch_size = 4)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#'
#' # Access images and captions
#' batch$x  # batched image tensors with shape (4, 3, 224, 224)
#' batch$y  # batched character vector containing all five captions concatenated.
#' }
#'
#' @name flickr30k_caption_dataset
#' @aliases flickr30k_caption_dataset
#' @title Flickr30k Caption Dataset
#' @export
flickr30k_caption_dataset <- torch::dataset(
  name = "flickr30k",
  inherit = flickr8k_caption_dataset,
  resources = list(
    c("https://uofi.app.box.com/shared/static/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl.gz", "985ac761bbb52ca49e0c474ae806c07c"),
    c("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip", "4fa8c08369d22fe16e41dc124bd1adc2")
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$split <- if (train) "train" else "test"

    cli_inform("{.cls {class(self)[[1]]}} Dataset (~4.1GB) will be downloaded and processed if not already cached.")

    if (download)
      self$download()

    if (!self$check_exists()) 
      cli_abort("Dataset not found. Use `download = TRUE` to fetch it.")

    captions_path <- file.path(self$raw_folder, "dataset_flickr30k.json")
    captions_json <- jsonlite::fromJSON(captions_path)

    imgs_df <- captions_json$images
    filtered <- imgs_df[imgs_df$split == self$split, ]
    self$filenames <- filtered$filename

    captions_map <- setNames(
      lapply(filtered$sentences, function(s) unname(vapply(s$raw, identity, character(1)))),
      filtered$filename
    )

    all_ids <- names(captions_map)
    caption_to_index <- setNames(seq_along(all_ids), all_ids)

    self$images <- file.path(self$raw_folder, "flickr30k-images", self$filenames)
    self$captions <- vapply(self$filenames, function(f) caption_to_index[[f]], integer(1))
    self$classes <- captions_map

    cli_inform("Split '{self$split}' loaded with {length(self$images)} samples.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$raw_folder, "dataset_flickr30k.json")) &&
    fs::dir_exists(file.path(self$raw_folder, "flickr30k-images"))
  },

  active = list(
    raw_folder = function() file.path(self$root, "flickr30k", "raw")
  )
)