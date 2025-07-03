#' Flickr8k Dataset
#'
#' The Flickr8k and Flickr30k collections are **image captionning** datasets
#' composed of 8,000 and 30,000 color images respectively, each paired with five
#' human-annotated captions. The images are in RGB format with varying spatial
#' resolutions, and these datasets are widely used for training and evaluating
#' vision-language models.
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
#' # Load the Flickr8k caption dataset
#' flickr8k <- flickr8k_caption_dataset(download = TRUE)
#'
#' # Access the first item
#' first_item <- flickr8k[1]
#' first_item$x  # image array with shape {3, H, W}
#' first_item$y  # character vector containing five captions.
#'
#' # Load the Flickr30k caption dataset
#' flickr30k <- flickr30k_caption_dataset(download = TRUE)
#'
#' # Access the first item
#' first_item <- flickr30k[1]
#' first_item$x  # image array with shape {3, H, W}
#' first_item$y  # character vector containing five captions.
#' }
#'
#' @name flickr_caption_dataset
#' @title Flickr Caption Datasets
#' @rdname flickr_caption_dataset
#' @family caption_dataset
#' @export
flickr8k_caption_dataset <- torch::dataset(
  name = "flickr8k",
  training_file = "train.rds",
  test_file = "test.rds",
  class_index_file = "classes.rds",
  archive_size = "1 GB",

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
    
    if (download)
      cli_inform("{.cls {class(self)[[1]]}} Dataset (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
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
      caption_df[c("file", "id")] <- read.delim(
        text = caption_df$file,
        sep = "#",
        header = FALSE
      )

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

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$images)} images across {length(self$classes)} classes.")
  },

  download = function() {

    if (self$check_exists()) 
      return()

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    fs::dir_create(self$raw_folder)

    for (r in self$resources) {
      archive <- download_and_cache(r[1], prefix = class(self)[1])
      if (tools::md5sum(archive) != r[2]) {
        cli_abort("Corrupt file! Delete the file at {archive} and try again.")
      }
      if (tools::file_ext(archive) == "zip") {
        utils::unzip(archive, exdir = self$raw_folder)
      } else if (tools::file_ext(archive) == "gz") {
        tar_path <- sub("\\.gz$", "", archive)
        R.utils::gunzip(archive, tar_path, overwrite = TRUE)
        utils::untar(tar_path, exdir = self$raw_folder)
      }
    }

    cli_inform("{.cls {class(self)[[1]]}} dataset downloaded and extracted successfully.")

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
#' @inheritParams flickr8k_caption_dataset
#' @param root Character. Root directory where the dataset will be stored under `root/flickr30k`.
#'
#' @return A torch dataset of class \code{flickr30k_caption_dataset}.
#' Each element is a named list:
#' - `x`: a H x W x 3 integer array representing an RGB image.
#' - `y`: a character vector containing all five captions associated with the image.
#'
#' @rdname flickr_caption_dataset
#' @export
flickr30k_caption_dataset <- torch::dataset(
  name = "flickr30k",
  inherit = flickr8k_caption_dataset,
  archive_size = "4.1 GB",
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

    if (download)
      cli_inform("{.cls {class(self)[[1]]}} Dataset (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
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

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$images)} images across {length(self$classes)} classes.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$raw_folder, "dataset_flickr30k.json")) &&
    fs::dir_exists(file.path(self$raw_folder, "flickr30k-images"))
  },

  active = list(
    raw_folder = function() file.path(self$root, "flickr30k", "raw")
  )
)