#' FER-2013 Facial Expression Dataset
#'
#' Loads the FER-2013 dataset for facial expression recognition. The dataset contains grayscale images
#' (48x48) of human faces, each labeled with one of seven emotion categories:
#' `"Angry"`, `"Disgust"`, `"Fear"`, `"Happy"`, `"Sad"`, `"Surprise"`, and `"Neutral"`.
#'
#' The dataset is split into:
#' - `"Train"`: training images labeled as `"Training"` in the original CSV.
#' - `"Test"`: includes both `"PublicTest"` and `"PrivateTest"` entries.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/fer2013`.
#' @param train Logical. Whether to load the training split (`TRUE`) or test split (`FALSE`). Default is `TRUE`.
#' @param transform Optional function to transform input images after loading. Each image is 48x48 by default.
#' @param target_transform Optional function to transform emotion labels. Default is `NULL`.
#' @param download Logical. Whether to download the dataset archive if not found locally. Default is `FALSE`.
#'
#' @return A torch dataset of class \code{fer_dataset}.
#' Each element is a named list:
#' - `x`: a 48x48 grayscale array
#' - `y`: a character emotion label from the 7-class list
#'
#' @examples
#' \dontrun{
#' fer <- fer_dataset(train = TRUE, download = TRUE)
#' first_item <- fer[1]
#' first_item$x  # 48x48 grayscale array
#' first_item$y  # "Happy", "Sad", etc.
#' }
#'
#' @name fer_dataset
#' @aliases fer_dataset
#' @title FER-2013 Facial Expression Dataset
#' @export
fer_dataset <- dataset(
  name = "fer_dataset",

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$root <- root
    self$train <- train
    self$transform <- transform
    self$target_transform <- target_transform
    self$split <- if (train) "Train" else "Test"

    self$folder_name <- "fer2013"
    self$url <- "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/fer2013.tar.gz"
    self$md5 <- "ca95d94fe42f6ce65aaae694d18c628a"

    self$classes <- c("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
    self$class_to_idx <- setNames(seq_along(self$classes), self$classes)

    cli::cli_inform("Preparing FER-2013 dataset ({self$split})...")

    if (download) {
      self$download()
    }

    if (!self$check_files()) {
      runtime_error("Dataset files missing. Use download = TRUE to fetch them.")
    }

    csv_file <- file.path(self$root, self$folder_name, "fer2013.csv")
    parsed <- read.csv(csv_file, stringsAsFactors = FALSE)

    if (self$train) {
      parsed <- parsed[parsed$Usage == "Training", ]
    } else {
      parsed <- parsed[parsed$Usage %in% c("PublicTest", "PrivateTest"), ]
    }

    rlang::inform("Parsing image data into 48x48 grayscale arrays...")
    self$x <- lapply(strsplit(parsed$pixels, " "), as.integer)

    self$y <- self$classes[parsed$emotion + 1L]

    file_size <- fs::file_info(csv_file)$size
    readable <- fs::fs_bytes(file_size)

    cli::cli_inform(
      "FER-2013 ({self$split}) loaded: {length(self$x)} images, 48x48 grayscale, {length(self$classes)} classes."
    )
  },

  .getitem = function(i) {
    raw_vec <- self$x[[i]]
    x <- matrix(raw_vec, nrow = 48, ncol = 48, byrow = TRUE)
    x <- array(x, dim = c(48, 48))

    y <- self$y[i]

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
  },

  .length = function() {
    length(self$y)
  },

  download = function() {
    if (self$check_files()) {
      rlang::inform("FER-2013 already exists. Skipping download.")
      return()
    }

    dest_dir <- file.path(self$root, self$folder_name)
    fs::dir_create(dest_dir)

    cli::cli_inform("Downloading FER-2013 dataset... (Size: ~100MB)")

    archive <- download_and_cache(self$url)

    if (!tools::md5sum(archive) == self$md5)
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")

    untar(archive, exdir = self$root)
  },

  check_files = function() {
    file.exists(file.path(self$root, self$folder_name, "fer2013.csv"))
  }
)
