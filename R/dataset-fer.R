#' FER2013 Dataset
#'
#' Loads the FER2013 dataset for facial expression recognition, consisting of grayscale 48x48 images.
#' The dataset contains facial images categorized into seven emotion classes:
#' - 0: Angry
#' - 1: Disgust
#' - 2: Fear
#' - 3: Happy
#' - 4: Sad
#' - 5: Surprise
#' - 6: Neutral
#'
#' @param root Character. Root directory where the `fer2013` folder exists or will be saved to if `download = TRUE`.
#' @param train Logical. If `TRUE`, loads the training split; if `FALSE`, loads the test split. Default is `TRUE`.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#' @param transform Optional function to transform input images.
#' @param target_transform Optional function to transform target labels.
#'
#' @return A FER2013 dataset object.
#'
#' @examples
#' \dontrun{
#' fer <- fer_dataset(train = TRUE, download = TRUE)
#' first_item <- fer[1]
#' # image in item 1
#' first_item$x
#' # label of item 1
#' first_item$y
#' # label name
#' first_item$class_name
#' }
#'
#' @name fer_dataset
#' @aliases fer_dataset
#' @title FER2013 dataset
#' @export
fer_dataset <- dataset(
  name = "fer_dataset",
  train_url = "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/train.csv.zip",
  test_url = "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/test.csv.zip",
  train_md5 = "e6c225af03577e6dcbb1c59a71d09905",
  test_md5 = "024ec789776ef0a390db67b1d7ae60a3",
  folder_name = "fer2013",
  classes = c("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"),
  initialize = function(root = tempdir(), train = TRUE, transform = NULL, target_transform = NULL, download = FALSE) {
    self$root <- root
    self$train <- train
    self$transform <- transform
    self$target_transform <- target_transform
    self$split <- if (train) "train" else "test"

    rlang::inform(glue::glue("Downloading and processing FER-2013 dataset ({self$split} split)..."))

    if (download)
      self$download()

    check <- self$check_files()
    if (!check)
      runtime_error("Files not found or corrupted. Use download = TRUE")

    data_file <- fs::path(self$root, self$folder_name, paste0(self$split, ".csv"))
    lines <- readLines(data_file)
    header <- strsplit(lines[1], ",")[[1]]
    parsed <- read.csv(data_file, stringsAsFactors = FALSE)

    self$x <- lapply(parsed$pixels, function(p) {
      img <- as.integer(strsplit(p, " ")[[1]])
      torch_tensor(img, dtype = torch_uint8())$view(c(1, 48, 48))
    })

    self$y <- as.integer(parsed$emotion) + 1L

    rlang::inform(glue::glue("FER-2013 dataset ({self$split} split) Processed Successfully !"))
  },
  .getitem = function(i) {
    x <- self$x[[i]]
    y <- self$y[i]

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y, class_name = self$classes[y])
  },
  .length = function() {
    length(self$y)
  },
  download = function() {
    if (self$check_files())
      return()

    dir <- fs::path(self$root, self$folder_name)
    fs::dir_create(dir)

    if (self$train) {
      zipfile <- download_and_cache(self$train_url)
      if (tools::md5sum(zipfile) != self$train_md5)
        runtime_error(paste("Corrupt file!", basename(zipfile), "does not match expected checksum."))
    } else {
      zipfile <- download_and_cache(self$test_url)
      if (tools::md5sum(zipfile) != self$test_md5)
        runtime_error(paste("Corrupt file!", basename(zipfile), "does not match expected checksum."))
    }

    utils::unzip(zipfile, exdir = dir)
  },
  check_files = function() {
    file <- fs::path(self$root, self$folder_name, paste0(if (self$train) "train" else "test", ".csv"))
    fs::file_exists(file)
  }
)
