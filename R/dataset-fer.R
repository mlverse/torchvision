#' FER-2013 Dataset
#'
#' Loads the FER-2013 facial expression dataset.
#' The dataset contains grayscale facial images (48x48) with 7 emotion labels:
#' - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
#'
#' Supports train and test splits.
#'
#' @param root Character. Root directory for dataset storage.
#' @param train Logical. If TRUE, loads training set; else loads test set.
#' @param transform Optional. A function to apply on input images.
#' @param target_transform Optional. A function to apply on labels.
#' @param download Logical. Whether to download the dataset if not found locally.
#'
#' @return A torch-style dataset object. Each item is a list with:
#' - `x`: 3D torch tensor (1x48x48)
#' - `y`: integer label (1-based index)
#'
#' @examples
#' \dontrun{
#' fer <- fer_dataset(download = TRUE)
#' first_item <- fer[1]
#' # image tensor of item 1
#' first_item$x
#' # label index of item 1
#' first_item$y
#' # label name of item 1
#' fer$get_classes[first_item$y]
#' }
#' 
#' @name fer_dataset
#' @aliases fer_dataset
#' @title FER-2013 Emotion Recognition Dataset
#' @export
fer_dataset <- dataset(
  name = "fer_dataset",

  initialize = function(root = tempdir(), train = TRUE,
                        transform = NULL, target_transform = NULL,
                        download = FALSE) {

    self$root <- root
    self$train <- train
    self$transform <- transform
    self$target_transform <- target_transform
    self$split <- if (train) "train" else "test"
    self$folder_name <- "fer2013"
    self$train_url <- "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/train.csv.zip"
    self$test_url <- "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/test.csv.zip"
    self$train_md5 <- "e6c225af03577e6dcbb1c59a71d09905"
    self$test_md5 <- "024ec789776ef0a390db67b1d7ae60a3"
    self$classes <- c("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
    self$class_to_idx <- setNames(seq_along(self$classes), self$classes)

    rlang::inform(glue::glue("Preparing FER-2013 dataset ({self$split} split)..."))

    if (download)
      self$download()

    if (!self$check_files())
      runtime_error("Dataset files missing. Use download = TRUE to fetch them.")

    data_file <- file.path(self$root, self$folder_name, glue::glue("{self$split}.csv"))
    parsed <- read.csv(data_file, stringsAsFactors = FALSE)

    self$x <- lapply(parsed$pixels, function(p) {
      img <- as.integer(strsplit(p, " ")[[1]])
      torch_tensor(img, dtype = torch_uint8())$view(c(1, 48, 48))
    })

    self$y <- as.integer(parsed$emotion) + 1L

    rlang::inform(glue::glue("FER-2013 ({self$split}) loaded successfully. Total samples: {length(self$y)}"))
  },
  .getitem = function(i) {
    x <- self$x[[i]]
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
  get_classes = function() {
    self$classes
  },
  download = function() {
    if (self$check_files())
      return()

    dir <- file.path(self$root, self$folder_name)
    fs::dir_create(dir)

    zipfile <- download_and_cache(
      url = if (self$train) self$train_url else self$test_url
    )

    expected_md5 <- if (self$train) self$train_md5 else self$test_md5
    if (tools::md5sum(zipfile) != expected_md5)
      runtime_error("MD5 checksum mismatch. File may be corrupted.")

    utils::unzip(zipfile, exdir = dir)
  },
  check_files = function() {
    file.exists(file.path(self$root, self$folder_name, glue::glue("{self$split}.csv")))
  }
)
