#' FER2013 Dataset
#'
#' Loads the FER2013 dataset, a facial expression recognition dataset containing
#' grayscale 48x48 pixel images labeled with one of 7 emotions.
#'
#' The dataset consists of two splits:
#' - "train": training set
#' - "test": test set
#'
#' @param root Character. Root directory for dataset storage (default folder: `root/fer2013/processed/`).
#' @param train Logical. If `TRUE`, loads the training split; otherwise, loads the test split. Default is `TRUE`.
#' @param download Logical. Whether to download the dataset if it is not found locally. Default is `FALSE`.
#' @param transform Optional function to transform input image tensors.
#' @param target_transform Optional function to transform target labels.
#'
#' @return A FER2013 dataset object.
#'
#' @examples
#' \dontrun{
#' fer <- fer_dataset(train = TRUE, download = TRUE)
#' first_item <- fer[1]
#' # image tensor of the first sample
#' first_item$x
#' # label tensor of the first sample
#' first_item$y
#' }
#'
#' @name fer_dataset
#' @aliases fer_dataset
#' @title FER2013 dataset
#' @export
fer_dataset <- dataset(
  name = "fer2013",
  training_file = "train.rds",
  test_file = "test.rds",
  train_url = "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/train.csv.zip",
  test_url = "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/test.csv.zip",
  train_md5 = "e6c225af03577e6dcbb1c59a71d09905",
  test_md5 = "024ec789776ef0a390db67b1d7ae60a3",

  initialize = function(root = rappdirs::user_cache_dir("torch"), train = TRUE, transform = NULL, target_transform = NULL, download = FALSE) {
    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. Use download = TRUE to download it.")

    file <- if (self$train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, file))

    self$data_array <- data$images  
    cat("Shape of loaded images:", paste(dim(data$images), collapse = " x "), "\n")

    self$targets <- torch_tensor(data$labels + 1L, dtype = torch_long())  
  },

  download = function() {
    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    if (self$train) {
      zip_url <- self$train_url
      zip_md5 <- self$train_md5
      split_name <- "train"
    } else {
      zip_url <- self$test_url
      zip_md5 <- self$test_md5
      split_name <- "test"
    }

    zip_name <- paste0(split_name, ".csv.zip")
    csv_name <- paste0(split_name, ".csv")
    zip_path <- file.path(self$raw_folder, zip_name)
    csv_path <- file.path(self$raw_folder, csv_name)

    p <- download_and_cache(zip_url, prefix = class(self)[1])
    fs::file_copy(p, zip_path, overwrite = TRUE)

    if (tools::md5sum(zip_path) != zip_md5)
      runtime_error(paste("MD5 mismatch for", zip_path))

    utils::unzip(zip_path, exdir = self$raw_folder)
    rlang::inform(glue::glue("Processing FER2013 {split_name} dataset..."))

    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    n <- nrow(df)
    images <- array(0L, dim = c(n, 1, 48, 48))

    for (i in seq_len(n)) {
      pixels <- as.integer(strsplit(df$pixels[i], " ")[[1]])
      images[i, 1, , ] <- matrix(pixels, nrow = 48, byrow = TRUE)
    }

    labels <- as.integer(df$emotion)

    saveRDS(list(images = images, labels = labels), file.path(self$processed_folder, paste0(split_name, ".rds")))
    rlang::inform(glue::glue("FER2013 {split_name} split downloaded and processed."))
  },

  check_exists = function() {
    file <- if (self$train) self$training_file else self$test_file
    fs::file_exists(file.path(self$processed_folder, file))
  },

  .getitem = function(index) {
    img_array <- self$data_array[index, , , , drop = FALSE]
    img <- torch_tensor(img_array, dtype = torch_float())

    if (!is.null(self$transform))
      img <- self$transform(img)

    target <- self$targets[index]
    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img, y = target)
  },

  .length = function() {
    dim(self$data_array)[1]
  },

  active = list(
    raw_folder = function() {
      file.path(self$root_path, "fer2013", "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, "fer2013", "processed")
    }
  )
)
