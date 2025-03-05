#' Caltech-101 Dataset Loader (from Local Storage)
#'
#' Loads the Caltech-101 dataset stored in the 'data/' folder.
#'
#' @param root Character. The root directory where the dataset is stored.
#' @param split Character. One of `train` or `test`.
#' @param transform Function. Optional image transformation.
#' @param target_transform Function. Optional label transformation.
#'
#' @return An R6 dataset object that inherits from `torch::dataset`.
#'
#' @export
# Required Libraries
library(torch)        # Torch for deep learning
library(torchvision)  # Image transformations
library(arrow)        # For reading Parquet files
library(jpeg)         # For reading JPEG images
library(base64enc)    # For decoding base64 images

# Define the Dataset Class
caltech101_dataset <- dataset(
  name = "Caltech101",

  initialize = function(root, split = "train") {
    self$root <- root
    self$split <- split
    self$data_file <- file.path(root, "data", paste0(split, ".parquet"))

    # Ensure dataset file exists
    if (!file.exists(self$data_file)) {
      stop("Dataset file not found: ", self$data_file)
    }

    # Read Parquet File
    self$data <- arrow::read_parquet(self$data_file)

    # Extract classes
    if ("label" %in% names(self$data)) {
      self$classes <- unique(self$data$label)
      self$class_to_idx <- setNames(seq_along(self$classes), self$classes)
    } else {
      stop("Label column missing in dataset.")
    }
  },

  .getitem = function(index) {
    example <- self$data[index, ]

    # Ensure the image column exists
    if (!"image" %in% names(example) && !"filename" %in% names(example)) {
      stop("Dataset must contain either 'image' or 'filename' column.")
    }

    # Load Image
    if ("image" %in% names(example)) {
      # Case 1: Image is stored as base64 string
      if (is.character(example$image)) {
        img_bytes <- base64enc::base64decode(example$image)
        img_array <- jpeg::readJPEG(img_bytes)
      } else {
        stop("Unknown image format in dataset.")
      }
    } else {
      # Case 2: Image is stored as a filename (load from disk)
      image_path <- file.path(self$root, "images", example$filename)

      if (!file.exists(image_path)) {
        stop("Image file not found: ", image_path)
      }

      img_array <- jpeg::readJPEG(image_path)
    }

    # Convert label to tensor
    label_idx <- torch::torch_tensor(as.integer(self$class_to_idx[[example$label]]), dtype = torch_long())$squeeze()

    list(x = img_array, y = label_idx)
  },

  .length = function() {
    nrow(self$data)
  }
)
