# caltech101_dataset_loader.R

#' Caltech-101 Dataset Loader (via Hugging Face API)
#'
#' Downloads and loads the Caltech-101 dataset using the Hugging Face API.
#' The dataset consists of 101 object categories.
#'
#' @param root Character. The root directory where the dataset will be stored.
#' @param download Logical. If `TRUE`, downloads the dataset if not already present.
#' @param transform Function. Optional transformation to be applied to the images.
#' @param target_transform Function. Optional transformation to be applied to the labels.
#'
#' @return An R6 dataset object that inherits from `torch::dataset`.
#'
#' @examples
#' \dontrun{
#'   ds <- caltech101_dataset(root = "./data/caltech101", download = TRUE)
#'   sample <- ds[1]
#'   print(sample$x)  # Image array
#'   print(sample$y)  # Label tensor
#' }
#' @export
caltech101_dataset <- torch::dataset(
  name = "caltech101",

  initialize = function(root,
                        download = FALSE,
                        transform = NULL,
                        target_transform = NULL) {
    self$root <- normalizePath(root, mustWork = FALSE)
    self$transform <- transform
    self$target_transform <- target_transform

    self$file <- file.path(self$root, "train-00000-of-00001.parquet")

    if (download) {
      self$download()
    }

    if (!file.exists(self$file)) {
      stop("Caltech-101 Parquet file not found. Set download = TRUE to download it.")
    }

    self$data <- arrow::read_parquet(self$file)

    self$load_meta()
  },

  load_meta = function() {
    if (!"label" %in% names(self$data)) {
      stop("The loaded data does not have a 'label' column.")
    }

    self$classes <- sort(unique(self$data$label))
    self$class_to_idx <- setNames(seq_along(self$classes) - 1, self$classes)
  },

  download = function() {
    if (!dir.exists(self$root)) {
      dir.create(self$root, recursive = TRUE, showWarnings = FALSE)
    }

    if (!file.exists(self$file) || file.size(self$file) == 0) {
      file_url <- "https://huggingface.co/datasets/bitmind/caltech-101/resolve/main/data/train-00000-of-00001.parquet"
      message("Downloading Caltech-101 dataset (158 MB approx.) ...")
      options(timeout = 600)
      utils::download.file(url = file_url, destfile = self$file, mode = "wb")
      message("Download complete.")
    }
  },

  .getitem = function(index) {
    row <- self$data[index, ]

   img_data <- row$image
    label_val <- row$label

    if (is.raw(img_data)) {
      con <- rawConnection(img_data)
      img_array <- jpeg::readJPEG(con)
      close(con)
    } else if (is.character(img_data) && file.exists(img_data)) {
      # Alternatively, if it is a file path, read directly.
      img_array <- jpeg::readJPEG(img_data)
    } else {
      stop("The image data type is not supported. It must be raw JPEG bytes or a valid file path.")
    }

    if (!is.null(self$transform)) {
      img_array <- self$transform(img_array)
    }

    if (!as.character(label_val) %in% names(self$class_to_idx)) {
      stop("Label not found in class_to_idx mapping: ", label_val)
    }
    label_idx <- torch::torch_tensor(
      as.integer(self$class_to_idx[[as.character(label_val)]]),
      dtype = torch_long()
    )$squeeze()

    if (!is.null(self$target_transform)) {
      label_idx <- self$target_transform(label_idx)
    }

    list(x = img_array, y = label_idx)
  },

  .length = function() {
    nrow(self$data)
  }
)
