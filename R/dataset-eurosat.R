#' EuroSAT Dataset Loader (via Hugging Face API)
#'
#' Downloads and loads the EuroSAT dataset using Hugging Face API.
#' The dataset consists of Sentinel-2 satellite images organized into 10 classes.
#'
#' @param root Character. The root directory where the dataset will be stored.
#' @param split Character. One of `train`, `validation`, or `test`.
#' @param download Logical. If `TRUE`, downloads the dataset rows from the API if not already present.
#' @param transform Function. Optional transformation to be applied to the images.
#' @param target_transform Function. Optional transformation to be applied to the labels.
#'
#' @return An R6 dataset object that inherits from `torch::dataset`.
#'
#' @examples
#' \dontrun{
#' # Initialize the dataset
#' ds <- eurosat_dataset(root = "./data/eurosat", split = "train", download = TRUE)
#'
#' # Access the first sample
#' sample <- ds[1]
#' print(sample$x) # Image
#' print(sample$y) # Label
#' }
#' @export
eurosat_dataset <- torch::dataset(
  name = "eurosat",
  
  initialize = function(root, split = "train", download = FALSE, transform = NULL, target_transform = NULL) {
    self$root <- normalizePath(root, mustWork = FALSE)
    self$split <- split
    self$transform <- transform
    self$target_transform <- target_transform
    self$data_file <- file.path(self$root, paste0("eurosat_", split, ".json"))
    
    if (!split %in% c("train", "validation", "test")) {
      stop("Invalid split. Choose one of 'train', 'validation', or 'test'.")
    }
    
    if (download) {
      self$download()
    }
    
    if (!file.exists(self$data_file)) {
      stop("Dataset not found. Use `download = TRUE` to download it.")
    }
    
    data <- tryCatch(
      jsonlite::fromJSON(self$data_file),
      error = function(e) stop("Failed to parse dataset JSON file.")
    )
    
    self$data <- data$rows
  },
  
  download = function() {
    if (file.exists(self$data_file) && file.size(self$data_file) > 0) {
      message("Dataset already exists. Skipping download.")
      return(invisible(NULL))
    }
    
    dir.create(self$root, recursive = TRUE, showWarnings = FALSE)
    split_url <- sprintf(
      "https://datasets-server.huggingface.co/rows?dataset=torchgeo%%2Feurosat&config=default&split=%s&offset=0&length=100",
      self$split
    )
    
    tryCatch({
      utils::download.file(
        split_url,
        destfile = self$data_file,
        mode = "wb"
      )
      if (file.size(self$data_file) == 0) stop("Downloaded file is empty.")
    }, error = function(e) {
      stop("Failed to download dataset: ", conditionMessage(e))
    })
  },
  
  .getitem = function(index) {
    row <- self$data[index, ]
    
    img <- tryCatch(
      base64enc::base64decode(row$image),
      error = function(e) {
        warning(sprintf("Skipping row %d due to image decoding failure: %s", index, conditionMessage(e)))
        NULL
      }
    )
    
    if (is.null(img)) {
      return(NULL)
    }
    
    label <- row$label
    
    if (!is.null(self$transform)) {
      img <- self$transform(img)
    }
    if (!is.null(self$target_transform)) {
      label <- self$target_transform(label)
    }
    
    list(x = img, y = label)
  },
  
  .length = function() {
    nrow(self$data)
  }
)
