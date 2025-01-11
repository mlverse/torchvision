#' EuroSAT Dataset Loader
#'
#' Downloads and loads the RGB version of the EuroSAT dataset.
#' The dataset consists of Sentinel-2 satellite images organized into 10 classes.
#'
#' @param root Character. The root directory where the dataset will be stored.
#'   If `NULL`, a temporary directory is used.
#' @param download Logical. If `TRUE`, downloads the dataset if not already present.
#' @param transform Function. Optional transformation to be applied on the images.
#' @param target_transform Function. Optional transformation to be applied on the labels.
#'
#' @return An R6 dataset object that inherits from `torch::dataset`.
#'
#' @examples
#' \dontrun{
#' # Use a temporary directory and download the dataset
#' ds <- eurosat_dataset(download = TRUE)
#'
#' # Specify a custom root directory
#' ds <- eurosat_dataset(root = "./data/eurosat", download = TRUE)
#'
#' # Access the first sample
#' sample <- ds[1]
#' print(sample$x) # Image
#' print(sample$y) # Label
#' }
#'
#' @export
eurosat_dataset <- dataset(
  name = "eurosat",
  
  resources = list(
    url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
  ),
  
  initialize = function(root = NULL, download = FALSE, transform = NULL, target_transform = NULL) {
    
    self$root <- if (is.null(root)) tempfile(fileext = "/") else root
    self$transform <- transform
    self$target_transform <- target_transform
    
    if (!dir.exists(self$root)) {
      dir.create(self$root, recursive = TRUE, showWarnings = FALSE)
    }
    
    if (download) {
      self$download()
    }
    
    if (!self$check_exists()) {
      rlang::abort("Files not found. Use download=TRUE to retrieve them.")
    }
    
    self$image_paths <- list.files(
      path = file.path(self$root, "2750"),
      pattern = "\\.jpg$",
      recursive = TRUE,
      full.names = TRUE
    )
    self$labels <- sapply(
      basename(dirname(self$image_paths)), 
      function(class_name) match(class_name, c("AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                                               "Industrial", "Pasture", "PermanentCrop", "Residential",
                                               "River", "SeaLake"))
    )
  },
  
  #' Download the EuroSAT dataset
  #'
  #' Downloads the EuroSAT dataset ZIP file to the specified root directory
  #' and extracts it if necessary. Skips download if the dataset is already
  #' present and extracted.
  #'
  #' @return Invisible `NULL`.
  download = function() {
    zip_file <- file.path(self$root, "EuroSAT.zip")
    extracted_dir <- file.path(self$root, "2750")
    
    if (dir.exists(extracted_dir)) {
      message("Dataset already extracted. Skipping download.")
      return(invisible(NULL))
    }
    
    if (file.exists(zip_file)) {
      message("File already downloaded. Skipping re-download.")
    } else {
      utils::download.file(self$resources$url, destfile = zip_file, mode = "wb")
    }
    
    utils::unzip(zip_file, exdir = self$root)
  },
  
  #' Check if the EuroSAT dataset exists
  #'
  #' Verifies whether the EuroSAT dataset has already been downloaded
  #' and extracted in the specified root directory.
  #'
  #' @return Logical. `TRUE` if the dataset exists, `FALSE` otherwise.
  check_exists = function() {
    dir.exists(file.path(self$root, "2750"))
  },
  
  #' Get a sample from the dataset
  #'
  #' Retrieves a single sample (image and label) from the dataset.
  #'
  #' @param i Integer. Index of the sample to retrieve.
  #'
  #' @return A list containing:
  #'   - `x`: The image as an array or tensor.
  #'   - `y`: The label as an integer.
  .getitem = function(i) {
    img_path <- self$image_paths[i]
    img <- jpeg::readJPEG(img_path)
    
    label <- self$labels[i]
    
    if (!is.null(self$transform)) {
      img <- self$transform(img)
    }
    
    if (!is.null(self$target_transform)) {
      label <- self$target_transform(label)
    }
    
    list(x = img, y = label)
  },
  
  #' Get the length of the dataset
  #'
  #' @return Integer. The number of samples in the dataset.
  .length = function() {
    length(self$image_paths)
  }
)
