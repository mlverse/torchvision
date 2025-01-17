#' EuroSAT Dataset Loader
#'
#' Downloads and loads the RGB version of the EuroSAT dataset.
#' The dataset consists of Sentinel-2 satellite images organized into 10 classes.
#'
#' @param root Character. The root directory where the dataset will be stored.
#'   If `NULL`, a temporary directory is used.
#' @param download Logical. If `TRUE`, downloads the dataset if not already present.
#' @param transform Function. Optional transformation to be applied to the images.
#' @param target_transform Function. Optional transformation to be applied to the labels.
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
eurosat_dataset <- torch::dataset(
  name = "eurosat",
  
  resources = list(
    url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip",
    md5 = "c8fa014336c82ac7804f0398fcb19387"
  ),
  
  initialize = function(root = NULL, download = FALSE, transform = NULL, target_transform = NULL) {
    self$root <- normalizePath(if (is.null(root)) tempfile(fileext = "/") else root, mustWork = FALSE)
    self$base_folder <- file.path(self$root, "eurosat")
    self$data_folder <- file.path(self$base_folder, "2750")
    self$transform <- transform
    self$target_transform <- target_transform
    
    if (download) {
      self$download()
    }
    
    if (!self$check_exists()) {
      stop("Dataset not found. Use download = TRUE to download it.")
    }
    
    self$image_paths <- list.files(
      path = self$data_folder,
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
  
  .length = function() {
    length(self$image_paths)
  },
  
  check_exists = function() {
    dir.exists(self$data_folder)
  },
  
  download = function() {
    if (self$check_exists()) {
      message("Dataset already exists. Skipping download.")
      return(invisible(NULL))
    }
    
    dir.create(self$base_folder, recursive = TRUE, showWarnings = FALSE)
    zip_file <- file.path(self$base_folder, "EuroSAT.zip")
    
    utils::download.file(
      self$resources$url,
      destfile = zip_file,
      mode = "wb",
      method = "curl",
      extra = "--insecure"
    )
    
    md5_actual <- tools::md5sum(zip_file)
    if (md5_actual != self$resources$md5) {
      stop("Downloaded file has an invalid MD5 checksum. Please try again.")
    }
    
    utils::unzip(zip_file, exdir = self$base_folder)
  }
)
