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
  
  initialize = function(root,
                        split = "train",
                        download = FALSE,
                        transform = NULL,
                        target_transform = NULL) {
    self$root <- normalizePath(root, mustWork = FALSE)
    self$split <- split
    self$transform <- transform
    self$target_transform <- target_transform
    
    if (!split %in% c("train", "val", "test")) {
      stop("Invalid split. Must be one of 'train', 'val', or 'test'.")
    }
    
    self$zip_file <- file.path(self$root, "EuroSAT.zip")
    self$images_dir <- file.path(self$root, "images")
    self$split_file <- file.path(self$root, sprintf("eurosat-%s.txt", split))
    
    if (download) {
      self$download()
    }
    
    if (!file.exists(self$split_file)) {
      stop(sprintf("Split file not found for split='%s'.", split))
    }
    if (!dir.exists(self$images_dir)) {
      stop("Images directory not found: ", self$images_dir)
    }
    
    # Suppress the "incomplete final line" warning:
    self$data <- readLines(self$split_file, warn = FALSE)
  },
  
  download = function() {
    if (!file.exists(self$zip_file) || file.size(self$zip_file) == 0) {
      dir.create(self$root, recursive = TRUE, showWarnings = FALSE)
      zip_url <- "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip?download=true"
      message("Downloading EuroSAT ZIP...")
      utils::download.file(url = zip_url, destfile = self$zip_file, mode = "wb")
      message("EuroSAT ZIP downloaded.")
    }
    
    if (!dir.exists(self$images_dir)) {
      message("Extracting EuroSAT ZIP...")
      utils::unzip(self$zip_file, exdir = self$images_dir)
      message("Extraction complete.")
    }
    
    txt_url <- sprintf(
      "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/eurosat-%s.txt?download=true",
      self$split
    )
    message("Downloading split text file: ", txt_url)
    utils::download.file(url = txt_url, destfile = self$split_file, mode = "wb")
    if (file.size(self$split_file) == 0) {
      stop("Downloaded split file is empty: ", self$split_file)
    }
  },
  
  .getitem = function(index) {
    filename <- self$data[index]
    label <- sub("_.*", "", filename)
    
    image_path <- file.path(self$images_dir, "2750", label, filename)
    if (!file.exists(image_path)) {
      stop("Image file not found: ", image_path)
    }
    
    img_array <- jpeg::readJPEG(image_path)
    
    if (!is.null(self$transform)) {
      img_array <- self$transform(img_array)
    }
    if (!is.null(self$target_transform)) {
      label <- self$target_transform(label)
    }
    
    list(x = img_array, y = label)
  },
  
  .length = function() {
    length(self$data)
  }
)
