#' EuroSAT Dataset
#'
#' Downloads and prepare the EuroSAT dataset from Hugging Face datasets.
#' The dataset consists of Land Use and Land Cover Classification with Sentinel-2
#'  satellite images. Images are openly and freely made available by the Earth
#'  observation program Copernicus. Images are organized into 10 classes.
#'
#' @name eurosat_datasets
#' @details
#' `eurosat_dataset()` provides a total of 27,000 RGB labeled images.
#'
#' @inheritParams mnist_dataset
#' @param root (Optional) Character. The root directory where the dataset will be stored.
#'  if empty, will use the default `rappdirs::user_cache_dir("torch")`.
#' @param split Character. Must be one of `train`, `val`, or `test`.
#'
#' @return A `torch::dataset` object. Each item is a list with:
#' * `x`: a 64x64 image tensor with 3 (RGB) or 13 (all bands) channels
#' * `y`: the class label
#'
#' @examples
#' \dontrun{
#' # Initialize the dataset
#' ds <- eurosat100_dataset(split = "train", download = TRUE)
#'
#' # Access the first item
#' head <- ds[1]
#' print(head$x) # Image
#' print(head$y) # Label
#' }
#' @family datasets
#' @export
eurosat_dataset <- torch::dataset(
  name = "eurosat",
  archive_url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip?download=true",
  archive_md5 = "c8fa014336c82ac7804f0398fcb19387",
  split_url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/eurosat-{split}.txt?download=true",

  initialize = function(root,
                        split = "train",
                        download = FALSE,
                        transform = NULL,
                        target_transform = NULL) {
    self$root <- normalizePath(root, mustWork = FALSE)
    self$split <- match.arg(split, c("train", "val", "test"))
    self$transform <- transform
    self$target_transform <- target_transform

    cli_inform("{.cls {class(self)[[1]]}} Dataset will be downloaded and processed if not already available.")

    self$split_url <- glue::glue(self$split_url)
    self$images_dir <- file.path(self$root, class(self)[1], "images")
    self$split_file <- file.path(self$root, fs::path_ext_remove(basename(self$split_url)))

    if (download) {
      self$download()
    }
    self$img_files <- list.files(self$images_dir, pattern = "\\.(tif|jpg)", recursive = TRUE, full.names = TRUE)

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    self$data <- suppressWarnings(readLines(self$split_file))
    self$load_meta()

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$data)} images across {length(self$classes)} classes.")
  },

  load_meta = function() {
    self$classes <- unique(sub("_.*", "", self$data))
    self$class_to_idx <- setNames(seq_along(self$classes) - 1, self$classes)
  },

  download = function() {
    if (self$check_exists())
      return(NULL)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    fs::dir_create(self$root, recurse = TRUE, showWarnings = FALSE)

    # Download and extract the dataset archive
    archive <- download_and_cache(self$archive_url, prefix = class(self)[1])
    if (!tools::md5sum(archive) == self$archive_md5)
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")


    if (!dir.exists(self$images_dir)) {
      cli_inform("{.cls {class(self)[[1]]}} Extracting archive...")
      utils::unzip(archive, exdir = self$images_dir)
      cli_inform("{.cls {class(self)[[1]]}} Extraction complete.")
    }

    # Download the split-specific text file
    cli_inform("{.cls {class(self)[[1]]}} Downloading split text file: {self$split_url}")
    p <- download_and_cache(self$split_url, prefix = class(self)[1])
    fs::file_copy(p, self$split_file, overwrite = TRUE)
    if (file.size(self$split_file) == 0) {
      runtime_error("Downloaded split file `{self$split_file}` is empty.")
    }
    cli_inform("{.cls {class(self)[[1]]}} dataset downloaded and extracted successfully.")
  },
  check_exists = function() {
    fs::file_exists(self$split_file) &&
      fs::file_exists(self$images_dir) &&
      length(self$img_files) > 1
  },

  .getitem = function(index) {
    filename <- fs::path_ext_remove(self$data[index])

    image_path <- grep(paste0(filename,"."), self$img_files, value = TRUE, fixed = TRUE)
    if (length(image_path) != 1) {
      value_error("Image file `{filename}` not found.")
    }
    image_ext <- fs::path_ext(image_path)
    if (image_ext == "jpg") {
      img_array <- jpeg::readJPEG(image_path)
    } else {
      img_array <- suppressWarnings(tiff::readTIFF(image_path)) %>% aperm(c(3,1,2))
    }

    if (!is.null(self$transform)) {
      img_array <- self$transform(img_array)
    }

    # Ensure label exists in class_to_idx
    label <- sub("_.*", "", filename)  # Ensure label is a character string
    if (!label %in% names(self$class_to_idx)) {
      value_error("Label `{label}` not found in class_to_idx." )
    }

    # Convert label index to torch tensor with dtype = torch_long()
    label_idx <- torch::torch_tensor(
        as.integer(self$class_to_idx[[label]]), dtype = torch_long()
      )$squeeze()

    list(x = img_array, y = label_idx)
  },

  .length = function() {
    length(self$data)
  }
)


#' EuroSAT All Bands dataset
#'
#' Downloads and prepares the EuroSAT dataset with 13 spectral bands.
#'
#' @rdname eurosat_datasets
#' @details `eurosat_all_bands_dataset()` provides a total of 27,000 labeled images with 13 spectral channel bands.
#'
#' @export
eurosat_all_bands_dataset <- torch::dataset(
  name = "eurosat_all_bands",
  inherit = eurosat_dataset,
  archive_url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip?download=true",
  archive_md5 = "5ac12b3b2557aa56e1826e981e8e200e",
  split_url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/eurosat-{split}.txt?download=true"
)




#' EuroSAT-100 dataset
#'
#' A subset of 100 images with 13 spectral bands useful for workshops and demos.
#'
#' @rdname eurosat_datasets
#' @details `eurosat100_dataset()` provides a subset of 100 labeled images with 13 spectral channel bands.
#'
#' @export
eurosat100_dataset <- torch::dataset(
  name = "eurosat100",
  inherit = eurosat_dataset,
  archive_url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT100.zip?download=true",
  archive_md5 = "c21c649ba747e86eda813407ef17d596",
  split_url = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/eurosat-100-{split}.txt?download=true"
)


