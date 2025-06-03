#' MNIST dataset
#'
#' Prepares the MNIST dataset and optionally downloads it.
#'
#' @param root (string): Root directory of dataset where
#'   `MNIST/processed/training.pt` and  `MNIST/processed/test.pt` exist.
#' @param train (bool, optional): If True, creates dataset from
#'   `training.pt`, otherwise from `test.pt`.
#' @param download (bool, optional): If true, downloads the dataset from the
#'   internet and puts it in root directory. If dataset is already downloaded,
#'   it is not downloaded again.
#' @param transform (callable, optional): A function/transform that  takes in an
#'   PIL image and returns a transformed version. E.g,
#'   [transform_random_crop()].
#' @param target_transform (callable, optional): A function/transform that takes
#'   in the target and transforms it.
#'
#' @export
mnist_dataset <- dataset(
  name = "mnist",
  resources = list(
    c("https://torch-cdn.mlverse.org/datasets/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    c("https://torch-cdn.mlverse.org/datasets/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    c("https://torch-cdn.mlverse.org/datasets/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    c("https://torch-cdn.mlverse.org/datasets/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
  ),
  training_file = 'training.rds',
  test_file = 'test.rds',
  classes = c('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
             '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine'),
  initialize = function(root, train = TRUE, transform = NULL, target_transform = NULL,
                        download = FALSE) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform

    self$train <- train

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    if (self$train)
      data_file <- self$training_file
    else
      data_file <- self$test_file

    data <- readRDS(file.path(self$processed_folder, data_file))
    self$data <- data[[1]]
    self$targets <- data[[2]] + 1L
  },
  download = function() {

    if (self$check_exists())
      return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      filename <- tail(strsplit(r[1], "/")[[1]], 1)
      destpath <- file.path(self$raw_folder, filename)

      p <- download_and_cache(r[1], prefix = class(self)[1])
      fs::file_copy(p, destpath)

      if (!tools::md5sum(destpath) == r[2])
        runtime_error("MD5 sums are not identical for file: {r[1]}.")

    }

    rlang::inform("Processing...")

    training_set <- list(
      read_sn3_pascalvincent(file.path(self$raw_folder, 'train-images-idx3-ubyte.gz')),
      read_sn3_pascalvincent(file.path(self$raw_folder, 'train-labels-idx1-ubyte.gz'))
    )

    test_set <- list(
      read_sn3_pascalvincent(file.path(self$raw_folder, 't10k-images-idx3-ubyte.gz')),
      read_sn3_pascalvincent(file.path(self$raw_folder, 't10k-labels-idx1-ubyte.gz'))
    )

    saveRDS(training_set, file.path(self$processed_folder, self$training_file))
    saveRDS(test_set, file.path(self$processed_folder, self$test_file))

    rlang::inform("Done!")

  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img <- self$data[index, ,]
    target <- self$targets[index]

    if (!is.null(self$transform))
      img <- self$transform(img)

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img, y = target)
  },
  .length = function() {
    dim(self$data)[1]
  },
  active = list(
    raw_folder = function() {
      file.path(self$root_path, "mnist", "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, "mnist", "processed")
    }
  )
)

#' Kuzushiji-MNIST
#'
#' Prepares the [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) dataset
#'   and optionally downloads it.
#'
#' @param root (string): Root directory of dataset where
#'   `KMNIST/processed/training.pt` and  `KMNIST/processed/test.pt` exist.
#' @param train (bool, optional): If TRUE, creates dataset from `training.pt`,
#'   otherwise from `test.pt`.
#' @param download (bool, optional): If true, downloads the dataset from the
#'   internet and puts it in root directory. If dataset is already downloaded,
#'   it is not downloaded again.
#' @param transform (callable, optional): A function/transform that  takes in an
#'   PIL image and returns a transformed version. E.g, [transform_random_crop()].
#' @param target_transform (callable, optional): A function/transform that takes
#'   in the target and transforms it.
#'
#' @export
kmnist_dataset <- dataset(
  name = "kminst_dataset",
  inherit = mnist_dataset,
  resources = list(
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134")
  ),
  classes = c('o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo')
)

read_sn3_pascalvincent <- function(path) {
  x <- gzfile(path, open = "rb")
  on.exit({close(x)})

  magic <- readBin(x, endian = "big", what = integer(), n = 1)
  n_dimensions <- magic %% 256
  ty <- magic %/% 256

  dim <- readBin(x, what = integer(), size = 4, endian = "big",
          n = n_dimensions)

  a <- readBin(
    x,
    what = "int", endian = "big", n = prod(dim),
    size = 1, signed = FALSE
  )

  a <- array(a, dim = rev(dim))
  a <- aperm(a, perm = rev(seq_along(dim)))
  a
}

#' QMNIST Dataset
#'
#' Loads and preprocesses the [QMNIST dataset](https://github.com/facebookresearch/qmnist),
#' an extended version of the classic MNIST digit classification dataset that includes
#' extra samples and richer label annotations.
#'
#' This dataset offers a drop-in replacement for MNIST with more data and improved label precision,
#' making it suitable for benchmarking modern models.
#'
#' @param root (string) Root directory where the dataset is or will be stored. The subdirectories
#'   `qmnist/raw/` and `qmnist/processed/` will be created inside this path to hold the original and
#'   processed data respectively.
#' @param train (logical, optional) If `TRUE`, loads the training split (`qmnist-train-*`). If `FALSE`,
#'   loads the test split (`qmnist-test-*`). Defaults to `TRUE`.
#' @param download (logical, optional) If `TRUE`, downloads the QMNIST dataset from the official
#'   repository if it does not already exist. Defaults to `FALSE`.
#' @param transform (function, optional) A function/transform that takes an image tensor and
#'   returns a modified version (e.g., normalization, cropping).
#' @param target_transform (function, optional) A function/transform that takes a target label and
#'   returns a transformed version (e.g., one-hot encoding).
#'
#' @return An R6 dataset object compatible with the `{torch}` package, supporting indexing and
#' iteration over (image, label) pairs.
#'
#' @section Details:
#' This loader mirrors the format of `mnist_dataset()` but includes support for the extended
#' label format of QMNIST (by default using only the digit class column for compatibility).
#' Images are 28x28 grayscale digits stored as 3D arrays.
#'
#' @section File structure:
#' The dataset consists of:
#' - Training images: `qmnist-train-images-idx3-ubyte.gz`
#' - Training labels: `qmnist-train-labels-idx2-int.gz`
#' - Test images: `qmnist-test-images-idx3-ubyte.gz`
#' - Test labels: `qmnist-test-labels-idx2-int.gz`
#'
#' The dataset will be cached under `root/qmnist/`.
#'
#' @seealso
#' - [`mnist_dataset()`] for the original MNIST dataset.
#' - [`kmnist_dataset()`] for the Kuzushiji-MNIST dataset.
#' - The official QMNIST repository: <https://github.com/facebookresearch/qmnist>
#'
#' @examples
#' if (torch::torch_is_installed()) {
#'   ds <- qmnist_dataset(root = tempfile(), download = TRUE)
#'   item <- ds[1]
#'   image <- item$x
#'   label <- item$y
#'   cat("Label:", label, "\n")
#'   image  # a 28x28 matrix
#' }
#'
#' @export
qmnist_dataset <- dataset(
  name = "qmnist_dataset",
  resources = list(
    c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz", "ed72d4157d28c017586c42bc6afe6370"),
    c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz", "0058f8dd561b90ffdd0f734c6a30e5e4"),
    c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz", "1394631089c404de565df7b7aeaf9412"),
    c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz", "5b5b05890a5e13444e108efe57b788aa")
  ),
  training_file = 'training.rds',
  test_file = 'test.rds',
  classes = c('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
              '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine'),
  initialize = function(root, train = TRUE, transform = NULL, target_transform = NULL,
                        download = FALSE) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform

    self$train <- train

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    if (self$train)
      data_file <- self$training_file
    else
      data_file <- self$test_file

    data <- readRDS(file.path(self$processed_folder, data_file))
    self$data <- data[[1]]
    self$targets <- data[[2]][, 1] + 1L
  },
  download = function() {

    if (self$check_exists())
      return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      filename <- tail(strsplit(r[1], "/")[[1]], 1)
      destpath <- file.path(self$raw_folder, filename)

      p <- download_and_cache(r[1], prefix = class(self)[1])
      fs::file_copy(p, destpath)

      if (!tools::md5sum(destpath) == r[2])
        runtime_error("MD5 sums are not identical for file: {r[1]}.")
    }

    rlang::inform("Processing...")

    training_set <- list(
      read_sn3_pascalvincent(file.path(self$raw_folder, 'qmnist-train-images-idx3-ubyte.gz')),
      read_sn3_pascalvincent(file.path(self$raw_folder, 'qmnist-train-labels-idx2-int.gz'))
    )

    test_set <- list(
      read_sn3_pascalvincent(file.path(self$raw_folder, 'qmnist-test-images-idx3-ubyte.gz')),
      read_sn3_pascalvincent(file.path(self$raw_folder, 'qmnist-test-labels-idx2-int.gz'))
    )

    saveRDS(training_set, file.path(self$processed_folder, self$training_file))
    saveRDS(test_set, file.path(self$processed_folder, self$test_file))

    rlang::inform("Done!")
  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img <- self$data[index, ,]
    target <- self$targets[index]

    if (!is.null(self$transform))
      img <- self$transform(img)

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img, y = target)
  },
  .length = function() {
    dim(self$data)[1]
  },
  active = list(
    raw_folder = function() {
      file.path(self$root_path, "qmnist", "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, "qmnist", "processed")
    }
  )
)
