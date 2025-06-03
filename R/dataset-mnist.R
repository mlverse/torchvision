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

#' Fashion-MNIST dataset
#'
#' @usage
#' fashion_mnist_dataset(
#'   root,
#'   train = TRUE,
#'   transform = NULL,
#'   target_transform = NULL,
#'   download = FALSE
#' )
#'
#' @param root (string): Root directory of dataset where
#' \code{FashionMNIST/processed/training.pt} and \code{FashionMNIST/processed/test.pt} exist.
#'
#' @param train (bool, optional): If TRUE, creates dataset from \code{training.pt},
#' otherwise from \code{test.pt}.
#'
#' @param transform (callable, optional): A function/transform that takes in an
#' image and returns a transformed version. E.g., \code{\link[=transform_random_crop]{transform_random_crop()}}.
#'
#' @param target_transform (callable, optional): A function/transform that takes
#' in the target and transforms it.
#'
#' @param download (bool, optional): If TRUE, downloads the dataset from the
#' internet and puts it in root directory. If dataset is already downloaded,
#' it is not downloaded again.
#'
#' @description
#' Prepares the \href{https://github.com/zalandoresearch/fashion-mnist}{Fashion-MNIST} dataset
#' and optionally downloads it.
#'
#' @seealso [mnist_dataset()], [kmnist_dataset()]
#'
#' @name fashion_mnist_dataset
#' @aliases fashion_mnist_dataset
#' @title Fashion-MNIST dataset
#' @export
fashion_mnist_dataset <- dataset(
  name = "fashion_mnist_dataset",
  inherit = mnist_dataset,
  resources = list(
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
  ),
  classes = c(
    '0 - T-shirt/top', '1 - Trouser', '2 - Pullover', '3 - Dress', '4 - Coat',
    '5 - Sandal', '6 - Shirt', '7 - Sneaker', '8 - Bag', '9 - Ankle boot'
  )
)

