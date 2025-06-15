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
  initialize = function(root = tempdir(), train = TRUE, transform = NULL, target_transform = NULL,
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

#' QMNIST Dataset
#'
#' Loads and preprocesses the [QMNIST dataset](https://github.com/facebookresearch/qmnist),
#' including optional support for the NIST digit subset.
#'
#' This dataset is an extended version of the original MNIST, offering more samples and precise label
#' information. It is suitable for benchmarking modern machine learning models and can serve as a
#' drop-in replacement for MNIST in most image classification tasks.
#'
#' @inheritParams mnist_dataset
#' @param split (string, optional) Which subset to load: one of `"train"`, `"test"`, or `"nist"`.
#'   Defaults to `"train"`. The `"nist"` option loads the full NIST digits set.
#'
#' @return An R6 dataset object compatible with the `{torch}` package, providing indexed access
#'   to (image, label) pairs from the specified QMNIST subset.
#'
#' @section Supported Subsets:
#' - `"train"`: 60,000 training examples (compatible with MNIST)
#' - `"test"`: 60,000 test examples (extended QMNIST test set)
#' - `"nist"`: Entire NIST digit dataset (for advanced benchmarking)
#'
#' @seealso [mnist_dataset()], [kmnist_dataset()], [fashion_mnist_dataset()]
#'
#' @examples
#' \dontrun{
#' qmnist <- qmnist_dataset(split = "train", download = TRUE)
#' first_item <- qmnist[1]
#' # image in item 1
#' first_item$x
#' # label of item 1
#' first_item$y
#' }
#'
#' @export
qmnist_dataset <- dataset(
  name = "qmnist_dataset",
  resources = list(
    train = list(
      c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz", "ed72d4157d28c017586c42bc6afe6370"),
      c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz", "0058f8dd561b90ffdd0f734c6a30e5e4")
    ),
    test = list(
      c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz", "1394631089c404de565df7b7aeaf9412"),
      c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz", "5b5b05890a5e13444e108efe57b788aa")
    ),
    nist = list(
      c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-images-idx3-ubyte.xz", "7f124b3b8ab81486c9d8c2749c17f834"),
      c("https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-labels-idx2-int.xz", "5ed0e788978e45d4a8bd4b7caec3d79d")
    )
  ),
  files = list(
    train = "training.rds",
    test = "test.rds",
    nist = "nist.rds"
  ),
  classes = c(
    '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
    '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine'
  ),

  initialize = function(root = tempdir(), split = "train", transform = NULL, target_transform = NULL, download = FALSE) {
    split <- match.arg(split, c("train", "test", "nist"))
    self$split <- split
    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to fetch it.")

    data_file <- self$files[[split]]
    data <- readRDS(file.path(self$processed_folder, data_file))

    self$data <- data[[1]]
    self$targets <- data[[2]][, 1] + 1L
  },

  download = function() {
    if (self$check_exists())
      return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources[[self$split]]) {
      filename <- basename(r[1])
      destpath <- file.path(self$raw_folder, filename)

      p <- download_and_cache(r[1], prefix = glue::glue("qmnist-{self$split}"))
      fs::file_copy(p, destpath, overwrite = TRUE)

      if (!tools::md5sum(destpath) == r[2])
        runtime_error(paste("MD5 mismatch for:", r[1]))
    }

    rlang::inform("Processing...")

    if (self$split == "train") {
      saveRDS(
        list(
          read_sn3_pascalvincent(file.path(self$raw_folder, "qmnist-train-images-idx3-ubyte.gz")),
          read_sn3_pascalvincent(file.path(self$raw_folder, "qmnist-train-labels-idx2-int.gz"))
        ),
        file.path(self$processed_folder, self$files$train)
      )
    }

    if (self$split == "test") {
      saveRDS(
        list(
          read_sn3_pascalvincent(file.path(self$raw_folder, "qmnist-test-images-idx3-ubyte.gz")),
          read_sn3_pascalvincent(file.path(self$raw_folder, "qmnist-test-labels-idx2-int.gz"))
        ),
        file.path(self$processed_folder, self$files$test)
      )
    }

    if (self$split == "nist") {
      saveRDS(
        list(
          read_sn3_pascalvincent(file.path(self$raw_folder, "xnist-images-idx3-ubyte.xz")),
          read_sn3_pascalvincent(file.path(self$raw_folder, "xnist-labels-idx2-int.xz"))
        ),
        file.path(self$processed_folder, self$files$nist)
      )
    }

    rlang::inform("Done!")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$files[[self$split]]))
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
#' @param train (bool, optional): If TRUE, creates dataset from \code{training.pt},
#' otherwise from \code{test.pt}.
#' @param transform (callable, optional): A function/transform that takes in an
#' image and returns a transformed version. E.g., \code{\link[=transform_random_crop]{transform_random_crop()}}.
#' @param target_transform (callable, optional): A function/transform that takes
#' in the target and transforms it.
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
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
  )
)

#' EMNIST Dataset
#'
#' Loads the EMNIST dataset, a set of handwritten digits and letters with multiple splits:
#' - "byclass": 62 classes (digits + uppercase + lowercase)
#' - "bymerge": 47 classes (merged uppercase and lowercase letters)
#' - "balanced": 47 classes balanced between digits and letters
#' - "letters": 26 letter classes only
#' - "digits": 10 digit classes only
#' - "mnist": classic 10 digit classes like the original MNIST dataset
#'
#' @param root Character. Root directory for dataset storage (default folder: `root/emnist/processed/`).
#' @param split Character. Dataset split to use. One of `"byclass"`, `"bymerge"`, `"balanced"`, `"letters"`, `"digits"`, or `"mnist"`. Default is `"balanced"`.
#' @param download Logical. Whether to download the dataset if it is not found locally. Default is `FALSE`.
#' @param transform Optional function to transform input images.
#' @param target_transform Optional function to transform labels.
#'
#' @return An EMNIST dataset object.
#'
#' @examples
#' \dontrun{
#' emnist <- emnist_dataset(split = "balanced", download = TRUE)
#' first_item <- emnist[1]
#' # image in item 1
#' first_item$x
#' # label of item 1
#' first_item$y
#' }
#'
#' @seealso [mnist_dataset()], [kmnist_dataset()], [fashion_mnist_dataset()]
#'
#' @name emnist_dataset
#' @aliases emnist_dataset
#' @title EMNIST dataset
#' @export
emnist_dataset <- dataset(
  name = "emnist_dataset",
  resources = list(
    c("https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", "58c8d27c78d21e728a6bc7b3cc06412e")
  ),
  training_file = function(split) glue::glue("training-{split}.rds"),
  test_file = function(split) glue::glue("test-{split}.rds"),
  classes_list = list(
    byclass = c(
      "0","1","2","3","4","5","6","7","8","9",
      LETTERS,
      letters[c(1:9,11,12,14,17,20)]
    ),
    bymerge = c(
      "0","1","2","3","4","5","6","7","8","9",
      LETTERS[1:26],
      letters[c(1:4,6,7,10,13,16,19,21,22,24,25,26)]
    ),
    balanced = c(
      "0","1","2","3","4","5","6","7","8","9",
      LETTERS,
      letters[c(1:9,11,12,14,17,20)]
    ),
    letters = letters,
    digits = as.character(0:9),
    mnist = as.character(0:9)
  ),

  initialize = function(root = tempdir(), split = "balanced", transform = NULL, target_transform = NULL,
                        download = FALSE) {
    rlang::inform(glue::glue(
      "Preparing to download EMNIST dataset. Archive size is ~0.5GB\n",
      "You may have to increase the download timeout in your session with `options()` in case of failure\n",
      "- Will extract and convert for all {length(self$classes_list)} splits\n"
    ))
    split <- match.arg(split, choices = names(self$classes_list))
    self$split <- split
    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$classes <- self$classes_list[[split]]

    if (download)
      self$download()

    if (!self$check_exists())
      runtime_error("Dataset not found. Use `download = TRUE` to fetch it.")

    file_to_load_train <- self$training_file(split)
    file_to_load_test <- self$test_file(split)

    training_data <- readRDS(file.path(self$processed_folder, file_to_load_train))
    test_data <- readRDS(file.path(self$processed_folder, file_to_load_test))

    self$data <- training_data[[1]]
    self$targets <- training_data[[2]] + 1L
    self$test_data <- test_data[[1]]
    self$test_targets <- test_data[[2]] + 1L

    self$is_train <- TRUE
    rlang::inform("EMNIST dataset processed successfully!")
  },

  download = function() {
    if (self$check_exists()) return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    url <- self$resources[[1]][1]
    expected_md5 <- self$resources[[1]][2]
    zip_path <- download_and_cache(url, prefix = class(self)[1])

    actual_md5 <- digest::digest(file = zip_path, algo = "md5")
    if (!identical(actual_md5, expected_md5))
      runtime_error("Downloaded EMNIST archive has incorrect checksum.")

    unzip_dir <- file.path(self$raw_folder, "unzipped")
    fs::dir_create(unzip_dir)
    unzip(zip_path, exdir = unzip_dir)

    unzipped_root <- fs::dir_ls(unzip_dir, type = "directory", recurse = FALSE)[1]

    process_split <- function(split_name) {
      train_img <- file.path(unzipped_root, glue::glue("emnist-{split_name}-train-images-idx3-ubyte.gz"))
      train_lbl <- file.path(unzipped_root, glue::glue("emnist-{split_name}-train-labels-idx1-ubyte.gz"))
      test_img  <- file.path(unzipped_root, glue::glue("emnist-{split_name}-test-images-idx3-ubyte.gz"))
      test_lbl  <- file.path(unzipped_root, glue::glue("emnist-{split_name}-test-labels-idx1-ubyte.gz"))

      train_set <- list(read_sn3_pascalvincent(train_img),
                        read_sn3_pascalvincent(train_lbl))
      test_set <- list(read_sn3_pascalvincent(test_img),
                       read_sn3_pascalvincent(test_lbl))

      saveRDS(train_set, file.path(self$processed_folder, self$training_file(split_name)))
      saveRDS(test_set, file.path(self$processed_folder, self$test_file(split_name)))
    }

    for (split_name in names(self$classes_list)) {
      process_split(split_name)
    }
  },

  check_exists = function() {
    all(sapply(names(self$classes_list), function(split_name) {
      fs::file_exists(file.path(self$processed_folder, self$training_file(split_name))) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file(split_name)))
    }))
  },

  .getitem = function(index) {
    data_set <- if (self$is_train) self$data else self$test_data
    targets_set <- if (self$is_train) self$targets else self$test_targets

    img <- data_set[index, , ]
    target <- targets_set[index]

    if (!is.null(self$transform))
      img <- self$transform(img)

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = img, y = target)
  },

  .length = function() {
    data_set <- if (self$is_train) self$data else self$test_data
    dim(data_set)[1]
  },

  active = list(
    raw_folder = function() {
      file.path(self$root_path, "emnist", "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, "emnist", "processed")
    }
  ),

  set_train = function(train = TRUE) {
    self$is_train <- train
    invisible(self)
  }
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
