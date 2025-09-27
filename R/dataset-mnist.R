#' MNIST and Derived Datasets
#'
#' Prepares various MNIST-style image classification datasets and optionally downloads them.
#' Images are thumbnails images of 28 x 28 pixels of grayscale values encoded as integer.
#'
#' - **MNIST**: Original handwritten digit dataset.
#' - **Fashion-MNIST**: Clothing item images for classification.
#' - **Kuzushiji-MNIST**: Japanese cursive character dataset.
#' - **QMNIST**: Extended MNIST with high-precision NIST data.
#' - **EMNIST**: A collection of letters and digits with multiple datasets and  splits.
#'
#' @param root Root directory for dataset storage. The dataset will be stored under `root/<dataset-name>`. Defaults to `tempdir()`.
#' @param train Logical. If TRUE, use the training set; otherwise, use the test set. Not applicable to all datasets.
#' @param split Character. Used in `emnist_dataset()` and `qmnist_dataset()` to specify the subset. See individual descriptions for valid values.
#' @param download Logical. If TRUE, downloads the dataset to `root/`. If the dataset is already present, download is skipped.
#' @param transform Optional. A function that takes an image and returns a transformed version (e.g., normalization, cropping).
#' @param target_transform Optional. A function that transforms the label.
#'
#' @return A torch dataset object, where each items is a list of `x` (image) and `y` (label).
#'
#' @section Supported `dataset`s for `emnist_collection()`:
#' - `"byclass"`: 62 classes (digits + uppercase + lowercase)
#' - `"bymerge"`: 47 classes (merged uppercase and lowercase)
#' - `"balanced"`: 47 classes, balanced digits and letters
#' - `"letters"`: 26 uppercase letters
#' - `"digits"`: 10 digit classes
#' - `"mnist"`: Standard MNIST digit classes
#'
#' @section Supported `split`s for `qmnist_dataset()`:
#' - `"train"`: 60,000 training samples (MNIST-compatible)
#' - `"test"`: Extended test set
#' - `"nist"`: Full NIST digit set
#'
#' @examples
#' \dontrun{
#' ds <- mnist_dataset(download = TRUE)
#' item <- ds[1]
#' item$x  # image
#' item$y  # label
#'
#' qmnist <- qmnist_dataset(split = "train", download = TRUE)
#' item <- qmnist[1]
#' item$x
#' item$y
#'
#' emnist <- emnist_collection(dataset = "balanced", split = "test", download = TRUE)
#' item <- emnist[1]
#' item$x
#' item$y
#'
#' kmnist <- kmnist_dataset(download = TRUE, train = FALSE)
#' fmnist <- fashion_mnist_dataset(download = TRUE, train = TRUE)
#' }
#'
#' @family classification_dataset
#' @name mnist_dataset
#' @rdname mnist_dataset
#' @export
mnist_dataset <- dataset(
  name = "mnist",
  archive_size = "12 MB",
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
  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train

    if (download){
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    if (self$train)
      data_file <- self$training_file
    else
      data_file <- self$test_file

    data <- readRDS(file.path(self$processed_folder, data_file))
    self$data <- data[[1]]
    self$targets <- data[[2]] + 1L

    cli_inform("Dataset {.cls {class(self)[[1]]}} loaded with {length(self$targets)} images.")
  },

  download = function() {

    if (self$check_exists())
      return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      filename <- tail(strsplit(r[1], "/")[[1]], 1)
      destpath <- file.path(self$raw_folder, filename)

      archive <- download_and_cache(r[1], prefix = class(self)[1])
      fs::file_copy(archive, destpath)

      if (!tools::md5sum(destpath) == r[2])
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")

    }

        cli_inform("Downloading {.cls {class(self)[[1]]}} ...")
    cli_inform("Processing {.cls {class(self)[[1]]}}...")

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

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")

  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },

  .getitem = function(index) {
    x <- self$data[index, ,]
    y <- self$targets[index]

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
  },

  .getbatch = function(index) {
    x <- self$data[index, ,]
    if (length(index) > 1) {
      x <-  aperm(x, c(2, 3, 1))
    }

    y <- self$targets[index]

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
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

#' @describeIn mnist_dataset Kuzushiji-MNIST cursive Japanese character dataset.
#' @export
kmnist_dataset <- dataset(
  name = "kminst_dataset",
  inherit = mnist_dataset,
  archive_size = "21 MB",
  resources = list(
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7"),
    c("http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134")
  ),
  classes = c('o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo')
)

#' @describeIn mnist_dataset Extended MNIST dataset with high-precision test data (QMNIST).
#' @export
qmnist_dataset <- dataset(
  name = "qmnist_dataset",
  inherit = mnist_dataset,
  archive_size = "70 MB",

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

  initialize = function(
    root = tempdir(),
    split = "train",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    split <- match.arg(split, c("train", "test", "nist"))
    self$split <- split
    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform

    if (download){
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

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

    cli_inform("Downloading {.cls {class(self)[[1]]}} ...")
    for (r in self$resources[[self$split]]) {
      filename <- basename(r[1])
      destpath <- file.path(self$raw_folder, filename)

      archive <- download_and_cache(r[1], prefix = glue::glue("qmnist-{self$split}"))
      fs::file_copy(archive, destpath, overwrite = TRUE)

      if (!tools::md5sum(destpath) == r[2])
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }

    cli_inform("Processing {.cls {class(self)[[1]]}} ...")


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

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$files[[self$split]]))
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

#' @describeIn mnist_dataset Fashion-MNIST clothing image dataset.
#' @export
fashion_mnist_dataset <- dataset(
  name = "fashion_mnist_dataset",
  inherit = mnist_dataset,
  archive_size = "30 MB",
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

#' @describeIn mnist_dataset EMNIST dataset with digits and letters and multiple split modes.
#' @param kind change the classes into one of "byclass", "bymerge", "balanced" representing the kind of emnist dataset. You
#' can look at dataset attribute `$classes` to see the actual classes.
#' @export
emnist_collection <- dataset(
  name = "emnist_collection",
  archive_size = "540 MB",

  resources = list(
    c("https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", "58c8d27c78d21e728a6bc7b3cc06412e")
  ),
  rds_file = function(split, kind) paste0(split,"-",kind,".rds"),
  classes_all_dataset = list(
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

  initialize = function(
    root = tempdir(),
    split = "test",
    dataset = "balanced",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$split <- match.arg(split, choices = c("train", "test"))
    self$dataset <- match.arg(dataset,  choices = names(self$classes_all_dataset))
    self$root_path <- root
    self$raw_folder <- file.path(root, class(self)[1], "raw")
    self$processed_folder <- file.path(root, class(self)[1], "processed")
    self$transform <- transform
    self$target_transform <- target_transform
    self$class <- self$classes_all_dataset[[self$dataset]]

    if (download) {
      cli_inform("{.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    dataset_lst <- readRDS(file.path(self$processed_folder, self$rds_file(self$split, self$dataset)))
    self$data <- dataset_lst[[1]]
    self$targets <- dataset_lst[[2]] + 1L

    cli_inform("Split {.val {self$split}} of dataset {.val {self$dataset}} from {.cls {class(self)[[1]]}} processed successfully!")
  },

  download = function() {
    if (self$check_exists()) return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    url <- self$resources[[1]][1]
    archive <- download_and_cache(url, prefix = class(self)[1])

    if (!tools::md5sum(archive) == self$resources[[1]][2])
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")

    unzip_dir <- file.path(self$raw_folder, "unzipped")
    fs::dir_create(unzip_dir)
    unzip(archive, exdir = unzip_dir)

    # unzip second level of archives
    unzipped_root <- fs::dir_ls(unzip_dir, type = "directory", recurse = FALSE)[1]

    # only manage extraction of the 2 ubyte.gz under interest
    img <- file.path(unzipped_root, glue::glue("emnist-{self$dataset}-{self$split}-images-idx3-ubyte.gz"))
    lbl <- file.path(unzipped_root, glue::glue("emnist-{self$dataset}-{self$split}-labels-idx1-ubyte.gz"))
    dataset_set <- list(read_sn3_pascalvincent(img), read_sn3_pascalvincent(lbl))
    saveRDS(dataset_set, file.path(self$processed_folder, self$rds_file(self$split, self$dataset)))

  },
  # only manage existence of the rds file under interest
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$rds_file(self$split, self$dataset)))
  },

  .getitem = function(index) {

    x <- self$data[index, , ]
    y <- self$targets[index]

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
  },

  .length = function() {
    dim(self$data)[1]
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
