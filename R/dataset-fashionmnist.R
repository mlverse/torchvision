#' FashionMNIST dataset
#'
#' Prepares the FashionMNIST dataset, similar to `mnist1_dataset`.
#'
#' @param root (string): Root directory of dataset.
#' @param train (bool, optional): If TRUE, creates dataset from `training.rds`, otherwise from `test.rds`.
#' @param download (bool, optional): If TRUE, downloads the dataset from the internet.
#' @param transform (callable, optional): Function to transform input data.
#' @param target_transform (callable, optional): Function to transform target labels.
#'
#' @export
fashion_mnist_dataset <- dataset(
  name = "fashion_mnist",
  resources = list(
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    c("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310")
  ),
  training_file = 'training.rds',
  test_file = 'test.rds',
  classes = c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'),
  initialize = function(root, train = TRUE, transform = NULL, target_transform = NULL, download = FALSE) {
    self$root_path <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train

    if (download) self$download()

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use `download = TRUE` to download it.")
    }

    data_file <- if (self$train) self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, data_file))
    self$data <- data[[1]]
    self$targets <- data[[2]] + 1L
  },
  download = function() {
    if (self$check_exists()) return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      filename <- tail(strsplit(r[1], "/")[[1]], 1)
      destpath <- file.path(self$raw_folder, filename)

      p <- download_and_cache(r[1], prefix = class(self)[1])
      fs::file_copy(p, destpath)

      if (!tools::md5sum(destpath) == r[2]) {
        runtime_error("MD5 sums do not match for file: {r[1]}")
      }
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

    rlang::inform("Download and processing complete!")
  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img <- self$data[index, , ]
    target <- self$targets[index]

    if (!is.null(self$transform)) img <- self$transform(img)
    if (!is.null(self$target_transform)) target <- self$target_transform(target)

    list(x = img, y = target)
  },
  .length = function() {
    dim(self$data)[1]
  },
  active = list(
    raw_folder = function() {
      file.path(self$root_path, "fashion_mnist", "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, "fashion_mnist", "processed")
    }
  )
)
