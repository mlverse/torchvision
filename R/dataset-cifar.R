cifar10_dataset <- torch::dataset(
  name = "cifar10_dataset",
  url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
  md5 = "c32a1d4ab5d03f1284b67883e8d87530",
  fname = "cifar-10-batches-bin",
  type = 10,
  label_fname = "batches.meta.txt",
  initialize = function(root, train = TRUE, transform = NULL, taregt_transform = NULL,
                        download = FALSE) {
    self$root <- root

    if (download)
      self$download()

    check <- self$check_files()

    if (!check)
      runtime_error("Files not found. Use download = TRUE")

    if (train) {
      files <- self$get_files()$train
    } else {
      files <- self$get_files()$test
    }

    batches <- lapply(files, function(x) read_batch(x, self$type))

    if (self$type == 10)
      data <- combine_batches(batches)
    else
      data <- batches[[1]]

    self$.load_meta()

    self$x <- data$imgs
    self$y <- data$labels + 1L
  },
  .load_meta = function() {
    cl <- readLines(fs::path(self$root, self$fname, self$label_fname))
    self$class_to_idx <- setNames(seq_along(cl), cl)
    self$classes <- cl
  },
  .getitem = function(i) {
    x <- self$x[i,,,]
    y <- self$y[i]

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
  },
  .length = function() {
    length(self$y)
  },
  download = function() {

    if(self$check_files())
      return()

    p <- download_and_cache(self$url)

    if (!tools::md5sum(p) == self$md5)
      runtime_error("Corrupt file!")

    utils::untar(p, exdir = self$root)
  },
  check_files = function() {

    if (!fs::dir_exists(self$root))
      return(FALSE)

    p <- fs::path(self$root, self$fname)
    if (!fs::dir_exists(p))
      return(FALSE)

    f <- self$get_files()

    if (!length(f$train) == 5 && self$type == 10)
      return(FALSE)

    if (!length(f$train) == 1 && self$type == 100)
      return(FALSE)

    if (!length(f$test) == 1)
      return(FALSE)

    return(TRUE)
  },
  get_files = function() {
    p <- fs::path(self$root, self$fname)

    if (self$type == 10) {
      list(
        train = fs::dir_ls(p, regexp = "data_batch"),
        test = fs::dir_ls(p, regexp = "test_batch")
      )
    } else {
      list(
        train = fs::dir_ls(p, regexp = "train"),
        test = fs::dir_ls(p, regexp = "test")
      )
    }
  }
)

cifar100_dataset <- torch::dataset(
  name = "cifar100_dataset",
  inherit = cifar10_dataset,
  url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
  md5 = "03b5dce01913d631647c71ecec9e9cb8",
  fname = "cifar-100-binary",
  type = 100,
  label_fname = "fine_label_names.txt"
)

read_batch <- function(path, type = 10) {

  if (type == 10)
    n <- 10000
  else if (type == 100 && grepl("test", path))
    n <- 10000
  else
    n <- 50000

  imgs <- array(dim = c(n, 32, 32, 3))
  labels <- integer(length = n)
  if (type == 100)
    fine_labels <- integer(length = n)

  con <- file(path, open = "rb")
  on.exit({close(con)}, add = TRUE)

  for (i in seq_len(n)) {

    labels[i] <- readBin(con, integer(), size=1, n=1, endian="big")

    if (type == 100) {
      fine_labels[i] <- readBin(con, integer(), size=1, n=1, endian="big")
    }

    r <- as.integer(readBin(con, raw(), size=1, n=1024, endian="big"))
    g <- as.integer(readBin(con, raw(), size=1, n=1024, endian="big"))
    b <- as.integer(readBin(con, raw(), size=1, n=1024, endian="big"))

    imgs[i,,,1] <- matrix(r, ncol = 32, byrow = TRUE)
    imgs[i,,,2] <- matrix(g, ncol = 32, byrow = TRUE)
    imgs[i,,,3] <- matrix(b, ncol = 32, byrow = TRUE)
  }

  if (type == 100)
    list(imgs = imgs, labels = fine_labels)
  else
    list(imgs = imgs, labels = labels)
}

combine_batches <- function(batches) {

  n <- 10000

  imgs <- array(dim = c(length(batches)* n, 32, 32, 3))
  labels <- integer(length = length(batches)* n)

  for (i in seq_along(batches)) {
    imgs[((i-1)*n + 1):(i*n),,,] <- batches[[i]]$imgs
    labels[((i-1)*n + 1):(i*n)] <- batches[[i]]$labels
  }

  list(imgs = imgs, labels = labels)
}

