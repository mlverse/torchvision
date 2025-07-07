#' @export
lfw_people_dataset <- torch::dataset(
  name = "lfw_people",
  archive_size = "120 MB",
  resources = list(
    base_url = "https://huggingface.co/datasets/JimmyUnleashed/LFW/resolve/main",
    lfw_images = c("lfw-deepfunneled.zip", "e782a3d6f143c8964f531d0a5af1201a"),
    train = c("peopleDevTrain.csv", "ef0a2b842aa55831d7d55b6bb7d450be"),
    test = c("peopleDevTest.csv", "f95f156446aaee28e8fdeb6c9883eee8")
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$root <- root
    self$train <- train
    self$transform <- transform
    self$target_transform <- target_transform

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (train) {
      csv_file <- self$resources$train[1]
    } else {
      csv_file <- self$resources$test[1]
    }

    csv_path <- file.path(root, csv_file)

    df <- read.csv(csv_path)
    colnames(df) <- c("identity", "num_images")

    all_imgs <- c()
    all_labels <- c()
    class_to_idx <- setNames(seq_len(nrow(df)), df$identity)

    for (i in seq_len(nrow(df))) {
      name <- df$identity[i]
      count <- df$num_images[i]
      for (j in seq_len(count)) {
        file_name <- sprintf("%s_%04d.jpg", name, j)
        path <- file.path(root, "lfw-deepfunneled", name, file_name)
        if (file.exists(path)) {
          all_imgs <- c(all_imgs, path)
          all_labels <- c(all_labels, class_to_idx[[name]])
        }
      }
    }

    self$img_path <- all_imgs
    self$labels <- all_labels
    self$classes <- names(class_to_idx)
    self$class_to_idx <- class_to_idx

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$img_path)} images across {length(self$classes)} classes.")
  },

  download = function() {

    if (self$check_exists()) {
      return()
    }

    fs::dir_create(self$root)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    resources <- list(
      self$resources$lfw_images,
      self$resources$train,
      self$resources$test
    )

    for (res in resources) {
      filename <- res[1]
      expected_md5 <- res[2]

      url <- file.path(self$resources$base_url, filename)
      archive <- download_and_cache(url, prefix = class(self)[1])
      actual_md5 <- tools::md5sum(archive)

      if (actual_md5 != expected_md5) {
        runtime_error("Corrupt file! Delete the file in {archive} and try again.")
      }

      if (tools::file_ext(archive) == "zip") {
        unzip(archive, exdir = self$root)
      } else {
        dest_path <- file.path(self$root, filename)
        fs::file_move(archive, dest_path)
      }
    }

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  check_exists = function() {
    fs::dir_exists(file.path(self$root, "lfw-deepfunneled")) && fs::file_exists(file.path(self$root, self$resources$train[1])) && fs::file_exists(file.path(self$root, self$resources$test[1]))
  },

  .getitem = function(index) {

    x <- jpeg::readJPEG(self$img_path[index])

    y <- self$labels[index]

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }

    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    list(x = x, y = y)
  },

  .length = function() {
    length(self$img_path)
  }
)
