#' Labelled Faces in the Wild (LFW) Datasets
#'
#' The LFW dataset collection provides facial images for evaluating face recognition systems.
#' It includes two variants:
#' - `lfw_people_dataset`: A **multi-class classification** dataset where each image is labelled by person identity.
#' - `lfw_pairs_dataset`: A **face verification** dataset containing image pairs with binary labels (same or different person).
#'
#' @inheritParams oxfordiiitpet_dataset
#' @param root Root directory for dataset storage. The dataset will be stored under `root/lfw_people` or `root/lfw_pairs`.
#'
#' @return A torch dataset object \code{lfw_people_dataset} or \code{lfw_pairs_dataset}.
#' Each element is a named list with:
#' - \code{x}:
#'   - For \code{lfw_people_dataset}: a H x W x 3 numeric array representing a single RGB image.
#'   - For \code{lfw_pairs_dataset}: a list of two H x W x 3 numeric arrays representing a pair of RGB images.
#' - \code{y}:
#'   - For \code{lfw_people_dataset}: an integer index from 1 to the number of identities in the dataset.
#'   - For \code{lfw_pairs_dataset}: 1 if the pair shows the same person, 2 if different people.
#'
#' @details
#' This R implementation of the LFW dataset is based on the `torchvision.datasets.LFWPeople` and `LFWPairs` classes in PyTorch,
#' but deviates in a few key aspects due to dataset availability and R API conventions:
#'
#' - The \code{image_set} argument from Python (``original``, ``funneled``, ``deepfunneled``) has been removed. Only the
#'   ``deepfunneled`` version is supported in R, as the ``original`` and ``funneled`` versions are no longer accessible via
#'   the URLs used in the official PyTorch implementation.
#'
#' - The \code{split} argument in Python (``train``, ``test``, ``10fold``) is simplified to a \code{train} boolean flag in R.
#'   This follows existing conventions in the R `torchvision` dataset API. The ``10fold`` split is not supported because the
#'   official file required for this split is unavailable or incompatible with a clean split into image-label pairs.
#'
#' - The dataset uses the "deep funneled" version of LFW, offering aligned and cropped RGB face images.
#'   This ensures consistent image size and alignment, making it suitable for use in both classification and verification tasks.
#'
#' - The data is downloaded from [Hugging Face](https://huggingface.co/datasets/JimmyUnleashed/LFW),
#'   which provides consistent hosting for the deep funneled version.
#'
#' The LFW People dataset uses the "deep funneled" version of LFW, offering aligned and cropped RGB face images.
#' - Training split: 9525 images, 4038 identities
#' - Test split: 3708 images, 1711 identities
#'
#' @examples
#' \dontrun{
#' # Load training data for LFW People Dataset
#' lfw <- lfw_people_dataset(download = TRUE, train = TRUE)
#' first_item <- lfw[1]
#' first_item$x  # RGB image
#' first_item$y  # Label index
#' lfw$classes[first_item$y]  # person's name (e.g., "AJ_Cook")
#'
#' # Load test data for LFW People Dataset
#' lfw_test <- lfw_people_dataset(download = TRUE, train = FALSE)
#' first_item <- lfw_test[1]
#' first_item$x  # RGB image
#' first_item$y  # Label index
#' lfw_test$classes[first_item$y]  # e.g., "AJ_Lamas"
#'
#' # Load training data for LFW Pairs Dataset
#' lfw <- lfw_pairs_dataset(download = TRUE, train = TRUE)
#' first_item <- lfw[1]
#' first_item$x  # List of 2 RGB Images
#' first_item$x[[1]]  # RGB Image
#' first_item$x[[2]]  # RGB Image
#' first_item$y  # Label index
#' lfw$classes[first_item$y]  # Class Name
#'
#' # Load test data for LFW Pairs Dataset
#' lfw <- lfw_pairs_dataset(download = TRUE, train = FALSE)
#' first_item <- lfw[1]
#' first_item$x  # List of 2 RGB Images
#' first_item$x[[1]]  # RGB Image
#' first_item$x[[2]]  # RGB Image
#' first_item$y  # Label index
#' lfw$classes[first_item$y]  # Class Name
#' }
#'
#' @name lfw_dataset
#' @title LFW Datasets
#' @rdname lfw_dataset
#' @family classification_dataset
#' @export
lfw_people_dataset <- torch::dataset(
  name = "lfw_people",
  archive_size = "120 MB",
  base_url = "https://huggingface.co/datasets/JimmyUnleashed/LFW/resolve/main",
  resources = list(
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

    for (res in self$resources) {
      filename <- res[1]
      expected_md5 <- res[2]

      url <- file.path(self$base_url, filename)
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

#' @rdname lfw_dataset
#' @export
lfw_pairs_dataset <- torch::dataset(
  inherit = lfw_people_dataset,
  name = "lfw_pairs",
  archive_size = "120 MB",
  base_url = "https://huggingface.co/datasets/JimmyUnleashed/LFW/resolve/main",
  resources = list(
    lfw_images = c("lfw-deepfunneled.zip", "e782a3d6f143c8964f531d0a5af1201a"),
    match_train = c("matchpairsDevTrain.csv", "f7502e5a6350c7d2b795acf71406bc33"),
    mismatch_train = c("mismatchpairsDevTrain.csv", "80994187f484fdefb4713a96aa9ddeb6"),
    match_test = c("matchpairsDevTest.csv", "e4470bc1288afccf7e5cecc3623271a4"),
    mismatch_test = c("mismatchpairsDevTest.csv", "cb9f0975a0c19306cff2e6bdd7a7a7e3")
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
    self$classes <- c("Same", "Different")

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (train) {
      match_file <- self$resources$match_train[1]
      mismatch_file <- self$resources$mismatch_train[1]
    } else {
      match_file <- self$resources$match_test[1]
      mismatch_file <- self$resources$mismatch_test[1]
    }

    match_path <- file.path(root, match_file)
    mismatch_path <- file.path(root, mismatch_file)

    match_df <- read.csv(match_path)
    mismatch_df <- read.csv(mismatch_path)

    matched <- data.frame(
      img1 = file.path(match_df$name, sprintf("%s_%04d.jpg", match_df$name, match_df$imagenum1)),
      img2 = file.path(match_df$name, sprintf("%s_%04d.jpg", match_df$name, match_df$imagenum2)),
      same = 1
    )

    mismatched <- data.frame(
      img1 = file.path(mismatch_df$name, sprintf("%s_%04d.jpg", mismatch_df$name, mismatch_df$imagenum1)),
      img2 = file.path(mismatch_df$name.1, sprintf("%s_%04d.jpg", mismatch_df$name.1, mismatch_df$imagenum2)),
      same = 2
    )

    self$pairs <- rbind(matched, mismatched)
    self$img_path <- c(self$pairs$img1, self$pairs$img2)
    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$img_path)} images across {length(self$classes)} classes.")
  },


  .getitem = function(index) {
    pair <- self$pairs[index, ]

    img1_path <- file.path(self$root, "lfw-deepfunneled", pair$img1)
    img2_path <- file.path(self$root, "lfw-deepfunneled", pair$img2)

    x1 <- jpeg::readJPEG(img1_path)
    x2 <- jpeg::readJPEG(img2_path)
    x <- list(x1, x2)
    y <- as.integer(pair$same)

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }

    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    list(x = x, y = y)
  },

  .length = function() {
    nrow(self$pairs)
  },

  check_exists = function() {
    required_files <- c(
      self$resources$match_train[1],
      self$resources$mismatch_train[1],
      self$resources$match_test[1],
      self$resources$mismatch_test[1]
    )
    fs::dir_exists(file.path(self$root, "lfw-deepfunneled")) && all(fs::file_exists(file.path(self$root, required_files)))
  }
)
