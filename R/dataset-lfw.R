#' Labelled Faces in the Wild (LFW) Datasets
#'
#' The LFW dataset collection provides facial images for evaluating face recognition systems.
#' It includes two variants:
#' - `lfw_people_dataset`: A **multi-class classification** dataset where each image is labelled by person identity.
#' - `lfw_pairs_dataset`: A **face verification** dataset containing image pairs with binary labels (same or different person).
#'
#' @inheritParams oxfordiiitpet_dataset
#' @param root Root directory for dataset storage. The dataset will be stored under `root/lfw_people` or `root/lfw_pairs`.
#' @param split Which version of the dataset to use. One of `"original"` or `"funneled"`. Defaults to `"original"`.
#' @param train For `lfw_pairs_dataset`, whether to load the training (`pairsDevTrain.txt`) or test (`pairsDevTest.txt`) split.
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
#' This R implementation of the LFW dataset is based on the `fetch_lfw_people()` and `fetch_lfw_pairs()` functions from the `scikit-learn` library,
#' but deviates in a few key aspects due to dataset availability and R API conventions:
#'
#' - The \code{color} and \code{resize} arguments from Python are not directly exposed. Instead, all images are RGB with a fixed size of 250x250.
#'
#' - The \code{split} argument in Python (e.g., ``train``, ``test``, ``10fold``) is simplified to a \code{train} boolean flag in R.
#'   The ``10fold`` split is not supported, as the original protocol files are unavailable or incompatible with clean separation of image-label pairs.
#' 
#' - The \code{split} parameter in R controls which version of the dataset to use: `"original"` (unaligned) or `"funneled"` (aligned using funneling).
#'   The funneled version contains geometrically normalized face images, offering better alignment and typically improved performance for face recognition models.
#'
#' - The dataset is downloaded from [Figshare](https://figshare.com/authors/_/3118605),
#'   which hosts the same files referenced in `scikit-learn`'s dataset utilities.
#'
#' - `lfw_people_dataset`: 13,233 images across multiple identities (using either `"original"` or `"funneled"` splits)
#' - `lfw_pairs_dataset`:
#'   - Training split (`train = TRUE`): 2,200 image pairs
#'   - Test split (`train = FALSE`): 1,000 image pairs
#'
#' @examples
#' \dontrun{
#' # Load data for LFW People Dataset
#' lfw <- lfw_people_dataset(download = TRUE)
#' first_item <- lfw[1]
#' first_item$x  # RGB image
#' first_item$y  # Label index
#' lfw$classes[first_item$y]  # person's name (e.g., "Aaron_Eckhart")
#'
#' # Load training data for LFW Pairs Dataset
#' lfw <- lfw_pairs_dataset(download = TRUE, train = TRUE)
#' first_item <- lfw[1]
#' first_item$x  # List of 2 RGB Images
#' first_item$x[[1]]  # RGB Image
#' first_item$x[[2]]  # RGB Image
#' first_item$y  # Label index
#' lfw$classes[first_item$y]  # Class Name (e.g., "Same" or "Different")
#'
#' # Load test data for LFW Pairs Dataset
#' lfw <- lfw_pairs_dataset(download = TRUE, train = FALSE)
#' first_item <- lfw[1]
#' first_item$x  # List of 2 RGB Images
#' first_item$x[[1]]  # RGB Image
#' first_item$x[[2]]  # RGB Image
#' first_item$y  # Label index
#' lfw$classes[first_item$y]  # Class Name (e.g., "Same" or "Different")
#' }
#'
#' @name lfw_dataset
#' @title LFW Datasets
#' @rdname lfw_dataset
#' @family classification_dataset
#' @export
lfw_people_dataset <- torch::dataset(
  name = "lfw_people",
  archive_size_table = list(
    "original" = "170 MB",
    "funneled" = "230 MB"
  ),
  base_url = "https://ndownloader.figshare.com/files/",
  resources = list(
    original = c("5976018", "a17d05bd522c52d84eca14327a23d494"),
    funneled = c("5976015", "1b42dfed7d15c9b2dd63d5e5840c86ad")
  ),

  initialize = function(
    root = tempdir(),
    transform = NULL,
    split = "original",
    target_transform = NULL,
    download = FALSE
  ) {

    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$archive_size <- self$archive_size_table[[split]]

    self$split <- split
    if (split == "original"){
      self$split_name <- "lfw"
    } else {
      self$split_name <- "lfw_funneled"
    }

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    image_dir <- file.path(root, self$split_name)

    person_dirs <- list.dirs(image_dir, full.names = TRUE, recursive = FALSE)
    class_names <- basename(person_dirs)
    class_to_idx <- setNames(seq_along(class_names), class_names)

    df_list <- lapply(person_dirs, function(person) {
      images <- list.files(person, pattern = "\\.jpg$", full.names = TRUE)
      label <- class_to_idx[[basename(person)]]
      data.frame(img_path = images, label = label, stringsAsFactors = FALSE)
    })

    df <- do.call(rbind, df_list)

    self$img_path <- df$img_path
    self$labels <- df$label
    self$classes <- class_names
    self$class_to_idx <- class_to_idx

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  download = function() {

    if (self$check_exists()) {
      return()
    }

    fs::dir_create(self$root)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    res <- self$resources[[self$split]]
    filename <- res[1]
    expected_md5 <- res[2]

    url <- file.path(self$base_url, filename)
    archive <- download_and_cache(url, prefix = class(self)[1])
    actual_md5 <- tools::md5sum(archive)

    if (actual_md5 != expected_md5) {
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }

    untar(archive, exdir = self$root)

    if (class(self)[[1]] == "lfw_pairs") {
      for (name in c("pairsDevTrain.txt", "pairsDevTest.txt", "pairs.txt")) {
        res <- self$resources[[name]]
        filename <- res[1]
        expected_md5 <- res[2]
        url <- file.path(self$base_url, filename)
        archive <- download_and_cache(url, prefix = class(self)[1])
        actual_md5 <- tools::md5sum(archive)

        if (actual_md5 != expected_md5) {
          runtime_error("Corrupt file! Delete the file in {archive} and try again.")
        }
        dest_path <- file.path(self$root, name)
        fs::file_move(archive, dest_path)
      }
    }

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  check_exists = function() {
    fs::dir_exists(file.path(self$root, self$split_name))
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
  archive_size_table = list(
    "original" = "170 MB",
    "funneled" = "230 MB"
  ),
  base_url = "https://ndownloader.figshare.com/files/",
  resources = list(
    original = c("5976018", "a17d05bd522c52d84eca14327a23d494"),
    funneled = c("5976015", "1b42dfed7d15c9b2dd63d5e5840c86ad"),
    pairsDevTrain.txt = c("5976012", "4f27cbf15b2da4a85c1907eb4181ad21"),
    pairsDevTest.txt = c("5976009", "5132f7440eb68cf58910c8a45a2ac10b"),
    pairs.txt = c("5976006", "9f1ba174e4e1c508ff7cdf10ac338a7d")
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    split = "original",
    target_transform = NULL,
    download = FALSE
  ) {
    self$root <- root
    self$train <- train
    self$transform <- transform
    self$target_transform <- target_transform
    self$classes <- c("Same", "Different")
    self$split <- split
    if (split == "original"){
      self$split_name <- "lfw"
    } else {
      self$split_name <- "lfw_funneled"
    }
    self$archive_size <- self$archive_size_table[[split]]

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (train) {
      pair_file <- file.path(root, "pairsDevTrain.txt")
    } else {
      pair_file <- file.path(root, "pairsDevTest.txt")
    }

    lines <- readLines(pair_file)
    lines <- lines[-1]  
    pair_list <- lapply(lines, function(line) {
      parts <- strsplit(line, "\\s+")[[1]]
      if (length(parts) == 3) {
        name <- parts[1]
        num1 <- as.integer(parts[2])
        num2 <- as.integer(parts[3])
        img1 <- file.path(name, sprintf("%s_%04d.jpg", name, num1))
        img2 <- file.path(name, sprintf("%s_%04d.jpg", name, num2))
        label <- 1
      } else {
        name1 <- parts[1]
        num1 <- as.integer(parts[2])
        name2 <- parts[3]
        num2 <- as.integer(parts[4])
        img1 <- file.path(name1, sprintf("%s_%04d.jpg", name1, num1))
        img2 <- file.path(name2, sprintf("%s_%04d.jpg", name2, num2))
        label <- 2
      }
      data.frame(img1 = img1, img2 = img2, label = label, stringsAsFactors = FALSE)
    })

    self$pairs <- do.call(rbind, pair_list)
    self$img_path <- c(self$pairs$img1, self$pairs$img2)

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  .getitem = function(index) {
    pair <- self$pairs[index, ]

    img1_path <- file.path(self$root, self$split_name, pair$img1)
    img2_path <- file.path(self$root, self$split_name, pair$img2)

    x1 <- jpeg::readJPEG(img1_path)
    x2 <- jpeg::readJPEG(img2_path)
    x <- list(x1, x2)
    y <- as.integer(pair$label)

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
    required_files <- c("pairs.txt", "pairsDevTest.txt", "pairsDevTrain.txt")
    fs::dir_exists(file.path(self$root, self$split_name)) && all(fs::file_exists(file.path(self$root, required_files)))
  }
)
