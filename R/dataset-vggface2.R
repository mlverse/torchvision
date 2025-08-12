#' VGGFace2 Dataset
#'
#' The VGGFace2 dataset is a large-scale face recognition dataset containing images
#' of celebrities from a wide range of ethnicities, professions, and ages.
#' Each identity has multiple images with large variations in pose, age, illumination,
#' ethnicity, and profession.
#'
#' @inheritParams oxfordiiitpet_dataset
#' @param root Character. Root directory where the dataset will be stored under `root/vggface2`.
#'
#' @return A torch dataset object `vggface2_dataset`:
#' - `x`: RGB image array.
#' - `y`: Integer label (1…N) for the identity.
#' 
#' `ds$classes` is a named list mapping integer labels to a list with:
#' - `name`: Character name of the person.
#' - `gender`: "Male" or "Female".
#'
#' @examples
#' \dontrun{
#' #Load the training set
#' ds <- vggface2_dataset(download = TRUE)
#' item <- ds[1]
#' item$x      # image tensor
#' item$y      # integer label
#' ds$classes[item$y]  # list(name=..., gender=...)
#'
#' #Load the test set
#' ds <- vggface2_dataset(download = TRUE, train = FALSE)
#' item <- ds[1]
#' item$x      # image tensor
#' item$y      # integer label
#' ds$classes[item$y]  # list(name=..., gender=...)
#' }
#'
#' @family segmentation_dataset
#' @export
vggface2_dataset <- torch::dataset(
  name = "vggface2",
  resources = list(
    train_images = "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/data/vggface2_train.tar.gz",
    test_images  = "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/data/vggface2_test.tar.gz",
    train_list   = "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/meta/train_list.txt",
    test_list    = "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/meta/test_list.txt",
    identity     = "https://huggingface.co/datasets/ProgramComputer/VGGFace2/raw/main/meta/identity_meta.csv"
  ),
  archive_size = "38 GB",
  training_file = "train.rds",
  test_file = "test.rds",

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$root_path <- root
    self$train <- train
    self$transform <- transform
    self$target_transform <- target_transform
    if (train) {
      self$split <- "train"
      self$archive_size <- "36 GB"
    } else {
      self$split <- "test"
      self$archive_size <- "2 GB"
    }

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    if (train) {
      data_file <- self$training_file
    } else {
      data_file <- self$test_file
    }
    data <- readRDS(file.path(self$processed_folder, data_file))

    self$img_path <- data$img_path
    self$labels <- data$labels
    self$classes <- data$classes

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  download = function() {
    if (self$check_exists()) {
      return()
    }

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    download_and_extract <- function(url, exdir) {
      archive <- download_and_cache(url, prefix = class(self)[1])
      utils::untar(archive, exdir = exdir)
    }

    download_and_extract(self$resources$train_images, self$raw_folder)
    download_and_extract(self$resources$test_images, self$raw_folder)

    train_list_file <- file.path(self$raw_folder, "train_list.txt")
    download.file(self$resources$train_list, train_list_file, mode = "wb")
    test_list_file <- file.path(self$raw_folder, "test_list.txt")
    download.file(self$resources$test_list, test_list_file, mode = "wb")

    identity_file <- file.path(self$raw_folder, "identity_meta.csv")
    download.file(self$resources$identity, identity_file, mode = "wb")

    identity_df <- read.csv(identity_file, sep = ",", stringsAsFactors = FALSE, strip.white = TRUE)
    identity_df$Class_ID <- trimws(identity_df$Class_ID)
    identity_map <- setNames(
      lapply(seq_len(nrow(identity_df)), function(i) {
        if(identity_df$Gender[i] == 'f'){
          gender <- "Female"
        }else{
          gender <- "Male"
        }
        list(
          name = identity_df$Name[i],
          gender = gender
        )
      }),
      identity_df$Class_ID
    )

    for (split in c("train", "test")) {
        if (split == "train") {
          list_file <- train_list_file
        } else {
          list_file <- test_list_file
        }
        files <- readLines(list_file)

        img_path <- file.path(self$raw_folder, split, files)
        class_ids <- sub("/.*$", "", files)
        unique_ids <- unique(class_ids)

        class_to_idx <- setNames(seq_along(unique_ids), unique_ids)

        labels <- as.integer(class_to_idx[class_ids])

        classes_list <- lapply(unique_ids, function(cid) {
            identity_map[[cid]]
        })

        saveRDS(
            list(
            img_path = img_path,
            labels = labels,
            classes = classes_list
            ),
            file.path(self$processed_folder, paste0(split, ".rds"))
        )
    }

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
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
  },

  active = list(
    raw_folder = function() {
      file.path(self$root_path, "vggface2", "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, "vggface2", "processed")
    }
  )
)