#' VGGFace2 Dataset
#'
#' The VGGFace2 dataset is a large-scale face recognition dataset containing images
#' of celebrities from a wide range of ethnicities, professions, and ages.
#' Each identity has multiple images with variations in context, pose, age, and illumination.
#'
#' @inheritParams eurosat_dataset
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
#' item$x      # image array RGB
#' item$y      # integer label
#' ds$classes[item$y]  # list(name=..., gender=...)
#'
#' #Load the test set
#' ds <- vggface2_dataset(download = TRUE, train = FALSE)
#' item <- ds[1]
#' item$x      # image array RGB
#' item$y      # integer label
#' ds$classes[item$y]  # list(name=..., gender=...)
#' }
#'
#' @family classification_dataset
#' @export
vggface2_dataset <- torch::dataset(
  name = "vggface2",
  resources = data.frame(
    split = c("train", "train", "test", "test", "identity"),
    kind = c("image", "label", "image", "label", "label"),
    url = c("https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/data/vggface2_train.tar.gz",
            "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/meta/train_list.txt",
            "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/data/vggface2_test.tar.gz",
            "https://huggingface.co/datasets/ProgramComputer/VGGFace2/resolve/main/meta/test_list.txt",
            "https://huggingface.co/datasets/ProgramComputer/VGGFace2/raw/main/meta/identity_meta.csv"),
    md5 = c("88813c6b15de58afc8fa75ea83361d7f",
            "4cfbab4a839163f454d7ecef28b68669",
            "bb7a323824d1004e14e00c23974facd3",
            "d08b10f12bc9889509364ef56d73c621",
            "d315386c7e8e166c4f60e27d9cc61acc"),
    size = c(37.9e9, 62e6, 2.03e9, 3e6, 335e3)
  ),
  initialize = function(
    root = tempdir(),
    split = "val",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$root_path <- root
    self$split <- match.arg(split, c("train", "test"))
    self$transform <- transform
    self$target_transform <- target_transform
    self$archive_url <- self$resources[self$resources$split %in% c(split, "identity"),]$url
    self$archive_size <- prettyunits::pretty_bytes(sum(self$resources[self$resources$split %in% c(split, "identity"),]$size))
    self$archive_md5 <- self$resources[self$resources$split %in% c(split, "identity"),]$md5
    self$split_file <- sapply(self$archive_url, function(x) file.path(rappdirs::user_cache_dir("torch"), class(self)[1], basename(x)))

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }


    data <- readRDS(file.path(self$processed_folder, glue::glue("{self$split}.rds")))

    self$img_path <- data$img_path
    self$labels <- data$labels
    self$class_to_idx <- data$class_to_idx
    self$identity_df <- data$identity_df

    cli_inform("Split {.val {self$split}} of dataset {.cls {class(self)[[1]]}} loaded with {self$.length()} samples.")
  },

  download = function() {
    if (self$check_exists())
      return(NULL)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    # Download and extract archives
    archive <- sapply(self$archive_url, function(x)  download_and_cache(x, prefix = class(self)[1]))
    if (!all(tools::md5sum(archive) == self$archive_md5))
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    for (file_id in seq_along(archive)) {
      fs::file_move(archive[file_id], self$split_file[file_id])
    }
    utils::untar(archive[[1]], exdir = self$raw_folder)

    identity_df <- read.csv(archive[[3]], sep = ",", stringsAsFactors = FALSE, strip.white = TRUE)
    identity_df$Class_ID <- trimws(identity_df$Class_ID)
    identity_df$Gender <- factor(identity_df$Gender, labels = c("Female", "Male"))

    files <- read.delim(archive[[2]], sep = "/", col.names = c("Class_ID", "img_path"), header = F)
    files$img_path <- file.path(self$raw_folder, self$split, files$Class_ID, files$img_path)
    class_to_idx <- setNames(seq_len(nlevels(as.factor(files$Class_ID))), levels(as.factor(files$Class_ID)))
    labels <- names(class_to_idx)
    saveRDS(list(img_path = files,identity_df = identity_df, labels = labels,
                   class_to_idx = class_to_idx),
              file.path(self$processed_folder, glue::glue("{self$split}.rds"))
      )

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  check_exists = function() {
    all(fs::file_exists(self$split_file)) &&
      fs::file_exists(file.path(self$processed_folder, glue::glue("{self$split}.rds")))
  },

  .getitem = function(index) {
    x <- jpeg::readJPEG(self$img_path$img_path[index])
    y <- self$class_to_idx[self$img_path$Class_ID[index]]

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }
    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }
    list(x = x, y = y)
  },

  .length = function() {
    nrow(self$img_path)
  },

  active = list(
    raw_folder = function() {
      fs::dir_create(file.path(self$root_path, class(self)[1], "raw"))
      file.path(self$root_path, class(self)[1], "raw")
    },
    processed_folder = function() {
      fs::dir_create(file.path(self$root_path, class(self)[1], "processed"))
      file.path(self$root_path, class(self)[1], "processed")
    }
  )
)
