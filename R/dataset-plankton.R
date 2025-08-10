
#' WHOI-Plankton Dataset
#'
#' The WHOI-Plankton and WHOI-Plankton small are **image classification** datasets
#' composed of 957k and 58k grayscale images respectively, within classes
#' The images are in format with varying spatial resolutions.
#'
#' @inheritParams eurosat_dataset
#'
#' @return A torch dataset.
#' Each element is a named list:
#' - `x`: a H x W x 1 integer array representing an grayscale image.
#' - `y`: the class id of the image.
#'
#' @examples
#' \dontrun{
#' # Load the small plankton dataset
#' plankton <- whoi_small_plankton_dataset(download = TRUE)
#'
#' # Access the first item
#' first_item <- plankton[1]
#' first_item$x  # image array with shape {H, W, 1}
#' first_item$y  # id of the plankton class.
#'
#' # Load the full plankton dataset
#' plankton <- whoi_plankton_dataset(download = TRUE)
#'
#' # Access the first item
#' first_item <- plankton[1]
#' first_item$x  # image array with shape {H, W, 1}
#' first_item$y  # id of the plankton class.
#' }
#'
#' @name whoi_plankton_dataset
#' @title WHOI Plankton Datasets
#' @family classification_dataset
#' @export
whoi_small_plankton_dataset <- torch::dataset(
  name = "whoi_small_plankton",
  resources = data.frame(
    split = c("test", "train", "train", "val"),
    url = c("https://huggingface.co/datasets/nf-whoi/whoi-plankton-small/resolve/main/data/test-00000-of-00001.parquet?download=true",
            "https://huggingface.co/datasets/nf-whoi/whoi-plankton-small/resolve/main/data/train-00000-of-00002.parquet?download=true",
            "https://huggingface.co/datasets/nf-whoi/whoi-plankton-small/resolve/main/data/train-00001-of-00002.parquet?download=true",
            "https://huggingface.co/datasets/nf-whoi/whoi-plankton-small/resolve/main/data/validation-00000-of-00001.parquet?download=true"),
    md5 = c("91b7794e4286c138bda667121ce91d2c",
            "170921030aee26f9676725a0c55a4420",
            "332ebc8b822058cd98f778099927c50e",
            "f0747ae16fc7cd6946ea54c3fe1f30b4"),
    size = c("217 MB", "396 MB", "383 MB", "112 MB")
  ),

  initialize = function(
    split = "val",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$transform <- transform
    self$target_transform <- target_transform
    self$archive_url <- self$resources[self$resources$split == split,]$url
    self$archive_size <- self$resources[self$resources$split == split,]$size
    self$archive_md5 <- self$resources[self$resources$split == split,]$md5
    self$split_file <- sapply(self$archive_url, \(x) file.path(rappdirs::user_cache_dir("torch"), class(self)[1], sub("\\?download=.*", "", basename(x))))

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")

    if (!requireNamespace("arrow", quietly = TRUE)) {
      install.packages("arrow")
    }
    self$.data <- arrow::open_dataset(self$split_file)
    self$classes <- jsonlite::parse_json(self$.data$metadata$huggingface, simplifyVector = TRUE)$info$features$label$names
    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  download = function() {
    if (self$check_exists())
      return(NULL)

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    # Download the dataset parquet
    archive <- sapply(self$archive_url, \(x)  download_and_cache(x, prefix = class(self)[1]))
    if (!all(tools::md5sum(archive) == self$archive_md5))
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    for (file_id in seq_along(archive)) {
      fs::file_move(archive[file_id], self$split_file[file_id])
    }
  },
  check_exists = function() {
    all(sapply(self$split_file, \(x) fs::file_exists(x)))
  },

  .getitem = function(index) {
    df <- collect(self$.data[index,])
    x <- df$image$bytes %>% unlist() %>% as.raw() %>% png::readPNG()
    y <- df$label

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
  },

  .length = function() {
    self$.data$num_rows
  }

)


#' WHOI-Plankton Dataset
#'
#' @inheritParams whoi_plankton_dataset#'
#' @rdname whoi_plankton_dataset
#' @export
whoi_plankton_dataset <- torch::dataset(
  name = "whoi_plankton",
  inherit = whoi_small_plankton_dataset,
  archive_size = "4.1 GB",
  resources = list(
    c("https://uofi.app.box.com/shared/static/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl.gz", "985ac761bbb52ca49e0c474ae806c07c"),
    c("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip", "4fa8c08369d22fe16e41dc124bd1adc2")
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
    self$train <- train
    self$split <- ifelse(train, "train", "test")

    if (download)
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()

    if (!self$check_exists())
      cli_abort("Dataset not found. Use `download = TRUE` to download it.")

    captions_path <- file.path(self$raw_folder, "dataset_flickr30k.json")
    captions_json <- jsonlite::fromJSON(captions_path)

    imgs_df <- captions_json$images
    filtered <- imgs_df[imgs_df$split == self$split, ]
    self$filenames <- filtered$filename

    captions_map <- setNames(
      lapply(filtered$sentences, function(s) unname(vapply(s$raw, identity, character(1)))),
      filtered$filename
    )

    all_ids <- names(captions_map)
    caption_to_index <- setNames(seq_along(all_ids), all_ids)

    self$images <- file.path(self$raw_folder, "flickr30k-images", self$filenames)
    self$captions <- vapply(self$filenames, function(f) caption_to_index[[f]], integer(1))
    self$classes <- captions_map

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$images)} images across {length(self$classes)} classes.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$raw_folder, "dataset_flickr30k.json")) &&
    fs::dir_exists(file.path(self$raw_folder, "flickr30k-images"))
  },

  active = list(
    raw_folder = function() file.path(self$root, "flickr30k", "raw")
  )
)
