
#' WHOI-Plankton Dataset
#'
#' The WHOI-Plankton and WHOI-Plankton small are **image classification** datasets
#' of submarine plankton small grayscale images of varying size, classified into 100 classes.
#' Dataset size is 957k and 58k respectively, and each provides a train / val / test split.
#'
#' @inheritParams eurosat_dataset
#'
#' @return A torch dataset with a
#' - `classes` attribute providing the vector of class names.
#'
#' Each element is a named list:
#' - `x`: a H x W x 1 integer array representing an grayscale image.
#' - `y`: the class id of the image.
#'
#' @examples
#' \dontrun{
#' # Load the small plankton dataset and turn images into tensor images
#' plankton <- whoi_small_plankton_dataset(download = TRUE, transform = transform_to_tensor)
#'
#' # Access the first item
#' first_item <- plankton[1]
#' first_item$x  # a tensor grayscale image with shape {1, H, W}
#' first_item$y  # id of the plankton class.
#' plankton$classes[first_item$y] # name of the plankton class
#'
#' # Load the full plankton dataset
#' plankton <- whoi_plankton_dataset(download = TRUE)
#'
#' # Access the first item
#' first_item <- plankton[1]
#' first_item$x  # grayscale image array with shape {H, W}
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
    size = c(217e6, 396e6, 383e6, 112e6)
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
    self$archive_size <- prettyunits::pretty_bytes(sum(self$resources[self$resources$split == split,]$size))
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
    y <- df$label + 1L

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
  archive_size = "9.1 GB",
  resources = data.frame(
    split = c(rep("test", 4), rep("train", 13), rep("val", 2)),
    url = c(paste0("https://huggingface.co/datasets/nf-whoi/whoi-plankton/resolve/main/data/test-0000",0:3,"-of-00004.parquet?download=true"),
            paste0("https://huggingface.co/datasets/nf-whoi/whoi-plankton/resolve/main/data/train-0000",0:9,"-of-00013.parquet?download=true"),
            paste0("https://huggingface.co/datasets/nf-whoi/whoi-plankton/resolve/main/data/train-000",10:12,"-of-00013.parquet?download=true"),
            paste0("https://huggingface.co/datasets/nf-whoi/whoi-plankton/resolve/main/data/validation-0000",0:1,"-of-00002.parquet?download=true")),
    md5 = c("cd41b344ec4b6af83e39c38e19f09190",
            "aa0965c0e59f7b1cddcb3c565d80edf3",
            "b2a75513f1a084724e100678d8ee7180",
            "a03c4d52758078bfb0799894926d60f6",
            "07eaff140f39868a8bcb1d3c02ebe60f",
            "87c927b9fbe0c327b7b9ae18388b4fcf",
            "456efd91901571a41c2157732880f6b8",
            "dc929fde45e3b2e38bdd61a15566cf32",
            "f92ab6cfb4a3dd7e0866f3fdf8dbc33c",
            "61c555bba39b6b3ccb4f02a5cf07e762",
            "57e03cecf2b5d97912ed37e1b8fc6263",
            "56081cc99e61c36e89db1566dbbf06c1",
            "60b7998630468cb18880660c81d1004a",
            "1fa94ceb54d4e53643a0d8cf323af901",
            "7a7be4e3dfdc39a50c8ca086a4d9a8de",
            "07194caf75805e956986cba68e6b398e",
            "0f4d47f240cd9c30a7dd786171fa40ca",
            "db827a7de8790cdcae67b174c7b8ea5e",
            "d3181d9ffaed43d0c01f59455924edca"),
    size = c(rep(450e6, 4), rep(490e6, 13), rep(450e6, 2))
  )
)
