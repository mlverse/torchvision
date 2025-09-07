#' @include folder-dataset.R
NULL

#' RoboFlow 100 Document dataset Collection
#'
#' Loads one of the [RoboFlow 100 Document](https://universe.roboflow.com/browse/documents) datasets with COCO-style
#' bounding box annotations for object detection tasks.
#'
#' @param dataset Dataset to select within \code{c("tweeter_post", "tweeter_profile", "document_part",
#'   "activity_diagram", "signature", "paper_part", "tabular_data", "paragraph")}.
#' @param split the subset of the dataset to choose between \code{c("train", "test", "valid")}.
#' @param download Logical. If TRUE, downloads the dataset if not present at `root`.
#' @param transform Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target.
#'
#' @return A torch dataset. Each element is a named list with:
#' - `x`: H x W x 3 array representing the image.
#' - `y`: a list containing the target with:
#'     - `labels`: character vector of object class names.
#'     - `boxes`: a tensor of shape (N, 4) with bounding boxes, if any, in \eqn{(x_{min}, y_{min}, x_{max}, y_{max})} format.
#'
#' The returned item inherits the class `image_with_bounding_box` so it can be
#' visualised with helper functions such as [draw_bounding_boxes()].
#'
#' @examples
#' \dontrun{
#' ds <- rf100_document_collection(
#'   dataset = "tweeter_post",
#'   split = "train",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#'
#' # Retrieve a sample and inspect annotations
#' item <- ds[1]
#' item$y$labels
#' item$y$boxes
#'
#' # Draw bounding boxes and display the image
#' boxed_img <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed_img)
#' }
#'
#' @name rf100_document_collection
#' @title RF100 Document Collection Datasets
#' @family detection_dataset
#' @export
rf100_document_collection <- torch::dataset(
  name = "rf100_document_collection",

  resources = data.frame(
    dataset = rep(c("tweeter_post", "tweeter_profile", "document_part",
                    "activity_diagram", "signature", "paper_part"), each = 3),
    split   = rep(c("train", "test", "valid"), times = 6),
    url = c(
      # tweeter_post
      "https://huggingface.co/datasets/Francesco/tweeter-posts/resolve/main/data/train-00000-of-00001-5ca0e754c63f9a31.parquet",
      "https://huggingface.co/datasets/Francesco/tweeter-posts/resolve/main/data/test-00000-of-00001-489e49c5dfdce787.parquet",
      "https://huggingface.co/datasets/Francesco/tweeter-posts/resolve/main/data/validation-00000-of-00001-2ab456c0d1f04f82.parquet",

      # tweeter_profile
      "https://huggingface.co/datasets/Francesco/tweeter-profile/resolve/main/data/train-00000-of-00001-1c7085071a0f4972.parquet",
      "https://huggingface.co/datasets/Francesco/tweeter-profile/resolve/main/data/test-00000-of-00001-be2dd504b504117f.parquet",
      "https://huggingface.co/datasets/Francesco/tweeter-profile/resolve/main/data/validation-00000-of-00001-747ade4e297e090a.parquet",

      # document_part
      "https://huggingface.co/datasets/Francesco/document-parts/resolve/main/data/train-00000-of-00001-5503c1fa031a4929.parquet",
      "https://huggingface.co/datasets/Francesco/document-parts/resolve/main/data/test-00000-of-00001-6cb74e4a35ca2ba5.parquet",
      "https://huggingface.co/datasets/Francesco/document-parts/resolve/main/data/validation-00000-of-00001-53417fa849d940f8.parquet",

      # activity_diagram
      "https://huggingface.co/datasets/Francesco/activity-diagrams-qdobr/resolve/main/data/train-00000-of-00001-9c2ac6dd4a9e53d8.parquet",
      "https://huggingface.co/datasets/Francesco/activity-diagrams-qdobr/resolve/main/data/test-00000-of-00001-acf5b67e3c7ca657.parquet",
      "https://huggingface.co/datasets/Francesco/activity-diagrams-qdobr/resolve/main/data/validation-00000-of-00001-d6f6f66b7dc88280.parquet",

      # signature
      "https://huggingface.co/datasets/Francesco/signatures-xc8up/resolve/main/data/train-00000-of-00001-aab07332622fb759.parquet",
      "https://huggingface.co/datasets/Francesco/signatures-xc8up/resolve/main/data/test-00000-of-00001-acf5b67e3c7ca657.parquet",
      "https://huggingface.co/datasets/Francesco/signatures-xc8up/resolve/main/data/validation-00000-of-00001-2cd116e72c9571b0.parquet",

      # paper_part
      "https://huggingface.co/datasets/Francesco/paper-parts/resolve/main/data/train-00000-of-00001-0f677be56de6ff94.parquet",
      "https://huggingface.co/datasets/Francesco/paper-parts/resolve/main/data/test-00000-of-00001-94db1ab1c191f5e2.parquet",
      "https://huggingface.co/datasets/Francesco/paper-parts/resolve/main/data/validation-00000-of-00001-2ce552e0b2a0aac5.parquet"
    ),

    md5 = c(
      # tweeter_post
      "30f8ee708cdfc443bfa8f9bc1d89e3b2",
      "00d694afad4f384d37fadbc9325a16ad",
      "14b26370147438e5c7c9b9de246d6891",
      # tweeter_profile
      "9a5fe681eded1fc8a08975a5ed142b24",
      "b54028739024b34e2acda860eb6a8068",
      "1eb31069867c3f855bda0aa269bb1eda",
      # document_part
      "5181f82eb8f91d92dd225dd23387b5e5",
      "a8bb1bd010ece905acb8a3ec2851de93",
      "85a20f23b6d53fb6a5148b125dd3ec4c",
      # activity_diagram
      "340249bc764ebcd4c00c243fdd75b773",
      "04d07c012caf23e643c4e28f15d43f83",
      "e2d3050bd5fed14664e71444e8df2ab9",
      # signature
      "742e26e8ee5d4d8605da68dda4df3c62",
      "e07083621e8b1cf73f6daea8f93e8943",
      "46836f6ebe462605e37b4640c0cc336d",
      # paper_part
      "253fd2189e89eb4664e89e9ed4b08dcc",
      "23cebd238571b4d130a11a6df760a180",
      "e02e6da02d92de11283739c5b0daeb4b"
    ),

    size = rep(50e6, 18)  # placeholder; optional
  ),

  initialize = function(
    dataset, split = c("train", "test", "valid"),
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")
    if (!requireNamespace("prettyunits", quietly = TRUE)) install.packages("prettyunits")

    self$dataset <- match.arg(dataset, self$resources$dataset)
    self$split   <- match.arg(split)
    self$transform <- transform
    self$target_transform <- target_transform

    sel <- self$resources$dataset == self$dataset & self$resources$split == self$split
    self$archive_url  <- self$resources$url[sel]
    self$archive_size <- prettyunits::pretty_bytes(sum(self$resources$size[sel]))
    self$archive_md5  <- self$resources$md5[sel]
    self$split_file   <- file.path(
      rappdirs::user_cache_dir("torch"),
      class(self)[1], self$dataset,
      sub("\\?download=.*", "", basename(self$archive_url))
    )

    if (download) {
      cli_inform("Dataset {.val {self$dataset}} split {.val {self$split}} of {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists())
      runtime_error("Dataset not found. Use download=TRUE or check that parquet files exist at the expected paths.")

    ads <- arrow::open_dataset(self$split_file)
    self$classes <- jsonlite::parse_json(ads$metadata$huggingface, simplifyVector = TRUE)$info$features$objects$feature$category$names

    if (sum(sel) == 1) {
      self$.data <- arrow::read_parquet(self$split_file)
    } else {
      self$.data <- ads$to_data_frame()
    }

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images for {.val {self$dataset}}.")
  },

  download = function() {
    if (self$check_exists()) return(invisible(NULL))
    cli_inform("Downloading {.val {self$dataset}}...")

    archive <- sapply(self$archive_url, download_and_cache, prefix = file.path(class(self)[1], self$dataset))

    if (!all(tools::md5sum(archive) == self$archive_md5))
      runtime_error("Corrupt file! Delete the cached files and try again.")

    for (i in seq_along(archive)) fs::file_move(archive[i], self$split_file[i])
  },

  check_exists = function() all(fs::file_exists(self$split_file)),

  .getitem = function(index) {
    df <- self$.data[index, ]
    x_raw <- unlist(df$image$bytes) |> as.raw()
    if (tolower(tools::file_ext(df$image$path)) == "jpg") {
      x <- jpeg::readJPEG(x_raw)
    } else {
      x <- png::readPNG(x_raw)
    }
    if (length(dim(x)) == 3 && dim(x)[3] == 4) x <- x[, , 1:3, drop = FALSE]

    if (!is.null(df$objects) && length(df$objects[[1]]) > 0) {
      bbox <- df$objects$bbox
      if (is.list(bbox)) {
        bbox <- do.call(rbind, bbox[[1]])
      }
      boxes  <- torch::torch_tensor(bbox, dtype = torch::torch_float())
      labels <- df$objects$category[[1]]
      if (is.null(labels)) {
        labels <- df$objects$label[[1]]
      }
      # labels <- as.character(unlist(labels))
    } else {
      boxes  <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- 0L
    }

    y <- list(labels = labels, boxes = boxes)
    if (!is.null(self$transform)) x <- self$transform(x)
    if (!is.null(self$target_transform)) y <- self$target_transform(y)

    item <- list(x = x, y = y)
    class(item) <- "image_with_bounding_box"
    item
  },

  .length = function() nrow(self$.data)
)
