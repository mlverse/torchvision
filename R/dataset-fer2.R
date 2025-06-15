fer_dataset <- dataset(
  name = "fer_dataset",

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
    self$split <- if (train) "Train" else "Test"

    self$folder_name <- "fer2013"
    self$url <- "https://huggingface.co/datasets/JimmyUnleashed/FER-2013/resolve/main/fer2013.tar.gz"
    self$md5 <- "ca95d94fe42f6ce65aaae694d18c628a"

    self$classes <- c("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
    self$class_to_idx <- setNames(seq_along(self$classes), self$classes)

    rlang::inform(glue::glue("Preparing FER-2013 dataset ({self$split})..."))

    if (download) {
      self$download()
    }

    if (!self$check_files()) {
      runtime_error("Dataset files missing. Use download = TRUE to fetch them.")
    }

    csv_file <- file.path(self$root, self$folder_name, "fer2013.csv")
    parsed <- read.csv(csv_file, stringsAsFactors = FALSE)

    if (self$train) {
      parsed <- parsed[parsed$Usage == "Training", ]
    } else {
      parsed <- parsed[parsed$Usage %in% c("PublicTest", "PrivateTest"), ]
    }

    rlang::inform("Parsing image data into tensors (1x48x48 per sample)...")
    self$x <- lapply(parsed$pixels, function(pixels) {
      vals <- as.integer(strsplit(pixels, " ")[[1]])
      torch_tensor(vals, dtype = torch_uint8())$view(c(1, 48, 48))
    })

    self$y <- self$classes[as.integer(parsed$emotion) + 1L]

    file_size <- fs::file_info(csv_file)$size
    readable <- fs::fs_bytes(file_size)

    rlang::inform(glue::glue(
      "FER-2013 ({self$split}) loaded: {length(self$x)} images (~{readable}), 48x48 grayscale, {length(self$classes)} classes."
    ))
  },

  .getitem = function(i) {
    x <- self$x[[i]]
    y <- self$y[i]

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }

    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    list(x = x, y = y)
  },

  .length = function() {
    length(self$y)
  },

  get_classes = function() {
    self$classes
  },

  download = function() {
    if (self$check_files()) {
      rlang::inform("FER-2013 already exists. Skipping download.")
      return()
    }

    dest_dir <- file.path(self$root, self$folder_name)
    fs::dir_create(dest_dir)

    rlang::inform(glue::glue("Downloading FER-2013 dataset..."))

    archive_path <- download_and_cache(self$url)
    
    if (!tools::md5sum(archive_path) == self$md5)
      runtime_error("Corrupt file! Delete the file in {p} and try again.")

    rlang::inform("Extracting dataset...")
    untar(archive_path, exdir = self$root)
    rlang::inform("Extraction complete.")
  },

  check_files = function() {
    file.exists(file.path(self$root, self$folder_name, "fer2013.csv"))
  }
)
