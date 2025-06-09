lfw_people_dataset <- dataset(
  name = "lfw_people",
  resources = list(
    list(
      url = "https://huggingface.co/datasets/vilsonrodrigues/lfw/resolve/main/lfw_multifaces-ingestion.zip",
      md5 = "266d95cb38eb91c44262dd8df6b471af"
    ),
    list(
      url = "https://huggingface.co/datasets/vilsonrodrigues/lfw/resolve/main/lfw_multifaces-retrieval.zip",
      md5 = "736e29ffaca4e20f8f954e24dde030a5"
    ),
    list(
      url = "https://huggingface.co/datasets/JimmyUnleashed/LFW/resolve/main/lfw-names-ingestion.txt",
      md5 = "e37633a162015b0cf4a16f0213d1c9aa"
    ),
    list(
      url = "https://huggingface.co/datasets/JimmyUnleashed/LFW/resolve/main/lfw-names-retrieval.txt",
      md5 = "e37633a162015b0cf4a16f0213d1c9aa"
    )
  ),
  training_file = "training.rds",
  test_file = "test.rds",
  initialize = function(root, split = "train", image_set = "deepfunneled", transform = NULL, target_transform = NULL, download = FALSE) {
    self$root_path <- root
    self$split <- split
    self$image_set <- image_set
    self$transform <- transform
    self$target_transform <- target_transform
    if (download)
      self$download()
    if (!self$check_exists())
      rlang::abort("Dataset not found. Use `download = TRUE` to download it.")
    file <- if (split == "train") self$training_file else self$test_file
    data <- readRDS(file.path(self$processed_folder, file))
    self$data <- data[[1]]
    self$targets <- data[[2]]
    self$class_names <- data[[3]]
  },
  download = function() {
    rlang::inform(glue::glue("Downloading LFW People dataset split = {self$split}, image_set = {self$image_set}..."))
    if (self$check_exists())
      return(NULL)
    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)
    downloaded <- list()
    for (res in self$resources) {
      zip_path <- download_and_cache(res$url, prefix = class(self)[1])
      local_md5 <- unname(tools::md5sum(zip_path))
      if (!is.null(res$md5) && local_md5 != res$md5) {
        rlang::abort(glue::glue("MD5 checksum failed for {basename(res$url)}"))
      }
      fs::file_copy(zip_path, file.path(self$raw_folder, basename(res$url)), overwrite = TRUE)
      downloaded[[basename(res$url)]] <- zip_path
    }
    utils::unzip(downloaded[["lfw_multifaces-ingestion.zip"]], exdir = self$raw_folder)
    utils::unzip(downloaded[["lfw_multifaces-retrieval.zip"]], exdir = self$raw_folder)
    rlang::inform(glue::glue("Processing LFW People dataset split = {self$split}, image_set = {self$image_set}...(This will take a while)..."))
    dir_name <- if (self$split == "train") "lfw_multifaces-ingestion" else "lfw_multifaces-retrieval"
    img_dir <- file.path(self$raw_folder, dir_name)
    label_file <- if (self$split == "train") "lfw-names-ingestion.txt" else "lfw-names-retrieval.txt"
    label_path <- file.path(self$raw_folder, label_file)
    names <- readLines(label_path)
    class_names <- sort(unique(names))
    class_to_idx <- setNames(seq_along(class_names), class_names)
    all_imgs <- fs::dir_ls(img_dir, regexp = "\\.jpg$", recurse = FALSE)
    imgs <- list()
    targets <- list()
    for (img_path in all_imgs) {
      file_name <- fs::path_file(img_path)
      name <- sub("_[0-9]{4}\\.jpg$", "", file_name)
      class_idx <- class_to_idx[[name]]
      if (is.null(class_idx))
        next
      imgs[[length(imgs) + 1]] <- img_path
      targets[[length(targets) + 1]] <- class_idx
    }
    imgs_tensor <- array(NA_real_, dim = c(length(imgs), 250, 250, 3))
    for (i in seq_along(imgs)) {
      img_array <- jpeg::readJPEG(imgs[[i]])
      if (length(dim(img_array)) == 2)
        img_array <- array(rep(img_array, each = 3), dim = c(dim(img_array), 3))
      imgs_tensor[i,,,] <- img_array
    }
    targets_tensor <- as.integer(unlist(targets))
    saveRDS(list(imgs_tensor, targets_tensor, class_names), file.path(self$processed_folder, self$training_file))
    saveRDS(list(imgs_tensor, targets_tensor, class_names), file.path(self$processed_folder, self$test_file))
    rlang::inform(glue::glue("LFW People dataset split = {self$split}, image_set = {self$image_set} Processed Successfully !"))
  },
  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, self$training_file)) &&
      fs::file_exists(file.path(self$processed_folder, self$test_file))
  },
  .getitem = function(index) {
    img <- self$data[index,,,]
    target_idx <- self$targets[index]
    target <- self$class_names[target_idx]
    if (!is.null(self$transform))
      img <- self$transform(img)
    if (!is.null(self$target_transform))
      target_idx <- self$targets[index]
      target <- self$class_names[target_idx]
    list(x = img, y = target)
  },
  .length = function() {
    dim(self$data)[1]
  },
  active = list(
    raw_folder = function() file.path(self$root_path, "lfw_people", "raw"),
    processed_folder = function() file.path(self$root_path, "lfw_people", "processed"),
    classes = function() self$class_names
  )
)
