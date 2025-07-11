#' @export
pascal_segmentation_dataset <- torch::dataset(
  name = "pascal_segmentation_dataset",

  resources = list(
    `2007` = list(
      trainval = list(
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        md5 = "c52e279531787c972589f7e41ab4ae64"
      ),
      test = list(
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        md5 = "b6e924de25625d8de591ea690078ad9f"
      )
    ),
    `2008` = list(
      trainval = list(
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        md5 = "2629fa636546599198acfcfbfcf1904a"
      )
    ),
    `2009` = list(
      trainval = list(
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        md5 = "a3e00b113cfcfebf17e343f59da3caa1"
      )
    ),
    `2010` = list(
      trainval = list(
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        md5 = "da459979d0c395079b5c75ee67908abb"
      )
    ),
    `2011` = list(
      trainval = list(
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        md5 = "6c3384ef61512963050cb5d687e5bf1e"
      )
    ),
    `2012` = list(
      trainval = list(
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        md5 = "6cd6e144f989b92b3379bac3b3de84fd"
      )
    )
  ),

  archive_size_table = list(
    "2007" = list(trainval = "430 MB", test = "440 MB"),
    "2008" = list(trainval = "550 MB"),
    "2009" = list(trainval = "890 MB"),
    "2010" = list(trainval = "1.3 GB"),
    "2011" = list(trainval = "1.7 GB"),
    "2012" = list(trainval = "1.9 GB")
  ),

  initialize = function(
    root = tempdir(),
    year = "2012",
    split = "train",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {
    self$root_path <- root
    self$year <- match.arg(year, choices = names(self$resources))
    self$split <- match.arg(split, choices = c("train", "val", "trainval", "test"))
    self$transform <- transform
    self$target_transform <- target_transform

    archive_key <- if (self$split == "test") "test" else "trainval"
    self$archive_size <- self$archive_size_table[[self$year]][[archive_key]]

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Pascal VOC dataset for year {self$year} not found. Use `download = TRUE` to fetch it.")
    }

    data_file <- file.path(self$processed_folder, paste0(self$split, ".rds"))
    data <- readRDS(data_file)
    self$img_paths <- data$img_paths
    self$mask_paths <- data$mask_paths
  },

  download = function() {
    if (self$check_exists()) return()

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    archive_key <- if (self$split == "test") "test" else "trainval"
    resource <- self$resources[[self$year]][[archive_key]]
    archive <- download_and_cache(resource$url, prefix = class(self)[1])
    actual_md5 <- tools::md5sum(archive)
    if (actual_md5 != resource$md5) {
      cli_abort("Downloaded archive has incorrect MD5. Please delete and retry.")
    }

    utils::untar(archive, exdir = self$raw_folder)

    voc_dir <- file.path(self$raw_folder, "VOCdevkit", paste0("VOC", self$year))
    voc_root <- self$raw_folder
    if (self$year == "2011") {
    voc_root <- file.path(voc_root, "TrainVal")
    }
    voc_dir <- file.path(voc_root, "VOCdevkit", paste0("VOC", self$year))

    split_file <- file.path(voc_dir, "ImageSets", "Segmentation", paste0(self$split, ".txt"))

    if (!fs::file_exists(split_file)) {
      cli_abort("Split file not found: {split_file}")
    }

    ids <- readLines(split_file)
    img_paths <- file.path(voc_dir, "JPEGImages", paste0(ids, ".jpg"))
    mask_paths <- file.path(voc_dir, "SegmentationClass", paste0(ids, ".png"))

    saveRDS(list(
      img_paths = img_paths,
      mask_paths = mask_paths
    ), file.path(self$processed_folder, paste0(self$split, ".rds")))

    cli_inform("Pascal VOC {self$year} dataset extracted and preprocessed.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, paste0(self$split, ".rds")))
  },

  .getitem = function(index) {
    img_path <- self$img_paths[index]
    mask_path <- self$mask_paths[index]

    x <- jpeg::readJPEG(img_path)
    y <- png::readPNG(mask_path)[, , 1]

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }
    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    structure(list(x = x, y = y), class = "image_with_segmentation_mask")
  },

  .length = function() {
    length(self$img_paths)
  },

  active = list(
    raw_folder = function() {
      file.path(self$root_path, paste0("pascal_voc_", self$year), "raw")
    },
    processed_folder = function() {
      file.path(self$root_path, paste0("pascal_voc_", self$year), "processed")
    }
  )
)
