#' @export
pascal_segmentation_dataset <- torch::dataset(
  name = "pascal_segmentation_dataset",

  resources = list(
    `2007` = list(
      trainval = list(url = "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_06-Nov-2007.tar",md5 = "c52e279531787c972589f7e41ab4ae64"),
      test = list(url = "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtest_06-Nov-2007.tar",md5 = "b6e924de25625d8de591ea690078ad9f")
    ),
    `2008` = list(
      trainval = list(url = "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_14-Jul-2008.tar",md5 = "2629fa636546599198acfcfbfcf1904a")
    ),
    `2009` = list(
      trainval = list(url = "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_11-May-2009.tar",md5 = "59065e4b188729180974ef6572f6a212")
    ),
    `2010` = list(
      trainval = list(url = "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_03-May-2010.tar",md5 = "da459979d0c395079b5c75ee67908abb")
    ),
    `2011` = list(
      trainval = list(url = "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_25-May-2011.tar",md5 = "6c3384ef61512963050cb5d687e5bf1e")
    ),
    `2012` = list(
      trainval = list(url = "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_11-May-2012.tar",md5 = "6cd6e144f989b92b3379bac3b3de84fd")
    )
  ),

  archive_size_table = list(
    "2007" = list(trainval = "440 MB", test = "440 MB"),
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
    if (self$split == "test"){
        self$archive_key <- "test"
    } else {
        self$archive_key <- "trainval"
    }
    self$archive_size <- self$archive_size_table[[self$year]][[self$archive_key]]

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    data_file <- file.path(self$processed_folder, paste0(self$split, ".rds"))
    data <- readRDS(data_file)
    self$img_path <- data$img_path
    self$mask_paths <- data$mask_paths

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$img_path)} images.")
  },

  download = function() {

    if (self$check_exists()) {
      return()
    }

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    resource <- self$resources[[self$year]][[self$archive_key]]
    archive <- download_and_cache(resource$url, prefix = class(self)[1])
    actual_md5 <- tools::md5sum(archive)

    if (actual_md5 != resource$md5) {
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }

    utils::untar(archive, exdir = self$raw_folder)

    voc_dir <- file.path(self$raw_folder, "VOCdevkit", paste0("VOC", self$year))
    voc_root <- self$raw_folder
    if (self$year == "2011") {
    voc_root <- file.path(voc_root, "TrainVal")
    }
    voc_dir <- file.path(voc_root, "VOCdevkit", paste0("VOC", self$year))

    split_file <- file.path(voc_dir, "ImageSets", "Segmentation", paste0(self$split, ".txt"))

    ids <- readLines(split_file)
    img_path <- file.path(voc_dir, "JPEGImages", paste0(ids, ".jpg"))
    mask_paths <- file.path(voc_dir, "SegmentationClass", paste0(ids, ".png"))

    saveRDS(list(
      img_path = img_path,
      mask_paths = mask_paths
    ), file.path(self$processed_folder, paste0(self$split, ".rds")))

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  check_exists = function() {
    fs::file_exists(file.path(self$processed_folder, paste0(self$split, ".rds")))
  },

  .getitem = function(index) {
    img_path <- self$img_path[index]
    mask_path <- self$mask_paths[index]

    x <- jpeg::readJPEG(img_path)
    masks <- png::readPNG(mask_path)*255
    masks <- torch_tensor(masks, dtype = torch_bool())$permute(c(3, 1, 2))

    y <- list(
      masks = masks
    )

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }
    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    structure(list(x = x, y = y), class = "image_with_segmentation_mask")
  },

  .length = function() {
    length(self$img_path)
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

#' @export
pascal_detection_dataset <- torch::dataset(
  name = "pascal_detection_dataset",

  inherit = pascal_segmentation_dataset,

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
    if (self$split == "test") {
      self$archive_key <- "test"
    } else {
      self$archive_key <- "trainval"
    }
    self$archive_size <- self$archive_size_table[[self$year]][[self$archive_key]]

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")
    }

    voc_dir <- file.path(self$raw_folder, "VOCdevkit", paste0("VOC", self$year))
    if (self$year == "2011") {
      voc_dir <- file.path(self$raw_folder, "TrainVal", "VOCdevkit", paste0("VOC", self$year))
    }

    ids_file <- file.path(voc_dir, "ImageSets", "Main", paste0(self$split, ".txt"))
    ids <- readLines(ids_file)

    self$img_path <- file.path(voc_dir, "JPEGImages", paste0(ids, ".jpg"))
    self$annotation_paths <- file.path(voc_dir, "Annotations", paste0(ids, ".xml"))

    if (!requireNamespace("xml2", quietly = TRUE)) {
      install.packages("xml2")
    }

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$img_path)} images.")
  },

  .getitem = function(index) {

    x <- jpeg::readJPEG(self$img_path[index])
    ann_path <- self$annotation_paths[index]
    y <- self$parse_voc_xml(xml2::read_xml(ann_path))

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }
    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    structure(list(x = x, y = y), class = "image_with_bounding_box")
  },

  parse_voc_xml = function(xml) {
    objects <- xml2::xml_find_all(xml, ".//object")

    labels <- character(length(objects))
    boxes <- vector("list", length(objects))

    for (i in seq_along(objects)) {
      obj <- objects[[i]]

      labels[i] <- xml2::xml_text(xml2::xml_find_first(obj, "name"))

      bbox <- xml2::xml_find_first(obj, "bndbox")
      xmin <- as.integer(xml2::xml_text(xml2::xml_find_first(bbox, "xmin")))
      ymin <- as.integer(xml2::xml_text(xml2::xml_find_first(bbox, "ymin")))
      xmax <- as.integer(xml2::xml_text(xml2::xml_find_first(bbox, "xmax")))
      ymax <- as.integer(xml2::xml_text(xml2::xml_find_first(bbox, "ymax")))

      boxes[[i]] <- c(xmin, ymin, xmax, ymax)
    }

    boxes <- torch_tensor(do.call(rbind, boxes), dtype = torch_int64())

    list(
      labels = labels,
      boxes = boxes
    )
  }
)