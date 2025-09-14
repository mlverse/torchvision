#' Pascal VOC Segmentation Dataset
#'
#' The Pascal Visual Object Classes (VOC) dataset is a widely used benchmark for object detection and semantic segmentation tasks in computer vision.
#'
#' This dataset provides RGB images along with per-pixel class segmentation masks for 20 object categories, plus a background class.
#' Each pixel in the mask is labeled with a class index corresponding to one of the predefined semantic categories.
#'
#' The VOC dataset was released in yearly editions (2007 to 2012), with slight variations in data splits and annotation formats.
#' Notably, only the 2007 edition includes a separate `test` split; all other years (2008–2012) provide only the `train`, `val`, and `trainval` splits.
#'
#' The dataset defines 21 semantic classes: `"background"`, `"aeroplane"`, `"bicycle"`, `"bird"`, `"boat"`, `"bottle"`, `"bus"`, `"car"`, `"cat"`, `"chair"`,
#' `"cow"`, `"dining table"`, `"dog"`, `"horse"`, `"motorbike"`, `"person"`, `"potted plant"`, `"sheep"`, `"sofa"`, `"train"`, and `"tv/monitor"`.
#' They are available through the `classes` variable of the dataset object.
#'
#' This dataset is frequently used for training and evaluating semantic segmentation models, and supports tasks requiring dense, per-pixel annotations.
#'
#' @inheritParams oxfordiiitpet_dataset
#' @param root Character. Root directory where the dataset will be stored under `root/pascal_voc_<year>`.
#' @param year Character. VOC dataset version to use. One of `"2007"`, `"2008"`, `"2009"`, `"2010"`, `"2011"`, or `"2012"`. Default is `"2012"`.
#' @param split Character. One of `"train"`, `"val"`, `"trainval"`, or `"test"`. Determines the dataset split. Default is `"train"`.
#'
#' @return A torch dataset of class \code{pascal_segmentation_dataset}.
#'
#' The returned list inherits class \code{image_with_segmentation_mask}, which allows generic visualization
#' utilities to be applied.
#'
#' Each element is a named list with the following structure:
#' - `x`: a H x W x 3 array representing the RGB image.
#' - `y`: A named list containing:
#'     - `masks`: A `torch_tensor` of dtype `bool` and shape `(21, H, W)`, representing a multi-channel segmentation mask.
#'       Each of the 21 channels corresponds to a Pascal VOC classes
#'     - `labels`: An integer vector indicating the indices of the classes present in the mask.
#'
#' @examples
#' \dontrun{
#' # Load Pascal VOC segmentation dataset (2007 train split)
#' pascal_seg <- pascal_segmentation_dataset(
#'  transform = transform_to_tensor,
#'  download = TRUE,
#'  year = "2007"
#' )
#'
#' # Access the first image and its mask
#' first_item <- pascal_seg[1]
#' first_item$x  # Image
#' first_item$y$masks  # Segmentation mask
#' first_item$y$labels  # Unique class labels in the mask
#' pascal_seg$classes[first_item$y$labels]  # Class names
#'
#' # Visualise the first image and its mask
#' masked_img <- draw_segmentation_masks(first_item)
#' tensor_image_browse(masked_img)
#'
#' # Load Pascal VOC detection dataset (2007 train split)
#' pascal_det <- pascal_detection_dataset(
#'  transform = transform_to_tensor,
#'  download = TRUE,
#'  year = "2007"
#' )
#'
#' # Access the first image and its bounding boxes
#' first_item <- pascal_det[1]
#' first_item$x  # Image
#' first_item$y$labels  # Object labels
#' first_item$y$boxes  # Bounding box tensor
#'
#' # Visualise the first image with bounding boxes
#' boxed_img <- draw_bounding_boxes(first_item)
#' tensor_image_browse(boxed_img)
#' }
#'
#' @name pascal_voc_datasets
#' @title Pascal VOC Datasets
#' @rdname pascal_voc_datasets
#' @family segmentation_dataset
#' @export
pascal_segmentation_dataset <- torch::dataset(
  name = "pascal_segmentation_dataset",

  resources = data.frame(
    year = c("2007", "2007", "2008", "2009", "2010", "2011", "2012"),
    type = c("trainval", "test", "trainval", "trainval", "trainval", "trainval", "trainval"),
    url = c("https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_06-Nov-2007.tar",
            "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtest_06-Nov-2007.tar",
            "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_14-Jul-2008.tar",
            "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_11-May-2009.tar",
            "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_03-May-2010.tar",
            "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_25-May-2011.tar",
            "https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/VOCtrainval_11-May-2012.tar"),
    md5 = c("c52e279531787c972589f7e41ab4ae64",
            "b6e924de25625d8de591ea690078ad9f",
            "2629fa636546599198acfcfbfcf1904a",
            "59065e4b188729180974ef6572f6a212",
            "da459979d0c395079b5c75ee67908abb",
            "6c3384ef61512963050cb5d687e5bf1e",
            "6cd6e144f989b92b3379bac3b3de84fd"),
    size = c("440 MB", "440 MB", "550 MB", "890 MB", "1.3 GB", "1.7 GB", "1.9 GB")
  ),
  classes = c(
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train",
    "tv/monitor"
  ),
  voc_colormap = c(
    c(0, 0, 0), c(128, 0, 0), c(0, 128, 0), c(128, 128, 0),
    c(0, 0, 128), c(128, 0, 128), c(0, 128, 128), c(128, 128, 128),
    c(64, 0, 0), c(192, 0, 0), c(64, 128, 0), c(192, 128, 0),
    c(64, 0, 128), c(192, 0, 128), c(64, 128, 128), c(192, 128, 128),
    c(0, 64, 0), c(128, 64, 0), c(0, 192, 0), c(128, 192, 0),
    c(0, 64, 128)
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
    self$year <- match.arg(year, choices = unique(self$resources$year))
    self$split <-  match.arg(split, choices = c("train", "val", "trainval", "test"))
    self$transform <- transform
    self$target_transform <- target_transform
    if (self$split == "test"){
        self$archive_key <- "test"
    } else {
        self$archive_key <- "trainval"
    }
    self$archive_size <- self$resources[self$resources$year == self$year & self$resources$type == self$archive_key,]$size

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

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
  },

  download = function() {

    if (self$check_exists()) {
      return()
    }

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    resource <- self$resources[self$resources$year == self$year & self$resources$type == self$archive_key,]
    archive <- download_and_cache(resource$url, prefix = "pascal_dataset")
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
    mask_data <- png::readPNG(mask_path) * 255

    flat_mask <- matrix(
      c(as.vector(mask_data[, , 1]),
        as.vector(mask_data[, , 2]),
        as.vector(mask_data[, , 3])),
      ncol = 3
    )
    colormap_mat <- matrix(self$voc_colormap, ncol = 3, byrow = TRUE)
    rgb_to_int <- function(mat) {
      as.integer(mat[, 1]) * 256^2 + as.integer(mat[, 2]) * 256 + as.integer(mat[, 3])
    }
    match_indices <- match(rgb_to_int(flat_mask), rgb_to_int(colormap_mat)) - 1
    class_idx <- matrix(match_indices, nrow = dim(mask_data)[1], ncol = dim(mask_data)[2])
    class_idx_tensor <- torch_tensor(class_idx, dtype = torch_long())
    class_ids <- torch_arange(0, 20, dtype = torch_long())$view(c(21, 1, 1))
    masks <- (class_ids == class_idx_tensor$unsqueeze(1))$to(dtype = torch_bool())
    labels <- which(as_array(masks$any(dim = c(2, 3))))

    y <- list(labels = labels, masks = masks)

    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }
    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }

    result <- list(x = x, y = y)
    class(result) <- c("image_with_segmentation_mask", class(result))
    result
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

#' Pascal VOC Detection Dataset
#'
#' @inheritParams pascal_segmentation_dataset
#'
#' @return A torch dataset of class \code{pascal_detection_dataset}.
#'
#' The returned list inherits class \code{image_with_bounding_box}, which allows generic visualization
#' utilities to be applied.
#'
#' Each element is a named list:
#' - `x`: a H x W x 3 array representing the RGB image.
#' - `y`: a list with:
#'     - `labels`: a character vector with object class names.
#'     - `boxes`: a tensor of shape (N, 4) with bounding box coordinates in `(xmin, ymin, xmax, ymax)` format.
#'
#' @rdname pascal_voc_datasets
#' @family detection_dataset
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
    self$year <- match.arg(year, choices = unique(self$resources$year))
    self$split <- match.arg(split, choices = c("train", "val", "trainval", "test"))
    self$transform <- transform
    self$target_transform <- target_transform
    if (self$split == "test") {
      self$archive_key <- "test"
    } else {
      self$archive_key <- "trainval"
    }
    self$archive_size <- self$resources[self$resources$year == self$year & self$resources$type == self$archive_key,]$size

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

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {self$.length()} images across {length(self$classes)} classes.")
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

    result <- list(x = x, y = y)
    class(result) <- c("image_with_bounding_box", class(result))
    result
  },

  parse_voc_xml = function(xml) {
    objects <- xml2::xml_find_all(xml, ".//object")

    labels <- xml2::xml_text(xml2::xml_find_all(objects, "name"))

    bboxes <- xml2::xml_find_all(objects, "bndbox")

    xmin <- xml2::xml_integer(xml2::xml_find_all(bboxes, "xmin"))
    ymin <- xml2::xml_integer(xml2::xml_find_all(bboxes, "ymin"))
    xmax <- xml2::xml_integer(xml2::xml_find_all(bboxes, "xmax"))
    ymax <- xml2::xml_integer(xml2::xml_find_all(bboxes, "ymax"))

    boxes <- torch_tensor(data.frame(xmin, ymin, xmax, ymax) %>% as.matrix(), dtype = torch_long())

    list(
      labels = labels,
      boxes = boxes
    )
  }
)
