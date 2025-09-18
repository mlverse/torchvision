#' COCO Detection Dataset
#'
#' Loads the MS COCO dataset for object detection and segmentation.
#'
#' @rdname coco_detection_dataset
#' @param root Root directory where the dataset is stored or will be downloaded to.
#' @param train Logical. If TRUE, loads the training split; otherwise, loads the validation split.
#' @param year Character. Dataset version year. One of \code{"2014"} or \code{"2017"}.
#' @param download Logical. If TRUE, downloads the dataset if it's not already present in the \code{root} directory.
#' @param transform Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target (labels, boxes, etc.).
#'
#' @return An object of class `coco_detection_dataset`. Each item is a list:
#' - `x`: a `(C, H, W)` array representing the image.
#' - `y$boxes`: a `(N, 4)` `torch_tensor` of bounding boxes in the format  \eqn{(x_{min}, y_{min}, x_{max}, y_{max})}.
#' - `y$labels`: an integer `torch_tensor` with the class label for each object.
#' - `y$area`: a float `torch_tensor` indicating the area of each object.
#' - `y$iscrowd`: a boolean `torch_tensor`, where `TRUE` marks the object as part of a crowd.
#' - `y$segmentation`: a list of segmentation polygons for each object.
#' - `y$masks`: a `(N, H, W)` boolean `torch_tensor` containing binary segmentation masks.
#'
#' The returned object has S3 classes \code{"image_with_bounding_box"} and \code{"image_with_segmentation_mask"}
#' to enable automatic dispatch by visualization functions such as \code{draw_bounding_boxes()} and \code{draw_segmentation_masks()}.
#'
#' @details
#' The returned image is in CHW format (channels, height, width), matching the torch convention.
#' The dataset `y` offers object detection annotations such as bounding boxes, labels,
#' areas, crowd indicators, and segmentation masks from the official COCO annotations.
#'
#' @examples
#' \dontrun{
#' ds <- coco_detection_dataset(
#'   train = FALSE,
#'   year = "2017",
#'   download = TRUE
#' )
#'
#' item <- ds[1]
#'
#' # Visualize bounding boxes
#' boxed <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed)
#'
#' # Visualize segmentation masks (if present)
#' masked <- draw_segmentation_masks(item)
#' tensor_image_browse(masked)
#' }
#' @family detection_dataset
#' @importFrom jsonlite fromJSON
#' @export
coco_detection_dataset <- torch::dataset(
  name = "coco_detection_dataset",
  resources = data.frame(
    year = rep(c(2017, 2014), each = 4 ),
    content = rep(c("image", "annotation"), time = 2, each = 2),
    split = rep(c("train", "val"), time = 4),
    url = c("http://images.cocodataset.org/zips/train2017.zip", "http://images.cocodataset.org/zips/val2017.zip",
            rep("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", time = 2),
            "http://images.cocodataset.org/zips/train2014.zip", "http://images.cocodataset.org/zips/val2014.zip",
            rep("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", time = 2)),
    size = c("800 MB", "800 MB", rep("770 MB", time = 2), "6.33 GB", "6.33 GB", rep("242 MB", time = 2)),
    md5 = c(c("cced6f7f71b7629ddf16f17bbcfab6b2", "442b8da7639aecaf257c1dceb8ba8c80"),
            rep("f4bbac642086de4f52a3fdda2de5fa2c", time = 2),
            c("0da8cfa0e090c266b78f30e2d2874f1a", "a3d79f5ed8d289b7a7554ce06a5782b3"),
            rep("0a379cfc70b0e71301e0f377548639bd", time = 2)),
    stringsAsFactors = FALSE
  ),

  initialize = function(
    root = tempdir(),
    train = TRUE,
    year = c("2017", "2014"),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {

    year <- match.arg(year)
    split <- ifelse(train, "train", "val")

    root <- fs::path_expand(root)
    self$root <- root
    self$year <- year
    self$split <- split
    self$transform <- transform
    self$target_transform <- target_transform
    self$archive_size <- self$resources[self$resources$year == year & self$resources$split == split & self$resources$content == "image", ]$size

    self$data_dir <- fs::path(root, glue::glue("coco{year}"))

    image_year <- ifelse(year == "2016", "2014", year)
    self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{image_year}"))
    self$annotation_file <- fs::path(self$data_dir, "annotations",
                                     glue::glue("instances_{split}{year}.json"))

    if (download) {
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$image_ids)} images.")
  },

  check_exists = function() {
    fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)
  },

  .getitem = function(index) {
    image_id <- self$image_ids[index]
    image_info <- self$image_metadata[[as.character(image_id)]]

    img_path <- fs::path(self$image_dir, image_info$file_name)

    x <- base_loader(img_path)

    H <- dim(x)[1]
    W <- dim(x)[2]

    anns <- self$annotations[self$annotations$image_id == image_id, ]

    if (nrow(anns) > 0) {
      boxes_wh <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float())
      boxes <- box_xywh_to_xyxy(boxes_wh)

      label_ids <- anns$category_id
      labels <- as.character(self$categories$name[match(label_ids, self$categories$id)])

      area <- torch::torch_tensor(anns$area, dtype = torch::torch_float())
      iscrowd <- torch::torch_tensor(as.logical(anns$iscrowd), dtype = torch::torch_bool())

      masks <- lapply(seq_len(nrow(anns)), function(i) {
        seg <- anns$segmentation[[i]]
        if (is.list(seg) && length(seg) > 0) {
          mask <- coco_polygon_to_mask(seg, height = H, width = W)
          if (inherits(mask, "torch_tensor") && mask$ndim == 2) return(mask)
        }
        NULL
      })

      masks <- Filter(function(m) inherits(m, "torch_tensor") && m$ndim == 2, masks)

      if (length(masks) > 0) {
        masks_tensor <- torch::torch_stack(masks)
      } else {
        masks_tensor <- torch::torch_zeros(c(0, H, W), dtype = torch::torch_bool())
      }

    } else {
      # empty annotation
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
      area <- torch::torch_empty(0, dtype = torch::torch_float())
      iscrowd <- torch::torch_empty(0, dtype = torch::torch_bool())
      masks_tensor <- torch::torch_empty(c(0, H, W), dtype = torch::torch_bool())
      anns$segmentation <- list()
    }

    y <- list(
      boxes = boxes,
      labels = labels,
      area = area,
      iscrowd = iscrowd,
      segmentation = anns$segmentation,
      masks = masks_tensor
    )

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    result <- list(x = x, y = y)
    class(result) <- c("image_with_bounding_box", "image_with_segmentation_mask", class(result))
    result
  },

  .length = function() {
    length(self$image_ids)
  },

  download = function() {
    annotation_filter <- self$resources$year == self$year & self$resources$split == self$split & self$resources$content == "annotation"
    image_filter <- self$resources$year == self$year & self$resources$split == self$split & self$resources$content == "image"

    cli_inform("Downloading {.cls {class(self)[[1]]}}...")

    ann_zip <- download_and_cache(self$resources[annotation_filter, ]$url, prefix = "coco_dataset")
    archive <- download_and_cache(self$resources[image_filter, ]$url, prefix = "coco_dataset")

    if (tools::md5sum(archive) != self$resources[image_filter, ]$md5) {
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }

    utils::unzip(ann_zip, exdir = self$data_dir)
    utils::unzip(archive, exdir = self$data_dir)

    cli_inform("Dataset {.cls {class(self)[[1]]}} downloaded and extracted successfully.")
  },

  load_annotations = function() {
    data <- jsonlite::fromJSON(self$annotation_file)

    self$image_metadata <- setNames(
      split(data$images, seq_len(nrow(data$images))),
      as.character(data$images$id)
    )

    self$annotations <- data$annotations
    self$categories <- data$categories
    self$category_names <- setNames(self$categories$name, self$categories$id)

    ids <- as.numeric(names(self$image_metadata))
    image_paths <- fs::path(self$image_dir,
                            sapply(ids, function(id) self$image_metadata[[as.character(id)]]$file_name))
    exist <- fs::file_exists(image_paths)
    self$image_ids <- ids[exist]
  }
)


#' COCO Caption Dataset
#'
#' Loads the MS COCO dataset for image captioning.
#'
#' @rdname coco_caption_dataset
#' @inheritParams coco_detection_dataset
#' @param year Character. Dataset version year. One of \code{"2014"}.
#'
#' @return An object of class `coco_caption_dataset`. Each item is a list:
#' - `x`: an `(H, W, C)` numeric array containing the RGB image.
#' - `y`: a character string with the image caption.
#'
#' @examples
#' \dontrun{
#' ds <- coco_caption_dataset(
#'   train = FALSE,
#'   download = TRUE
#' )
#' example <- ds[1]
#'
#' # Access image and caption
#' x <- example$x
#' y <- example$y
#'
#' # Prepare image for plotting
#' image_array <- as.numeric(x)
#' dim(image_array) <- dim(x)
#'
#' plot(as.raster(image_array))
#' title(main = y, col.main = "black")
#' }
#' @family caption_dataset
#' @export
coco_caption_dataset <- torch::dataset(
  name = "coco_caption_dataset",
  inherit = coco_detection_dataset,

  initialize = function(
    root = tempdir(),
    train = TRUE,
    year = c("2014"),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {

    year <- match.arg(year)
    split <- ifelse(train, "train", "val")

    root <- fs::path_expand(root)
    self$root <- root
    self$split <- split
    self$year <- year
    self$transform <- transform
    self$target_transform <- target_transform
    self$data_dir <- fs::path(root, glue::glue("coco{year}"))
    self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{year}"))
    self$annotation_file <- fs::path(self$data_dir, "annotations", glue::glue("captions_{split}{year}.json"))
    self$archive_size <- self$resources[self$resources$year == year & self$resources$split == split & self$resources$content == "image", ]$size

    if (download){
      cli_inform("Dataset {.cls {class(self)[[1]]}} (~{.emph {self$archive_size}}) will be downloaded and processed if not already available.")
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()

    cli_inform("{.cls {class(self)[[1]]}} dataset loaded with {length(self$image_ids)} images.")
  },

  load_annotations = function() {
    data <- jsonlite::fromJSON(self$annotation_file)
    self$annotations <- data$annotations
    self$image_ids <- unique(self$annotations$image_id)
  },

  .getitem = function(index) {
    if (index < 1 || index > length(self))
      rlang::abort("Index out of bounds")

    ann <- self$annotations[index, ]
    image_id <- ann$image_id
    y <- ann$caption

    prefix <- ifelse(self$split == "train", "COCO_train2014_", "COCO_val2014_")
    filename <- paste0(prefix, sprintf("%012d", image_id), ".jpg")
    image_path <- fs::path(self$image_dir, filename)

    x <- jpeg::readJPEG(image_path)

    if (!is.null(self$transform))
      x <- self$transform(x)

    if (!is.null(self$target_transform))
      y <- self$target_transform(y)

    list(x = x, y = y)
  }
)
