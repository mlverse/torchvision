#' COCO Detection Dataset
#'
#' Loads the MS COCO dataset for object detection and segmentation.
#'
#' @name coco_detection_dataset
#' @rdname coco_detection_dataset
#' @param root Root directory where the dataset is stored or will be downloaded to.
#' @param train Logical. If TRUE, loads the training split; otherwise, loads the validation split.
#' @param year Character. Dataset version year. One of \code{"2014"}, \code{"2016"}, or \code{"2017"}.
#' @param download Logical. If TRUE, downloads the dataset if it's not already present in the \code{root} directory.
#' @param transforms Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target (labels, boxes, etc.).
#'
#' @return
#' A torch dataset. Each example is a list with two elements:
#'
#' \describe{
#'   \item{x}{A 3D \code{torch_tensor} of shape \code{(C, H, W)} representing the image.}
#'   \item{y}{A list containing:
#'     \describe{
#'       \item{boxes}{A 2D \code{torch_tensor} of shape \code{(N, 4)} containing bounding boxes
#'       in the format c(\eqn{x_{min}}, \eqn{y_{min}}, \eqn{x_{max}}, \eqn{y_{max}}).}
#'       \item{labels}{A 1D \code{torch_tensor} of type integer, representing the class label for each object.}
#'       \item{area}{A 1D \code{torch_tensor} of type float, indicating the area of each object.}
#'       \item{iscrowd}{A 1D \code{torch_tensor} of type boolean, where \code{TRUE} indicates the object is part of a crowd.}
#'       \item{segmentation}{A list of segmentation polygons for each object.}
#'       \item{masks}{A 3D \code{torch_tensor} of shape \code{(N, H, W)} containing binary segmentation masks.}
#'     }
#'   }
#' }
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
#'   root = "~/data",
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
#' @importFrom jsonlite fromJSON
#' @export
coco_detection_dataset <- torch::dataset(
  name = "coco_detection_dataset",

  initialize = function(root, train = TRUE, year = c("2017", "2016", "2014"),
                        download = FALSE, transforms = NULL, target_transform = NULL) {

    year <- match.arg(year)
    split <- if (train) "train" else "val"

    root <- fs::path_expand(root)
    self$root <- root
    self$year <- year
    self$split <- split
    self$transforms <- transforms
    self$target_transform <- target_transform
    self$archive_size <- 1.12

    cli_inform("{.cls {class(self)[[1]]}} Dataset will be downloaded and processed if not already available.")

    self$data_dir <- fs::path(root, glue::glue("coco{year}"))

    image_year <- if (year == "2016") "2014" else year
    self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{image_year}"))
    self$annotation_file <- fs::path(self$data_dir, "annotations",
                                     glue::glue("instances_{split}{year}.json"))

    if (download) {
      self$download()
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
  },

  check_exists = function() {
    fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)
  },

  .getitem = function(index) {
    image_id <- self$image_ids[index]
    image_info <- self$image_metadata[[as.character(image_id)]]

    img_path <- fs::path(self$image_dir, image_info$file_name)

    img_array <- jpeg::readJPEG(img_path)
    if (length(dim(img_array)) == 2) {
      img_array <- array(rep(img_array, 3), dim = c(dim(img_array), 3))
    }
    img_array <- aperm(img_array, c(3, 1, 2))
    img_tensor <- torch::torch_tensor(img_array, dtype = torch::torch_float())

    H <- as.integer(img_tensor$shape[2])
    W <- as.integer(img_tensor$shape[3])

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
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
      area <- torch::torch_empty(0, dtype = torch::torch_float())
      iscrowd <- torch::torch_empty(0, dtype = torch::torch_bool())
      masks_tensor <- torch::torch_zeros(c(0, H, W), dtype = torch::torch_bool())
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

    structure(
      list(
        x = img_tensor,
        y = y
      ),
      class = c("image_with_bounding_box", "image_with_segmentation_mask")
    )
  },

  .length = function() {
    length(self$image_ids)
  },

  download = function() {
    info <- self$get_resource_info()

    ann_zip <- download_and_cache(info$ann_url)
    archive <- download_and_cache(info$img_url)

    if (tools::md5sum(archive) != info$img_md5) {
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }

    utils::unzip(ann_zip, exdir = self$data_dir)
    utils::unzip(archive, exdir = self$data_dir)
  },

  get_resource_info = function() {
    split <- self$split
    list(
      "2017" = list(
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ann_md5 = "f4bbac642086de4f52a3fdda2de5fa2c",
        img_url = glue::glue("http://images.cocodataset.org/zips/{split}2017.zip"),
        img_md5 = if (split == "train") "cced6f7f71b7629ddf16f17bbcfab6b2" else "442b8da7639aecaf257c1dceb8ba8c80"
      ),
      "2016" = list(
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2016.zip",
        ann_md5 = "0572d6c4f6e1b4efb6191f6b3b344b2a",
        img_url = glue::glue("http://images.cocodataset.org/zips/{split}2014.zip"),
        img_md5 = if (split == "train") "0da8cfa0e090c266b78f30e2d2874f1a" else "a3d79f5ed8d289b7a7554ce06a5782b3"
      ),
      "2014" = list(
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        ann_md5 = "d3e24dc9f5bda3e7887a1e6a5ff34c0c",
        img_url = glue::glue("http://images.cocodataset.org/zips/{split}2014.zip"),
        img_md5 = if (split == "train") "0da8cfa0e090c266b78f30e2d2874f1a" else "a3d79f5ed8d289b7a7554ce06a5782b3"
      )
    )[[self$year]]
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
#' @name coco_caption_dataset
#' @rdname coco_caption_dataset
#' @inheritParams coco_detection_dataset
#'
#' @examples
#' \dontrun{
#' ds <- coco_dataset(
#'   root = "~/data",
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
#' @export
coco_caption_dataset <- torch::dataset(
  name = "coco_caption_dataset",
  inherit = coco_detection_dataset,

  initialize = function(root,
                        train = TRUE,
                        year = c("2014"),
                        download = FALSE
  ) {
    year <- match.arg(year)
    split <- if (train) "train" else "val"

    root <- fs::path_expand(root)
    self$root <- root
    self$split <- split
    self$year <- year
    self$data_dir <- fs::path(root, glue::glue("coco{year}"))
    self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{year}"))
    self$annotation_file <- fs::path(self$data_dir, "annotations", glue::glue("captions_{split}{year}.json"))

    if (download)
      self$download()

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
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
    caption <- ann$caption

    prefix <- if (self$split == "train") "COCO_train2014_" else "COCO_val2014_"
    filename <- paste0(prefix, sprintf("%012d", image_id), ".jpg")
    image_path <- fs::path(self$image_dir, filename)

    image_array <- jpeg::readJPEG(image_path)

    list(x = image_array, y = caption)
  }
)
