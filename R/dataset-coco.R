#' COCO Detection Dataset
#'
#' Loads the MS COCO dataset for object detection and segmentation.
#'
#' @param root Root directory where the dataset is stored or will be downloaded to.
#' @param train Logical. If TRUE, loads the training split; otherwise, loads the validation split.
#' @param year Character. Dataset version year. One of \code{"2014"}, \code{"2016"}, or \code{"2017"}.
#' @param download Logical. If TRUE, downloads the dataset if it's not already present in the \code{root} directory.
#' @param transforms Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target (labels, boxes, etc.).
#'
#' @return
#' A torch dataset. Each item is a list with two elements:
#'
#' \describe{
#'   \item{image}{A 3D \code{torch_tensor} of shape \code{(C, H, W)} representing the image.}
#'   \item{target}{A list containing:
#'     \describe{
#'       \item{boxes}{A 2D \code{torch_tensor} of shape \code{(N, 4)} containing bounding boxes
#'       in the format c(\eqn{x_{min}}, \eqn{y_{min}}, \eqn{x_{max}}, \eqn{y_{max}}).}
#'       \item{labels}{A 1D \code{torch_tensor} of type integer, representing the class label for each object.}
#'       \item{area}{A 1D \code{torch_tensor} of type float, indicating the area of each object.}
#'       \item{iscrowd}{A 1D \code{torch_tensor} of type boolean, where \code{TRUE} indicates the object is part of a crowd.}
#'       \item{segmentation}{A list of segmentation polygons for each object.}
#'     }
#'   }
#' }
#' @details
#' The returned image is in CHW format (channels, height, width), matching the torch convention.
#' The dataset `target` offers object detection annotations such as bounding boxes, labels,
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
#' example <- ds[1]
#' image <- example$image
#' target <- example$target
#'
#' # Convert image to uint8
#' image_uint8 <- image$mul(255)$clamp(0, 255)$to(dtype = torch::torch_uint8())
#'
#' # Access the bounding boxes tensor from the target
#' boxes <- target$boxes
#'
#' # Map label IDs to category names
#' label_ids <- as.integer(torch::as_array(target$labels))
#' label_names <- ds$category_names[as.character(label_ids)]
#'
#' # Draw bounding boxes with label names
#' output <- draw_bounding_boxes(
#'   image = image_uint8,
#'   boxes = boxes,
#'   labels = label_names
#' )
#'
#' # Display the result
#' tensor_image_browse(output)
#' }
#'
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

    self$data_dir <- fs::path(root, glue::glue("coco{year}"))

    image_year <- if (year == "2016") "2014" else year
    self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{image_year}"))
    self$ann_file <- fs::path(self$data_dir, "annotations",
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
    fs::file_exists(self$ann_file) && fs::dir_exists(self$image_dir)
  },

  .getitem = function(index) {
    image_id <- self$image_ids[index]
    info <- self$images[[as.character(image_id)]]

    img_path <- fs::path(self$image_dir, info$file_name)

    img_arr <- jpeg::readJPEG(img_path)
    if (length(dim(img_arr)) == 2) {
      img_arr <- array(rep(img_arr, 3), dim = c(dim(img_arr), 3)) # Convert grayscale to RGB
    }
    img_arr <- aperm(img_arr, c(3, 1, 2)) # CHW format
    img_arr <- torch::torch_tensor(img_arr, dtype = torch::torch_float())

    anns <- self$annotations[self$annotations$image_id == image_id, ]

    if (nrow(anns) > 0) {
      boxes <- do.call(rbind, lapply(anns$bbox, function(b) c(b[1], b[2], b[1] + b[3], b[2] + b[4])))
      boxes <- torch::torch_tensor(boxes, dtype = torch::torch_float())

      labels <- torch::torch_tensor(anns$category_id, dtype = torch::torch_int())
      area <- torch::torch_tensor(anns$area, dtype = torch::torch_float())
      iscrowd <- torch::torch_tensor(as.logical(anns$iscrowd), dtype = torch::torch_bool())
      segmentation <- anns$segmentation
    } else {
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- torch::torch_empty(0, dtype = torch::torch_int())
      area <- torch::torch_empty(0, dtype = torch::torch_float())
      iscrowd <- torch::torch_empty(0, dtype = torch::torch_bool())
      segmentation <- list()
    }


    target <- list(
      boxes = boxes,
      labels = labels,
      area = area,
      iscrowd = iscrowd,
      segmentation = segmentation
    )

    if (!is.null(self$transforms))
      img_arr <- self$transforms(img_arr)

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(image = img_arr, target = target)
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
    json_data <- jsonlite::fromJSON(self$ann_file)

    self$images <- setNames(
      split(json_data$images, seq_len(nrow(json_data$images))),
      as.character(json_data$images$id)
    )

    self$annotations <- json_data$annotations
    self$categories <- json_data$categories
    self$category_names <- setNames(self$categories$name, self$categories$id)

    ids <- as.numeric(names(self$images))
    image_files <- fs::path(self$image_dir,
                            sapply(ids, function(id) self$images[[as.character(id)]]$file_name))
    exist <- fs::file_exists(image_files)
    self$image_ids <- ids[exist]
  }
)

#' COCO Caption Dataset
#'
#' Loads the MS COCO dataset for image captioning.
#'
#' @rdname coco_caption_dataset
#' @inheritParams coco_detection_dataset
#'
#' @examples
#' \dontrun{
#' ds <- coco_caption_dataset(
#'   root = "~/data",
#'   train = FALSE,
#'   download = TRUE
#' )
#' example <- ds[1]
#'
#' # Access image and caption
#' image <- example$x
#' caption <- example$y
#'
#' # Prepare image for plotting
#' image_array <- as.numeric(image)
#' dim(image_array) <- dim(image)
#'
#' plot(as.raster(image_array))
#' title(main = caption, col.main = "black")
#' }
#' @export

coco_caption_dataset  <- torch::dataset(
  name = "coco_caption_dataset",
  inherit = coco_detection_dataset,

  initialize = function(root, train = TRUE, year = c("2014"), download = FALSE) {
    year <- match.arg(year)
    split <- if (train) "train" else "val"

    root <- fs::path_expand(root)
    self$root <- root
    self$split <- split
    self$year <- year
    self$data_dir <- fs::path(root, glue::glue("coco{year}"))
    self$image_dir <- fs::path(self$data_dir, glue::glue("{split}{year}"))
    self$ann_file <- fs::path(self$data_dir, "annotations", glue::glue("captions_{split}{year}.json"))

    if (download)
      self$download()

    if (!self$check_files())
      rlang::abort("Dataset files not found. Use download = TRUE to fetch them.")

    self$load_annotations()
  },

  check_files = function() {
    fs::file_exists(self$ann_file) && fs::dir_exists(self$image_dir)
  },

  load_annotations = function() {
    annotations <- jsonlite::fromJSON(self$ann_file)
    self$samples <- annotations$annotations
  },

  .getitem = function(index) {
    if (index < 1 || index > length(self))
      rlang::abort("Index out of bounds")

    ann <- self$samples[index, ]
    image_id <- ann$image_id
    caption <- ann$caption

    prefix <- if (self$split == "train") "COCO_train2014_" else "COCO_val2014_"
    filename <- paste0(prefix, sprintf("%012d", image_id), ".jpg")
    image_path <- fs::path(self$image_dir, filename)

    image <- jpeg::readJPEG(image_path)

    list(x = image, y = caption)
  },

  .length = function() {
    nrow(self$samples)
  }
)

