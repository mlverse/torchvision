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
#' An R6 dataset object. Each item is a list with two elements:
#' \describe{
#'   \item{image}{A 3D \code{torch_tensor} of shape \code{[C, H, W]} (channel-first).}
#'   \item{target}{A list with:
#'     \describe{
#'       \item{boxes}{A matrix of bounding boxes in \code{[x1, y1, x2, y2]} format.}
#'       \item{labels}{An integer vector of class labels.}
#'       \item{area}{A numeric vector indicating the area of each object.}
#'       \item{iscrowd}{An integer vector (0 or 1) indicating whether objects are crowds.}
#'       \item{segmentation}{A list of segmentation polygons per object.}
#'     }
#'   }
#' }
#'
#' @details
#' The returned image is in CHW format (channels, height, width), matching the torch convention.
#' The dataset supports loading object detection annotations such as bounding boxes, labels,
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
#' sample <- ds[1]
#' image <- sample$image
#' target <- sample$target
#'
#' # Convert image to uint8
#' image_uint8 <- image$mul(255)$clamp(0, 255)$to(dtype = torch::torch_uint8())
#'
#' # Ensure bounding boxes are torch tensors
#' boxes <- if (!inherits(target$boxes, "torch_tensor")) {
#'   torch::torch_tensor(target$boxes, dtype = torch::torch_float())
#' } else {
#'   target$boxes
#' }
#'
#' # Convert labels to character
#' labels <- as.character(torch::as_array(target$labels))
#'
#' # Draw bounding boxes
#' output <- torchvision::draw_bounding_boxes(
#'   image = image_uint8,
#'   boxes = boxes,
#'   labels = labels
#' )
#'
#' # Convert to array and plot
#' output_array <- as.array(output$permute(c(2, 3, 1)))  # CHW to HWC
#' output_array <- as.numeric(output_array) / 255
#' dim(output_array) <- dim(as.array(output$permute(c(2, 3, 1))))
#' plot(as.raster(output_array))
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

      labels <- torch::torch_tensor(anns$category_id, dtype = torch::torch_int())   # ✔️ numeric integer
      area <- torch::torch_tensor(anns$area, dtype = torch::torch_float())          # ✔️ numeric float
      iscrowd <- torch::torch_tensor(anns$iscrowd, dtype = torch::torch_int())      # ✔️ integer (0 or 1)
      segmentation <- anns$segmentation                                             # ✔️ leave as list
    } else {
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())         # ✔️ zero-size tensor
      labels <- torch::torch_empty(0, dtype = torch::torch_int())                # ✔️ empty integer tensor
      area <- torch::torch_empty(0, dtype = torch::torch_float())                # ✔️ empty float tensor
      iscrowd <- torch::torch_empty(0, dtype = torch::torch_int())               # ✔️ empty integer tensor
      segmentation <- list()                                                    # ✔️ remains list
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
