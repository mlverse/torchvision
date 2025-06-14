# Fixing the COCO detection dataset based on mentor comments

#' COCO Detection Dataset
#'
#' Loads MS COCO Dataset for object detection and segmentation.
#'
#' @param root Root directory for data.
#' @param train Whether to load training split (TRUE) or validation (FALSE).
#' @param year Dataset year: "2014", "2016", or "2017".
#' @param download If TRUE, downloads the dataset if needed.
#' @param transforms Transform applied to image.
#' @param target_transform Transform applied to target.
#'
#' The returned image is in CHW format (channels, height, width), matching torch convention.
#'
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

    if (download)
      self$download()

    if (!self$check_files())
      rlang::abort("Dataset files not found. Use download = TRUE to fetch them.")

    self$load_annotations()
  },

  check_files = function() {
    fs::file_exists(self$ann_file) && fs::dir_exists(self$image_dir)
  },

  .getitem = function(index) {
    image_id <- self$image_ids[index]
    info <- self$images[[as.character(image_id)]]

    img_path <- fs::path(self$image_dir, info$file_name)
    img_arr <- jpeg::readJPEG(img_path)
    img_arr <- aperm(img_arr, c(3, 2, 1))  # CHW format

    anns <- self$annotations[self$annotations$image_id == image_id, ]

    if (nrow(anns) > 0) {
      boxes <- do.call(rbind, lapply(anns$bbox, function(b) c(b[1], b[2], b[1] + b[3], b[2] + b[4])))
      colnames(boxes) <- c("x1", "y1", "x2", "y2")
      labels <- anns$category_id
      area <- sapply(anns$area, identity)
      iscrowd <- sapply(anns$iscrowd, identity)
      segmentation <- anns$segmentation
    } else {
      boxes <- matrix(nrow = 0, ncol = 4, dimnames = list(NULL, c("x1", "y1", "x2", "y2")))
      labels <- area <- iscrowd <- integer(0)
      segmentation <- list()
    }

    target <- list(
      image_id = image_id,
      boxes = boxes,
      labels = labels,
      area = area,
      iscrowd = iscrowd,
      segmentation = segmentation,
      height = info$height,
      width = info$width
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
    if (tools::md5sum(ann_zip) != info$ann_md5) {
      stop("MD5 mismatch for annotations zip: corrupted download?")
    }

    img_zip <- download_and_cache(info$img_url)
    if (tools::md5sum(img_zip) != info$img_md5) {
      stop("MD5 mismatch for image zip: corrupted download?")
    }

    utils::unzip(ann_zip, exdir = self$data_dir)
    utils::unzip(img_zip, exdir = self$data_dir)
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

    self$images <- setNames(split(json_data$images, seq_len(nrow(json_data$images))),
                            vapply(json_data$images, function(x) as.character(x$id), character(1)))

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
