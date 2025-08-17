#' RF100 Document Dataset Collection
#'
#' Loads one of the RF100 document object detection datasets: "tweeter_post",
#' "tweeter_profile", "document_part", "activity_diagram", "signature",
#' "paper_part", "tabular_data", or "paragraph". Images are provided with
#' COCO-style bounding box annotations for object detection tasks.
#'
#' @param dataset Character. One of "tweeter_post", "tweeter_profile",
#'   "document_part", "activity_diagram", "signature", "paper_part",
#'   "tabular_data", or "paragraph".
#' @param split Character. One of "train", "test", or "valid".
#' @param root Character. Root directory where the dataset will be stored.
#' @param download Logical. If TRUE, downloads the dataset if not present at
#'   `root`.
#' @param transform Optional transform function applied to the image.
#' @param target_transform Optional transform function applied to the target
#'   (labels and boxes).
#'
#' @return A torch dataset. Each element is a named list with:
#' - `x`: H x W x 3 array representing the image.
#' - `y`: a list containing:
#'     - `labels`: character vector with object class names.
#'     - `boxes`: a tensor of shape (N, 4) with bounding boxes in
#'       `(xmin, ymin, xmax, ymax)` format.
#'
#' The returned item inherits the class `image_with_bounding_box` so it can be
#' visualised with helper functions such as [draw_bounding_boxes()].
#'
#' @examples
#' \dontrun{
#' # Load dataset and convert images to tensors
#' devtools::load_all()
#' ds <- rf100_document_collection(
#'   dataset = "tweeter_post",
#'   split = "train",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#'
#' # Retrieve a sample and inspect annotations
#' item <- ds[1]
#' item$y$labels
#' item$y$boxes
#'
#' # Draw bounding boxes and display the image
#' boxed_img <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed_img)
#' }
#'
#' @family detection_dataset
#' @export
rf100_document_collection <- torch::dataset(
  name = "rf100_document_collection",
  resources = data.frame(
    dataset = c(
      "tweeter_post", "tweeter_profile", "document_part",
      "activity_diagram", "signature", "paper_part",
      "tabular_data", "paragraph"
    ),
    url = paste0(
      "https://huggingface.co/datasets/akankshakoshti/rf100-doc/resolve/main/",
      c(
        "tweeter_post.zip",
        "tweeter_profile.zip",
        "document_part.zip",
        "activity_diagram.zip",
        "signature.zip",
        "paper_part.zip",
        "tabular_data.zip",
        "paragraph.zip"
      ),
      "?download=1"
    ),
    md5 = NA_character_
  ),

  initialize = function(
    dataset = c(
      "tweeter_post", "tweeter_profile", "document_part",
      "activity_diagram", "signature", "paper_part",
      "tabular_data", "paragraph"
    ),
    split = c("train", "valid", "test"),
    root = tempdir(),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- match.arg(dataset)
    self$split <- match.arg(split)
    self$root <- fs::path_expand(root)
    self$base_dir <- fs::path(self$root, "rf100-doc")
    self$dataset_dir <- fs::path(self$base_dir, self$dataset)
    self$transform <- transform
    self$target_transform <- target_transform

    if (download) self$download()

    # Try dataset-scoped COCO first
    resolved <- private$resolve_coco_dirs(self$dataset_dir, self$split)
    if (!identical(resolved, FALSE)) {
      self$mode <- "coco"
      self$image_dir <- resolved$image_dir
      self$annotation_file <- resolved$annotation_file
    } else {
      # YOLO dataset layout
      self$mode <- "yolo"
      split_candidates <- if (self$split == "valid") c("valid", "val")
      else if (self$split == "val") c("val", "valid")
      else self$split
      self$image_dir <- fs::path(self$dataset_dir, self$split, "images")
      self$label_dir <- fs::path(self$dataset_dir, self$split, "labels")
      self$yaml_path <- fs::path(self$dataset_dir, "data.yaml")

      if (!fs::dir_exists(self$image_dir) || !fs::dir_exists(self$label_dir)) {
        yolo_res <- private$resolve_yolo_dirs(self$dataset_dir, split_candidates)
        if (!identical(yolo_res, FALSE)) {
          self$image_dir <- yolo_res$image_dir
          self$label_dir <- yolo_res$label_dir
          self$yaml_path <- yolo_res$yaml_path
        }
      }
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. Use `download = TRUE` or check your paths.")
    }

    self$load_annotations()
  },

  download = function() {
    fs::dir_create(self$dataset_dir, recurse = TRUE)
    # Always unzip the selected dataset into <base>/<dataset>
    resource <- self$resources[self$resources$dataset == self$dataset, ]
    archive <- download_and_cache(resource$url, prefix = class(self)[1])
    utils::unzip(archive, exdir = self$dataset_dir)
    invisible(NULL)
  },

  check_exists = function() {
    if (identical(self$mode, "coco")) {
      fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)
    } else {
      fs::dir_exists(self$image_dir) && fs::dir_exists(self$label_dir)
    }
  },

  load_annotations = function() {
    if (identical(self$mode, "coco")) {
      ann <- jsonlite::fromJSON(self$annotation_file)
      self$categories <- ann$categories
      self$images <- ann$images
      self$annotations <- ann$annotations
      if (nrow(self$annotations) > 0) {
        self$annotations_by_image <- split(self$annotations, self$annotations$image_id)
      } else {
        self$annotations_by_image <- list()
      }
      self$image_paths <- fs::path(self$image_dir, self$images$file_name)
      self$classes <- as.character(self$categories$name)
    } else {
      if (fs::file_exists(self$yaml_path)) {
        if (!requireNamespace("yaml", quietly = TRUE)) utils::install.packages("yaml")
        info <- yaml::read_yaml(self$yaml_path)
        self$classes <- as.character(unlist(info$names %||% character()))
      } else {
        self$classes <- character()
      }
      self$image_paths <- sort(
        fs::dir_ls(self$image_dir, regexp = "\\.(jpg|jpeg|png)$", type = "file")
      )
      self$label_paths <- fs::path(
        self$label_dir,
        paste0(fs::path_ext_remove(fs::path_file(self$image_paths)), ".txt")
      )
    }
  },

  .getitem = function(index) {
    img_path <- self$image_paths[index]
    x <- private$read_image(img_path)

    if (identical(self$mode, "coco")) {
      img_info <- self$images[index, ]
      anns <- self$annotations_by_image[[as.character(img_info$id)]]
      if (is.null(anns) || nrow(anns) == 0) {
        boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
        labels <- character()
      } else {
        boxes_xywh <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float())
        boxes <- private$xywh_to_xyxy(boxes_xywh)
        labels <- as.character(self$categories$name[match(anns$category_id, self$categories$id)])
      }
    } else {
      h <- dim(x)[1]; w <- dim(x)[2]
      lbl_path <- self$label_paths[index]
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
      if (fs::file_exists(lbl_path)) {
        df <- tryCatch(
          utils::read.table(lbl_path, col.names = c("cls", "xc", "yc", "bw", "bh")),
          error = function(e) data.frame()
        )
        if (nrow(df) > 0) {
          xc <- df$xc * w; yc <- df$yc * h; bw <- df$bw * w; bh <- df$bh * h
          boxes_cxcywh <- torch::torch_tensor(cbind(xc, yc, bw, bh), dtype = torch::torch_float())
          x1 <- boxes_cxcywh[,1] - boxes_cxcywh[,3] / 2
          y1 <- boxes_cxcywh[,2] - boxes_cxcywh[,4] / 2
          x2 <- boxes_cxcywh[,1] + boxes_cxcywh[,3] / 2
          y2 <- boxes_cxcywh[,2] + boxes_cxcywh[,4] / 2
          boxes <- torch::torch_stack(list(x1, y1, x2, y2), dim = 2)$to(dtype = torch::torch_float())
          idx <- as.integer(df$cls) + 1L
          labels <- if (length(self$classes) >= max(idx, 0)) self$classes[idx] else as.character(idx)
        }
      }
    }

    y <- list(labels = labels, boxes = boxes)
    if (!is.null(self$transform)) x <- self$transform(x)
    if (!is.null(self$target_transform)) y <- self$target_transform(y)
    structure(list(x = x, y = y), class = "image_with_bounding_box")
  },

  .length = function() length(self$image_paths),

  private = list(
    read_image = function(path) {
      ext <- tolower(fs::path_ext(path))
      if (ext %in% c("jpg", "jpeg")) {
        img <- jpeg::readJPEG(path)
      } else if (ext == "png") {
        img <- png::readPNG(path)
      } else {
        cli::cli_abort("Unsupported image format {.val {ext}} in {path}.")
      }
      if (length(dim(img)) == 3 && dim(img)[3] == 4) img <- img[,,1:3, drop = FALSE]
      if (length(dim(img)) == 2) img <- array(rep(img, 3L), dim = c(dim(img), 3L))
      img
    },
    xywh_to_xyxy = function(b) {
      x1 <- b[,1]; y1 <- b[,2]; x2 <- b[,1] + b[,3]; y2 <- b[,2] + b[,4]
      torch::torch_stack(list(x1, y1, x2, y2), dim = 2)$to(dtype = torch::torch_float())
    },
    resolve_coco_dirs = function(dataset_dir, split) {
      # 1) <dataset>/<split>/_annotations.coco.json
      ann1 <- fs::path(dataset_dir, split, "_annotations.coco.json")
      if (fs::file_exists(ann1) && fs::dir_exists(fs::path(dataset_dir, split))) {
        return(list(
          image_dir = fs::path(dataset_dir, split),
          annotation_file = ann1
        ))
      }
      # 2) <dataset>/<dataset>/<split>/_annotations.coco.json  (double nest)
      ann2 <- fs::path(dataset_dir, fs::path_file(dataset_dir), split, "_annotations.coco.json")
      if (fs::file_exists(ann2) && fs::dir_exists(fs::path(dataset_dir, fs::path_file(dataset_dir), split))) {
        return(list(
          image_dir = fs::path(dataset_dir, fs::path_file(dataset_dir), split),
          annotation_file = ann2
        ))
      }
      # 3) Fallback: search recursively for a '<split>' dir with the annotation file
      cands <- fs::dir_ls(dataset_dir, recurse = 3, type = "file",
                          regexp = paste0("[/\\\\]", split, "[/\\\\]_annotations\\.coco\\.json$"))
      if (length(cands) >= 1) {
        ann <- cands[[1]]
        return(list(
          image_dir = fs::path_dir(ann),
          annotation_file = ann
        ))
      }
      FALSE
    },
    resolve_yolo_dirs = function(dataset_dir, split_candidates) {
      for (sp in split_candidates) {
        img <- fs::path(dataset_dir, sp, "images")
        lbl <- fs::path(dataset_dir, sp, "labels")
        if (fs::dir_exists(img) && fs::dir_exists(lbl)) {
          return(list(
            image_dir = img,
            label_dir = lbl,
            yaml_path = fs::path(dataset_dir, "data.yaml")
          ))
        }
      }
      for (sp in split_candidates) {
        img <- fs::path(dataset_dir, fs::path_file(dataset_dir), sp, "images")
        lbl <- fs::path(dataset_dir, fs::path_file(dataset_dir), sp, "labels")
        if (fs::dir_exists(img) && fs::dir_exists(lbl)) {
          return(list(
            image_dir = img,
            label_dir = lbl,
            yaml_path = fs::path(dataset_dir, fs::path_file(dataset_dir), "data.yaml")
          ))
        }
      }
      FALSE
    }
  )
)
