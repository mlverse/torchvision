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
#' item <- ds[2]
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
    self$root <- fs::path_expand(root)
    self$base_dir <- fs::path(self$root, "rf100-doc")
    self$split <- match.arg(split)
    self$transform <- transform
    self$target_transform <- target_transform

    # --- Detect COCO flat layout ---
    coco_ann <- fs::path(self$base_dir, self$split, "_annotations.coco.json")
    if (fs::file_exists(coco_ann)) {
      self$mode <- "coco"
      self$image_dir <- fs::path(self$base_dir, self$split)
      self$annotation_file <- coco_ann
      # dataset arg is ignored in COCO mode
    } else {
      # --- YOLO per-dataset fallback ---
      self$mode <- "yolo"
      self$dataset <- match.arg(dataset)
      self$dataset_dir <- fs::path(self$base_dir, self$dataset)
      # Accept both valid/val
      split_candidates <- if (self$split == "valid") c("valid", "val")
      else if (self$split == "val") c("val", "valid")
      else self$split
      # Expected path first
      self$image_dir <- fs::path(self$dataset_dir, self$split, "images")
      self$label_dir <- fs::path(self$dataset_dir, self$split, "labels")
      self$yaml_path <- fs::path(self$dataset_dir, "data.yaml")

      if (!fs::dir_exists(self$image_dir) || !fs::dir_exists(self$label_dir)) {
        # Try resolving alternate nestings and val/valid
        resolved <- private$resolve_yolo_dirs(self$dataset_dir, split_candidates)
        if (!isFALSE(resolved)) {
          self$image_dir <- resolved$image_dir
          self$label_dir <- resolved$label_dir
          self$yaml_path <- resolved$yaml_path
        }
      }

      if (download) {
        self$download()
      }
    }

    if (!self$check_exists()) {
      runtime_error("Dataset not found. You can use `download = TRUE` to download it.")
    }

    self$load_annotations()
  },

  download = function() {
    if (self$mode == "coco") {
      # Nothing to download for flat COCO layout
      return(invisible(NULL))
    }
    if (self$check_exists()) return(invisible(NULL))
    fs::dir_create(self$base_dir, recurse = TRUE)
    resource <- self$resources[self$resources$dataset == self$dataset, ]
    archive <- download_and_cache(resource$url, prefix = class(self)[1])
    # Unzip into base_dir; archive has <dataset>/...
    utils::unzip(archive, exdir = self$base_dir)
    invisible(NULL)
  },

  check_exists = function() {
    if (self$mode == "coco") {
      fs::file_exists(self$annotation_file) && fs::dir_exists(self$image_dir)
    } else {
      fs::dir_exists(self$image_dir) && fs::dir_exists(self$label_dir)
    }
  },

  load_annotations = function() {
    if (self$mode == "coco") {
      ann <- jsonlite::fromJSON(self$annotation_file)
      self$categories <- ann$categories
      self$images <- ann$images
      self$annotations <- ann$annotations
      if (nrow(self$annotations) > 0) {
        self$annotations_by_image <- split(self$annotations, self$annotations$image_id)
      } else {
        self$annotations_by_image <- list()
      }
      # Build image paths from file_name relative to split dir
      self$image_paths <- fs::path(self$image_dir, self$images$file_name)
      # class names
      self$classes <- as.character(self$categories$name)
    } else {
      # YOLO: classes from data.yaml if present
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

    if (self$mode == "coco") {
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
      # YOLO txt per image
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
          # cxcywh -> xyxy
          x1 <- boxes_cxcywh[,1] - boxes_cxcywh[,3] / 2
          y1 <- boxes_cxcywh[,2] - boxes_cxcywh[,4] / 2
          x2 <- boxes_cxcywh[,1] + boxes_cxcywh[,3] / 2
          y2 <- boxes_cxcywh[,2] + boxes_cxcywh[,4] / 2
          boxes <- torch::torch_stack(list(x1, y1, x2, y2), dim = 2)$to(dtype = torch::torch_float())
          idx <- as.integer(df$cls) + 1L
          if (length(self$classes) >= max(idx, 0)) labels <- self$classes[idx] else labels <- as.character(idx)
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
      if (length(dim(img)) == 3 && dim(img)[3] == 4) img <- img[,,1:3, drop = FALSE] # drop alpha
      if (length(dim(img)) == 2) img <- array(rep(img, 3L), dim = c(dim(img), 3L))   # gray->RGB
      img
    },
    xywh_to_xyxy = function(b) {
      # b: (N,4) [x,y,w,h]
      x1 <- b[,1]
      y1 <- b[,2]
      x2 <- b[,1] + b[,3]
      y2 <- b[,2] + b[,4]
      torch::torch_stack(list(x1, y1, x2, y2), dim = 2)$to(dtype = torch::torch_float())
    },
    resolve_yolo_dirs = function(dataset_dir, split_candidates) {
      # Try <dataset>/<split>/{images,labels}
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
      # Try <dataset>/<dataset>/<split>/{images,labels} (double nest)
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

