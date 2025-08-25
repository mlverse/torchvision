#' @include dataset-rf100-underwater.R
NULL

#' RF100 Document Dataset Collection
#'
#' Loads one of the RF100 document object detection datasets. COCO format,
#' per dataset folders, and train valid test splits are expected.
#'
#' @param dataset One of "tweeter_post", "tweeter_profile", "document_part",
#'   "activity_diagram", "signature", "paper_part", "tabular_data", "paragraph".
#' @inheritParams rf100_underwater_collection
#' @inherit rf100_underwater_collection return
#'
#' @examples
#' \dontrun{
#' devtools::load_all()
#' ds <- rf100_document_collection(
#'   dataset = "tweeter_post",
#'   split = "train",
#'   transform = transform_to_tensor,
#'   download = TRUE
#' )
#' item <- ds[1]
#' item$y$labels
#' item$y$boxes
#' boxed_img <- draw_bounding_boxes(item)
#' tensor_image_browse(boxed_img)
#' }
#' @family detection_dataset
#' @export
rf100_document_collection <- torch::dataset(
  name = "rf100_document_collection",
  inherit = rf100_underwater_collection,

  resources = data.frame(
    dataset = c("tweeter_post","tweeter_profile","document_part",
                "activity_diagram","signature","paper_part",
                "tabular_data","paragraph"),
    url = paste0(
      "https://huggingface.co/datasets/akankshakoshti/rf100-doc/resolve/main/",
      c("tweeter_post.zip","tweeter_profile.zip","document_part.zip",
        "activity_diagram.zip","signature.zip","paper_part.zip",
        "tabular_data.zip","paragraph.zip"),
      "?download=1"
    ),
    md5 = c(
      "f52c47bf174efb4d664898afac169e05",
      "f92174ce8969c105c1a822f2a2313d78",
      "8ddd1418f0a547491f99986fe6e77bdd",
      "c6774082f83cc34ad6f54330adaf0ff8",
      "8647c40a765248fee6d4fb19cad2a8e1",
      "5808824dfd7e5caafa56b024702c77af",
      "eb9231f3a9083a9f601ebedfd2c06359",
      NA_character_
    )
  ),

  initialize = function(
    dataset = c("tweeter_post","tweeter_profile","document_part",
                "activity_diagram","signature","paper_part",
                "tabular_data","paragraph"),
    split = c("train","test","valid"),
    root = tempdir(),
    download = FALSE,
    transform = NULL,
    target_transform = NULL
  ) {
    self$dataset <- match.arg(dataset)
    self$split   <- match.arg(split)
    self$root    <- fs::path_expand(root)
    self$transform <- transform
    self$target_transform <- target_transform

    # dataset scoped dirs; tolerate optional double nesting
    self$dataset_dir <- fs::path(self$root, "rf100-doc", self$dataset)
    self$split_dir   <- fs::path(self$dataset_dir, self$split)
    if (!fs::dir_exists(self$split_dir)) {
      alt <- fs::path(self$dataset_dir, self$dataset, self$split)
      if (fs::dir_exists(alt)) self$split_dir <- alt
    }
    self$image_dir <- self$split_dir
    self$annotation_file <- fs::path(self$split_dir, "_annotations.coco.json")

    res <- subset(self$resources, dataset == self$dataset, drop = TRUE)
    self$archive_url <- res$url

    if (download) self$download()
    if (!self$check_exists()) runtime_error("Dataset not found. Use download=TRUE or check paths.")
    self$load_annotations()
  },

  # override to support png and alpha; reuse rest from parent
  .getitem = function(index) {
    img_path <- self$image_paths[index]
    ext <- tolower(fs::path_ext(img_path))
    x <- if (ext %in% c("jpg","jpeg")) jpeg::readJPEG(img_path)
    else if (ext == "png")        png::readPNG(img_path)
    else                          jpeg::readJPEG(img_path)
    if (length(dim(x)) == 3 && dim(x)[3] == 4) x <- x[,,1:3, drop = FALSE]
    if (length(dim(x)) == 2) x <- array(rep(x, 3L), dim = c(dim(x), 3L))

    info <- self$images[index, ]
    anns <- self$annotations_by_image[[as.character(info$id)]]

    if (is.null(anns) || nrow(anns) == 0) {
      boxes <- torch::torch_zeros(c(0, 4), dtype = torch::torch_float())
      labels <- character()
    } else {
      b <- torch::torch_tensor(do.call(rbind, anns$bbox), dtype = torch::torch_float()) # x,y,w,h
      boxes <- torch::torch_stack(list(b[,1], b[,2], b[,1]+b[,3], b[,2]+b[,4]), dim = 2)
      labels <- as.character(self$categories$name[match(anns$category_id, self$categories$id)])
    }

    y <- list(labels = labels, boxes = boxes)
    if (!is.null(self$transform)) x <- self$transform(x)
    if (!is.null(self$target_transform)) y <- self$target_transform(y)
    structure(list(x = x, y = y), class = "image_with_bounding_box")
  }
)
