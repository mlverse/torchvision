
IMG_EXTENSIONS <-  c('jpg', 'jpeg', 'png', 'tif', 'tiff') # 'ppm', 'bmp', 'pgm', 'webp'

has_file_allowed_extension <- function(filename, extensions) {
  tolower(fs::path_ext(filename)) %in% tolower(extensions )
}

is_image_file <- function(filename) {
  has_file_allowed_extension(filename, IMG_EXTENSIONS)
}

folder_make_dataset <- function(directory, class_to_idx, extensions = NULL, is_valid_file = NULL) {
  directory <- normalizePath(directory)

  both_none <- is.null(extensions) && is.null(is_valid_file)
  both_something <- !is.null(extensions) && ! is.null(is_valid_file)

  if (both_none || both_something)
    value_error("Both extensions and is_valid_file cannot be None or not None at the same time")

  if (!is.null(extensions)) {
    is_valid_file <- function(filename) {
      has_file_allowed_extension(filename, extensions)
    }
  }

  paths <- c()
  indexes <- c()

  for (target_class in sort(names(class_to_idx))) {

    class_index <- class_to_idx[target_class]
    target_dir <- fs::path_join(c(directory, target_class))

    if (!fs::is_dir(target_dir))
      next

    fnames <- fs::dir_ls(target_dir, recurse = TRUE)
    fnames <- fnames[is_valid_file(fnames)]

    paths <- c(paths, fnames)
    indexes <- c(indexes, rep(class_index, length(fnames)))
  }

  list(
    paths,
    indexes
  )
}

folder_dataset <- torch::dataset(
  name = "folder",
  initialize = function(root, loader, extensions = NULL, transform = NULL,
                        target_transform = NULL, is_valid_file = NULL) {

    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform

    class_to_idx <- self$.find_classes(root)
    samples <- folder_make_dataset(self$root, class_to_idx, extensions, is_valid_file)

    if (length(samples[[1]]) == 0) {

      msg <- glue::glue("Found 0 files in subfolders of {self$root}")
      if (!is.null(extensions)) {
        msg <- paste0(msg, glue::glue("\nSupported extensions are {paste(extensions, collapse=',')}"))
      }

      runtime_error(msg)
    }

    self$loader <- loader
    self$extensions <- extensions

    self$classes <- names(class_to_idx)
    self$class_to_idx <- class_to_idx
    self$samples <- samples
    self$targets <- samples[[2]]

  },
  .find_classes = function(dir) {
    dirs <- fs::dir_ls(dir, recurse = FALSE, type = "directory")
    dirs <- sapply(fs::path_split(dirs), function(x) tail(x, 1))
    class_too_idx <- seq_along(dirs)
    names(class_too_idx) <- sort(dirs)
    class_too_idx
  },
  .getitem = function(index) {

    path <- self$samples[[1]][index]
    target <- self$samples[[2]][index]

    sample <- self$loader(path)

    if (!is.null(self$transform))
      sample <- self$transform(sample)

    if (!is.null(self$target_transform))
      target <- self$target_transform(target)

    list(x = sample, y = target)
  },
  .length = function() {
    length(self$samples[[1]])
  }
)

#' Load an Image using ImageMagick
#'
#' Load an image located at `path` using the `{magick}` package.
#'
#' @param path path to the image to load from.
#'
#' @export
magick_loader <- function(path) {
  magick::image_read(path)
}


#' Base loader
#'
#' Loads an image using `jpeg`, `png` or `tiff` packages depending on the
#' file extension.
#'
#' @param path path to the image to load from
#'
#' @export
base_loader <- function(path) {

  ext <- tolower(fs::path_ext(path))

  if (ext %in% c("jpg", "jpeg"))
    img <- jpeg::readJPEG(path)
  else if (ext %in% c("png"))
    img <- png::readPNG(path)
  else if (ext %in% c("tif", "tiff"))
    img <- tiff::readTIFF(path)
  else
    runtime_error("unknown extension {ext} in path {path}")

  if (length(dim(img)) == 2)
    img <- abind::abind(img, img, img, along = 3)
  else if (length(dim(img)) == 3 && dim(img)[1] == 1)
    img <- abind::abind(img, img, img, along = 1)

  img
}


#' Create an image folder dataset
#'
#' A generic data loader for images stored in folders.
#'   See `Details` for more information.
#'
#' @details This function assumes that the images for each class are contained
#'   in subdirectories of `root`. The names of these subdirectories are stored
#'   in the `classes` attribute of the returned object.
#'
#' An example folder structure might look as follows:
#'
#' ```
#' root/dog/xxx.png
#' root/dog/xxy.png
#' root/dog/xxz.png
#'
#' root/cat/123.png
#' root/cat/nsdf3.png
#' root/cat/asd932_.png
#' ```
#'
#' @param root Root directory path.
#' @param loader A function to load an image given its path.
#' @param transform  A function/transform that takes in an PIL image and returns
#'   a transformed version. E.g, [transform_random_crop()].
#' @param target_transform A function/transform that takes in the target and
#'   transforms it.
#' @param is_valid_file A function that takes path of an Image file and check if
#'   the file is a valid file (used to check of corrupt files)
#'
#' @family datasets
#'
#' @importFrom torch dataset
#' @export
image_folder_dataset <- dataset(
  "image_folder",
  inherit = folder_dataset,
  initialize = function(root, transform=NULL, target_transform=NULL,
                        loader=NULL, is_valid_file=NULL) {

    if (is.null(loader))
      loader <- base_loader

    if (!is.null(is_valid_file))
      extensions <- NULL
    else
      extensions <- IMG_EXTENSIONS

    super$initialize(root, loader, extensions, transform=transform,
                     target_transform=target_transform,
                     is_valid_file=is_valid_file)
    self$imgs <- self$samples
  }
)
