#' Caltech101 Dataset
#'
#' [Caltech101](https://data.caltech.edu/records/20086) Dataset.
#'
#' @param root (string): Root directory where the dataset will be stored or exists.
<<<<<<< HEAD
#' @param target_type (string or list): Type of target to return. Can be "category", "annotation",
=======
#' @param target_type (string or list): Type of target to return. Can be "category", "annotation", 
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
#'   or a list of both. "category" returns the class label, "annotation" returns contour points.
#' @param transform (callable, optional): A function/transform to apply to the image.
#' @param target_transform (callable, optional): A function/transform to apply to the target(s).
#' @param download (bool): If TRUE, downloads the dataset from the internet.
#'
#' @export
caltech101_dataset <- torch::dataset(
  name = "caltech101_dataset",
<<<<<<< HEAD
  initialize = function(root, target_type = "category", transform = NULL,
=======
  initialize = function(root, target_type = "category", transform = NULL, 
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
                        target_transform = NULL, download = FALSE) {
    self$root <- root
    self$transform <- transform
    self$target_transform <- target_transform
<<<<<<< HEAD

    # Validate target_type
    if (is.character(target_type)) target_type <- list(target_type)
    self$target_type <- lapply(target_type, function(t) {
      if (!t %in% c("category", "annotation"))
        stop("target_type must be 'category' and/or 'annotation'")
      t
    })

    if (download) self$download()
    if (!self$check_integrity())
      stop("Dataset not found/corrupted. Use download=TRUE to download")

=======
    
    # Validate target_type
    if (is.character(target_type)) target_type <- list(target_type)
    self$target_type <- lapply(target_type, function(t) {
      if (!t %in% c("category", "annotation")) 
        stop("target_type must be 'category' and/or 'annotation'")
      t
    })
    
    if (download) self$download()
    if (!self$check_integrity())
      stop("Dataset not found/corrupted. Use download=TRUE to download")
    
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
    # Setup categories
    cat_dir <- file.path(self$root, "101_ObjectCategories")
    self$categories <- sort(list.dirs(cat_dir, full.names = FALSE, recursive = FALSE))
    self$categories <- setdiff(self$categories, "BACKGROUND_Google")
<<<<<<< HEAD

=======
    
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
    # Annotation name mapping
    name_map <- list(
      Faces = "Faces_2", Faces_easy = "Faces_3",
      Motorbikes = "Motorbikes_16", airplanes = "Airplanes_Side_2"
    )
<<<<<<< HEAD
    self$anno_cats <- sapply(self$categories, function(x)
      if (x %in% names(name_map)) name_map[[x]] else x)

=======
    self$anno_cats <- sapply(self$categories, function(x) 
      if (x %in% names(name_map)) name_map[[x]] else x)
    
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
    # Build index
    self$index <- integer()
    self$y <- integer()
    for (i in seq_along(self$categories)) {
      imgs <- list.files(file.path(cat_dir, self$categories[i]), "\\.jpg$")
      nums <- as.integer(sub("image_(\\d+)\\.jpg", "\\1", imgs))
      self$index <- c(self$index, nums)
      self$y <- c(self$y, rep(i, length(imgs)))
    }
  },
  .getitem = function(i) {
    # Load image
    cat_idx <- self$y[i]
    cat <- self$categories[cat_idx]
    img_path <- file.path(
      self$root, "101_ObjectCategories", cat,
      sprintf("image_%04d.jpg", self$index[i])
    )
    img <- magick::image_read(img_path)
    img <- magick::image_convert(img, "RGB")
    img <- as.integer(magick::image_data(img))
    img <- torch::torch_tensor(img, dtype = torch::torch_uint8())
    img <- img$permute(c(3, 1, 2)) # HWC -> CHW
<<<<<<< HEAD

=======
    
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
    # Load targets
    targets <- list()
    for (t in self$target_type) {
      if (t == "category") {
        targets <- c(targets, cat_idx)
      } else if (t == "annotation") {
        anno_path <- file.path(
          self$root, "Annotations", self$anno_cats[cat_idx],
          sprintf("annotation_%04d.mat", self$index[i])
        )
        mat <- R.matlab::readMat(anno_path)
        targets <- c(targets, list(mat$obj.contour))
      }
    }
    target <- if (length(targets) > 1) targets else targets[[1]]
<<<<<<< HEAD

    # Apply transforms
    if (!is.null(self$transform)) img <- self$transform(img)
    if (!is.null(self$target_transform)) target <- self$target_transform(target)

=======
    
    # Apply transforms
    if (!is.null(self$transform)) img <- self$transform(img)
    if (!is.null(self$target_transform)) target <- self$target_transform(target)
    
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
    list(x = img, y = target)
  },
  .length = function() length(self$index),
  download = function() {
    if (self$check_integrity()) return()
<<<<<<< HEAD

=======
    
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
    urls <- c(
      "https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
      "https://drive.google.com/uc?id=175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m"
    )
    md5s <- c("b224c7392d521a49829488ab0f1120d9", "6f83eeb1f24d99cab4eb377263132c91")
    files <- c("101_ObjectCategories.tar.gz", "Annotations.tar")
<<<<<<< HEAD

=======
    
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
    for (i in 1:2) {
      dest <- file.path(self$root, files[i])
      utils::download.file(urls[i], dest)
      if (tools::md5sum(dest) != md5s[i])
        stop("Downloaded file corrupt. Delete and retry.")
      utils::untar(dest, exdir = self$root)
    }
  },
  check_integrity = function() {
    all(file.exists(file.path(self$root, c("101_ObjectCategories", "Annotations"))))
  }
<<<<<<< HEAD
)
=======
)
>>>>>>> ff616daa0b5351af066d946532c60ea5e750a1a1
