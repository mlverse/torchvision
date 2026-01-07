#' Cityscapes Dataset for Instance and Semantic Segmentation
#'
#' The Cityscapes dataset contains street scene images from 50 European cities
#' with pixel-level annotations for semantic and instance segmentation tasks.
#' This dataset is widely used for urban scene understanding research.
#'
#' @param root Character. Root directory where dataset is stored (default: `tempdir()`).
#' @param split Character. One of "train", "val", or "test" (default: "train").
#' @param mode Character. Annotation quality: "fine" for high-quality dense annotations
#'   or "coarse" for weaker annotations (default: "fine").
#' @param target_type Character vector. Types of annotations to load. Can include:
#'   \itemize{
#'     \item "instance": Instance segmentation masks (each object has unique ID)
#'     \item "semantic": Semantic segmentation masks (class IDs only)
#'     \item "polygon": Polygon annotations in JSON format
#'     \item "color": Color-coded visualization of annotations
#'   }
#'   Multiple types can be specified (default: "instance").
#' @param transform Function to transform the input image (default: NULL).
#' @param target_transform Function to transform the target annotations (default: NULL).
#'
#' @return A torch dataset object. Each item is a named list:
#' \itemize{
#'   \item `x`: RGB image array of shape (H, W, 3) or transformed tensor
#'   \item `y`: Named list containing requested target types:
#'     \itemize{
#'       \item `instance`: Instance segmentation mask (H, W) with unique IDs per object
#'       \item `semantic`: Semantic segmentation mask (H, W) with class IDs
#'       \item `color`: Color-coded visualization (H, W, 3)
#'       \item `polygon`: Polygon annotations (nested list structure)
#'     }
#' }
#'
#' @section Dataset Structure:
#' The Cityscapes dataset includes:
#' \itemize{
#'   \item 5,000 finely annotated images (2,975 train, 500 val, 1,525 test)
#'   \item 20,000 coarsely annotated images
#'   \item 19 semantic classes for urban scenes
#'   \item Dense pixel-level annotations at 1024x2048 resolution
#'   \item Images from 50 different European cities
#' }
#'
#' @section Semantic Classes:
#' The dataset includes 19 evaluation classes:
#' road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
#' vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle,
#' bicycle.
#'
#' @section Download:
#' Cityscapes requires manual download and registration:
#' \enumerate{
#'   \item Register at \url{https://www.cityscapes-dataset.com/}
#'   \item Download packages:
#'     \itemize{
#'       \item leftImg8bit_trainvaltest.zip (11GB) - RGB images
#'       \item gtFine_trainvaltest.zip (241MB) - Fine annotations
#'       \item gtCoarse.zip (1.3GB) - Coarse annotations (optional)
#'     }
#'   \item Extract to root directory
#' }
#'
#' Expected directory structure:
#' \preformatted{
#' root/
#' ├── leftImg8bit/
#' │   ├── train/
#' │   ├── val/
#' │   └── test/
#' ├── gtFine/
#' │   ├── train/
#' │   ├── val/
#' │   └── test/
#' └── gtCoarse/
#'     ├── train/
#'     ├── train_extra/
#'     └── val/
#' }
#'
#' @section Integration with draw_segmentation_masks:
#' The output is directly compatible with \code{\link{draw_segmentation_masks}}:
#' \preformatted{
#' item <- dataset[1]
#' # For instance segmentation
#' overlay <- draw_segmentation_masks(item$x, item$y$instance > 0)
#' # For semantic segmentation
#' overlay <- draw_segmentation_masks(item$x, item$y$semantic == class_id)
#' }
#'
#' @examples
#' \dontrun{
#' # Load Cityscapes with instance segmentation
#' cityscapes_train <- cityscapes_dataset(
#'   root = "~/datasets/cityscapes",
#'   split = "train",
#'   mode = "fine",
#'   target_type = "instance",
#'   transform = transform_to_tensor
#' )
#'
#' # Get first item
#' first <- cityscapes_train[1]
#' first$x  # Image tensor (3, H, W)
#' first$y$instance  # Instance mask (H, W)
#'
#' # Visualize with draw_segmentation_masks
#' # Create boolean mask for all instances
#' mask <- first$y$instance > 0
#' overlay <- draw_segmentation_masks(first$x, mask$unsqueeze(1), alpha = 0.5)
#' tensor_image_browse(overlay)
#'
#' # Load with multiple target types
#' cityscapes_multi <- cityscapes_dataset(
#'   root = "~/datasets/cityscapes",
#'   split = "val",
#'   mode = "fine",
#'   target_type = c("instance", "semantic"),
#'   transform = transform_to_tensor
#' )
#'
#' item <- cityscapes_multi[1]
#' # Semantic mask for specific class (e.g., cars = class 13)
#' car_mask <- item$y$semantic == 13
#' overlay <- draw_segmentation_masks(item$x, car_mask$unsqueeze(1))
#' tensor_image_browse(overlay)
#'
#' # Use with dataloader
#' dl <- torch::dataloader(cityscapes_train, batch_size = 4, shuffle = TRUE)
#' batch <- dl$.iter()$.next()
#' }
#'
#' @family segmentation_dataset
#' @export
cityscapes_dataset <- torch::dataset(
  name = "cityscapes",
  
  # Cityscapes evaluation classes (19 classes used for benchmarks)
  classes = c(
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
  ),
  
  # Class ID mapping (trainId -> name)
  # IDs 0-18 are evaluation classes, 255 is ignore/void
  class_ids = c(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18
  ),
  
  initialize = function(
    root = tempdir(),
    split = "train",
    mode = "fine",
    target_type = "instance",
    transform = NULL,
    target_transform = NULL
  ) {
    
    self$root_path <- root
    self$split <- split
    self$mode <- mode
    self$target_type <- target_type
    self$transform <- transform
    self$target_transform <- target_transform
    
    # Validate parameters
    if (!split %in% c("train", "val", "test")) {
      cli::cli_abort(
        "split must be one of 'train', 'val', or 'test', got {.val {split}}"
      )
    }
    
    if (!mode %in% c("fine", "coarse")) {
      cli::cli_abort(
        "mode must be either 'fine' or 'coarse', got {.val {mode}}"
      )
    }
    
    valid_types <- c("instance", "semantic", "polygon", "color")
    invalid_types <- setdiff(target_type, valid_types)
    if (length(invalid_types) > 0) {
      cli::cli_abort(
        "target_type must be one or more of: {.val {valid_types}}",
        "Got invalid types: {.val {invalid_types}}"
      )
    }
    
    # Check dataset exists
    if (!self$check_exists()) {
      cli::cli_abort(
        c(
          "Cityscapes dataset not found at {.path {self$root_path}}",
          "i" = "Download manually from {.url https://www.cityscapes-dataset.com/downloads/}",
          "i" = "Required packages: leftImg8bit_trainvaltest.zip, gtFine_trainvaltest.zip",
          "i" = "Extract to {.path {root}}"
        )
      )
    }
    
    # Build file lists
    self$images <- self$get_image_list()
    
    if (length(self$images) == 0) {
      cli::cli_abort(
        "No images found for split {.val {split}} in {.path {self$images_dir}}"
      )
    }
    
    cli::cli_inform(
      c(
        "v" = "Loaded {.cls {class(self)[[1]]}} dataset",
        "i" = "Split: {.val {split}}, Mode: {.val {mode}}",
        "i" = "Images: {.val {length(self$images)}}",
        "i" = "Target types: {.val {target_type}}"
      )
    )
  },
  
  get_image_list = function() {
    img_dir <- file.path(self$images_dir, self$split)
    
    if (!dir.exists(img_dir)) {
      cli::cli_abort("Image directory not found: {.path {img_dir}}")
    }
    
    # Find all city subdirectories
    cities <- list.dirs(img_dir, recursive = FALSE, full.names = FALSE)
    
    if (length(cities) == 0) {
      return(character(0))
    }
    
    # Collect all images from all cities
    images <- character(0)
    for (city in cities) {
      city_path <- file.path(img_dir, city)
      city_imgs <- list.files(
        city_path,
        pattern = "_leftImg8bit\\.png$",
        full.names = TRUE
      )
      images <- c(images, city_imgs)
    }
    
    sort(images)
  },
  
  get_target_path = function(img_path, target_type) {
    # Convert image path to target annotation path
    base_name <- basename(img_path)
    base_name <- sub("_leftImg8bit\\.png$", "", base_name)
    city <- basename(dirname(img_path))
    
    target_dir <- file.path(
      self$targets_dir,
      self$split,
      city
    )
    
    # Different suffixes for different annotation types
    suffix <- switch(
      target_type,
      "instance" = if (self$mode == "fine") "_gtFine_instanceIds.png" else "_gtCoarse_instanceIds.png",
      "semantic" = if (self$mode == "fine") "_gtFine_labelIds.png" else "_gtCoarse_labelIds.png",
      "color" = if (self$mode == "fine") "_gtFine_color.png" else "_gtCoarse_color.png",
      "polygon" = "_gtFine_polygons.json"  # Only available for fine mode
    )
    
    file.path(target_dir, paste0(base_name, suffix))
  },
  
  check_exists = function() {
    img_exists <- dir.exists(self$images_dir)
    target_exists <- dir.exists(self$targets_dir)
    img_exists && target_exists
  },
  
  .getitem = function(index) {
    img_path <- self$images[[index]]
    
    # Load image using magick (supports various formats)
    img <- magick::image_read(img_path)
    img_array <- magick::image_data(img, channels = "rgb")
    
    # Convert from (C, H, W) to (H, W, C) format
    x <- aperm(as.numeric(img_array), c(2, 3, 1))
    
    # Load targets based on target_type
    y <- list()
    
    for (ttype in self$target_type) {
      target_path <- self$get_target_path(img_path, ttype)
      
      if (!file.exists(target_path)) {
        cli::cli_warn(
          "Target file not found: {.path {basename(target_path)}}",
          "Returning NULL for {.val {ttype}}"
        )
        y[[ttype]] <- NULL
        next
      }
      
      if (ttype == "polygon") {
        # Load JSON polygon data
        y[[ttype]] <- jsonlite::fromJSON(target_path, simplifyVector = FALSE)
      } else if (ttype == "color") {
        # Load color visualization
        color_img <- magick::image_read(target_path)
        color_array <- magick::image_data(color_img, channels = "rgb")
        y[[ttype]] <- aperm(as.numeric(color_array), c(2, 3, 1))
      } else {
        # Load segmentation masks (instance or semantic)
        # These are 16-bit PNG files
        mask_img <- magick::image_read(target_path)
        mask_array <- magick::image_data(mask_img, channels = "gray")
        
        # Convert from 8-bit representation to actual IDs
        # Cityscapes stores IDs in 16-bit format
        mask_matrix <- as.integer(mask_array[1, , ])
        
        y[[ttype]] <- mask_matrix
      }
    }
    
    # Apply transforms
    if (!is.null(self$transform)) {
      x <- self$transform(x)
    }
    
    if (!is.null(self$target_transform)) {
      y <- self$target_transform(y)
    }
    
    result <- list(x = x, y = y)
    class(result) <- c("image_with_segmentation_mask", class(result))
    result
  },
  
  .length = function() {
    length(self$images)
  },
  
  active = list(
    images_dir = function() {
      file.path(self$root_path, "leftImg8bit")
    },
    
    targets_dir = function() {
      if (self$mode == "fine") {
        file.path(self$root_path, "gtFine")
      } else {
        file.path(self$root_path, "gtCoarse")
      }
    }
  )
)
