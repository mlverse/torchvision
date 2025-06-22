#' Oxford Flowers 102 Dataset
#'
#' Loads the Oxford 102 Category Flower Dataset. This dataset consists of 102 flower categories, with between 40 and 258 images per class. 
#'
#' The dataset is split into:
#' - `"train"`: training subset with labels.
#' - `"val"`: validation subset with labels.
#' - `"test"`: test subset with labels (used for evaluation).
#'
#' @inheritParams mnist_dataset
#' @param root (string, optional): Root directory for dataset storage,
#' the dataset will be stored under `root/flowers102`.
#' @param split (string, optional): One of `"train"`, `"val"`, or `"test"`,
#' defines which subset of the data to load. Default is `"train"`.
#'
#' @return An object of class \code{flowers102_dataset}, which behaves like a torch dataset.
#' Each element is a named list:
#' - `x`: a 3 x H x W numeric array representing an RGB image.
#' - `y`: an integer label indicating the class index.
#'
#' @examples
#' \dontrun{
#' flowers <- flowers102_dataset(split = "train", download = TRUE)
#'
#' # Define a custom collate function to resize images in the batch
#' resize_collate_fn <- function(batch) {
#'   xs <- lapply(batch, function(sample) {
#'     sample$x <- torch_tensor(sample$x)
#'     transform_resize(sample$x, c(224, 224))
#'   })
#'   xs <- torch_stack(xs)
#'   ys <- sapply(batch, function(sample) sample$y)
#'   list(x = xs, y = ys)
#' }
#'
#' dl <- dataloader(dataset = flowers, batch_size = 4, collate_fn = resize_collate_fn)
#' batch <- dataloader_next(dataloader_make_iter(dl))
#' batch$x  # batched image tensors resized to 224x224
#' batch$y  # batched integer labels
#' }
#'
#' @name flowers102_dataset
#' @aliases flowers102_dataset
#' @title Oxford Flowers 102 Dataset
#' @export
flowers102_dataset <- dataset(
  name = "flowers102",
  classes = c(
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
    "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia?",
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
  ),
  resources = list(
    c("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
    c("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
    c("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c")
  ),

  initialize = function(
    root = tempdir(),
    split = "train",
    transform = NULL,
    target_transform = NULL,
    download = FALSE
  ) {

    self$root_path <- root
    self$split <- match.arg(split, c("train", "val", "test"))
    self$transform <- transform
    self$target_transform <- target_transform
    self$classes <- self$classes

    if (download) {
      cli_inform("Oxford Flowers 102 (~350MB) will be downloaded and processed if not already cached.")
      self$download()
    }
    if (!self$check_exists(self$split))
      cli_abort("Dataset not found. You can use `download = TRUE` to download it.")

    meta <- readRDS(file.path(self$processed_folder, glue::glue("{self$split}.rds")))
    self$samples <- meta$samples
    self$labels <- meta$labels
    cli_inform("Split '{self$split}' loaded with {length(self$samples)} samples.")
  },

  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]

    img <- jpeg::readJPEG(img_path)
    img <- img * 255
    img <- aperm(img, c(3, 1, 2))

    if (!is.null(self$transform))
      img <- self$transform(img)

    if (!is.null(self$target_transform))
      label <- self$target_transform(label)

    list(x = img, y = label_idx)
  },

  .length = function() {
    length(self$samples)
  },

  download = function() {
    if (self$check_exists(self$split)) {
      cli_inform("Split '{self$split}' is already processed and cached.")
      return(NULL)
    }
    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      filename <- basename(r[1])
      destpath <- file.path(self$raw_folder, filename)
      archive <- download_and_cache(r[1], prefix = class(self)[1])
      fs::file_copy(archive, destpath, overwrite = TRUE)

      if (!tools::md5sum(destpath) == r[2])
        cli_abort("Corrupt file! Delete the file in {archive} and try again.")
    }

    cli_inform("Extracting images and processing dataset...")
    untar(file.path(self$raw_folder, "102flowers.tgz"), exdir = self$raw_folder)

    if (!requireNamespace("R.matlab", quietly = TRUE)) {
      cli_abort("Package 'R.matlab' is needed for this dataset. Please install it.")
    }
    labels <- R.matlab::readMat(file.path(self$raw_folder, "imagelabels.mat"))$labels
    setids <- R.matlab::readMat(file.path(self$raw_folder, "setid.mat"))

    set_map <- list(
      train = as.integer(setids$trnid),
      val = as.integer(setids$valid),
      test = as.integer(setids$tstid)
    )

    split_name <- self$split
    idxs <- set_map[[split_name]]
    jpg_dir <- file.path(self$raw_folder, "jpg")
    paths <- file.path(jpg_dir, glue::glue("image_{sprintf('%05d', idxs)}.jpg"))
    lbls <- as.integer(labels[idxs])
    saveRDS(list(samples = paths, labels = lbls), file.path(self$processed_folder, glue::glue("{split_name}.rds")))
  },

  check_exists = function(split) {
    fs::file_exists(file.path(self$processed_folder, glue::glue("{split}.rds")))
  },

  active = list(
    raw_folder = function() file.path(self$root_path, "flowers102", "raw"),
    processed_folder = function() file.path(self$root_path, "flowers102", "processed")
  )
)
