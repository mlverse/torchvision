#' Flowers 102 Dataset
#'
#' Loads the Oxford Flowers 102 Dataset, consisting of 102 flower categories with images and labels.
#' The dataset supports different splits: `"train"`, `"val"`, and `"test"`.
#'
#' @param root Character. Root directory for dataset storage. The dataset will be stored under `root/flowers102`.
#' @param split Character. Dataset split to use. One of `"train"`, `"val"`, or `"test"`. Default is `"train"`.
#' @param transform Optional function to transform input images after loading.
#' @param target_transform Optional function to transform labels.
#' @param download Logical. Whether to download the dataset if not found locally. Default is `FALSE`.
#'
#' @return A flowers102_dataset object representing the dataset.
#'
#' @examples
#' \dontrun{
#' root_dir <- tempfile()
#' flowers <- flowers102_dataset(root = root_dir, split = "train", download = TRUE)
#' first_item <- flowers[1]
#' # image tensor of first item
#' first_item$x
#' # label (flower class name) of first item
#' first_item$y
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
  image_size = c(224, 224),
  resources = list(
    c("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
    c("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
    c("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c")
  ),

  initialize = function(root, split = "train", transform = NULL, target_transform = NULL, download = FALSE) {
    self$root_path <- root
    self$split <- match.arg(split, c("train", "val", "test"))
    self$transform <- transform
    self$target_transform <- target_transform

    self$classes <- c(
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily"
    )


    if (download)
      self$download()

    if (!self$check_exists(self$split))
      runtime_error("Dataset not found. Use `download = TRUE` to fetch it.")

    meta <- readRDS(file.path(self$processed_folder, glue::glue("{self$split}.rds")))
    self$samples <- meta$samples
    self$labels <- meta$labels
  },

  .getitem = function(index) {
    img_path <- self$samples[[index]]
    label_idx <- self$labels[[index]]
    label <- self$classes[label_idx]


    img <- magick::image_read(img_path)
    img <- magick::image_resize(img, glue::glue("{self$image_size[1]}x{self$image_size[2]}"))
    img_tensor <- torchvision::transform_to_tensor(img)

    if (!is.null(self$transform))
      img_tensor <- self$transform(img_tensor)

    if (!is.null(self$target_transform))
      label <- self$target_transform(label)

    structure(list(x = img_tensor, y = label), class = "flowers102_item")
  },

  .length = function() {
    length(self$samples)
  },

  download = function() {
    rlang::inform(glue::glue("Downloading Flowers102 split: {self$split}"))

    if (self$check_exists(self$split))
      return(NULL)

    fs::dir_create(self$raw_folder)
    fs::dir_create(self$processed_folder)

    for (r in self$resources) {
      filename <- basename(r[1])
      destpath <- file.path(self$raw_folder, filename)
      p <- download_and_cache(r[1], prefix = class(self)[1])
      fs::file_copy(p, destpath, overwrite = TRUE)

      if (!tools::md5sum(destpath) == r[2])
        runtime_error(sprintf("MD5 mismatch for file: %s", r[1]))
    }

    rlang::inform("Extracting archive and processing metadata...")
    untar(file.path(self$raw_folder, "102flowers.tgz"), exdir = self$raw_folder)

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
    paths <- file.path(jpg_dir, sprintf("image_%05d.jpg", idxs))
    lbls <- as.integer(labels[idxs])
    saveRDS(list(samples = paths, labels = lbls), file.path(self$processed_folder, glue::glue("{split_name}.rds")))

    rlang::inform(glue::glue("Done processing split: {split_name}"))
  },

  check_exists = function(split) {
    fs::file_exists(file.path(self$processed_folder, glue::glue("{split}.rds")))
  },

  active = list(
    raw_folder = function() file.path(self$root_path, "flowers102", "raw"),
    processed_folder = function() file.path(self$root_path, "flowers102", "processed")
  )
)
