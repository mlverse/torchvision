#' ImageNet Class Labels
#'
#' Utilities for resolving ImageNet-1k class identifiers to their corresponding
#' human readable labels. The labels are retrieved from the same source used by
#' PyTorch's reference implementation.
#'
#' @return A character vector with 1000 entries representing the ImageNet-1k
#'   class labels.
#' @family class_resolution
#' @export
imagenet_classes <- function() {
  url <- "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
  labels <- readLines(url, warn = FALSE)
  labels[nzchar(labels)]
}

#' @rdname imagenet_classes
#' @param id Integer vector of 1-based class identifiers.
#' @return A character vector with the labels associated with `id`.
#' @family class_resolution
#' @export
imagenet_label <- function(id) {
  classes <- imagenet_classes()
  classes[id]
}

imagenet_1k_classes <- imagenet_classes
imagenet_1k_label <- imagenet_label

#' @return A character vector with 21k entries representing the ImageNet-21k
#'   class labels.
#' @family class_resolution
#' @export
imagenet_21k_classes <- function() {
  url <- "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt"
  ids <- readLines(url, warn = FALSE)
  url <- "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt"
  labels <- readLines(url, warn = FALSE)

  data.frame(id = ids, label = labels)
}

#' @rdname imagenet_classes
#' @param id Integer vector of 1-based class identifiers.
#' @return A character vector with the labels associated with `id`.
#' @family class_resolution
#' @export
imagenet_21k_label <- function(id) {
  classes <- imagenet_21k_classes()$label
  classes[id]
}

