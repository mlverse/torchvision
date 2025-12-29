#' Dataset Collection Catalog
#'
#' A comprehensive catalog of all collections RF100 (RoboFlow 100) and EMNIST datasets available in torchvision.
#' This data frame contains metadata about each dataset including descriptions, sizes,
#' available splits, and collection information.
#'
#' @format A data frame with datasets as rows and 17 columns:
#' \describe{
#'   \item{collection}{Collection name (biology, medical, infrared, damage, underwater, document, mnist)}
#'   \item{dataset}{Dataset identifier used in collection functions}
#'   \item{description}{Brief description of the dataset and its purpose}
#'   \item{task}{Machine learning task type (currently all "object_detection")}
#'   \item{num_classes}{Number of different object classes}
#'   \item{num_images}{Total images across all splits}
#'   \item{image_width}{Typical image width in pixels}
#'   \item{image_height}{Typical image height in pixels}
#'   \item{train_size_mb}{Size of training split in megabytes}
#'   \item{test_size_mb}{Size of test split in megabytes}
#'   \item{valid_size_mb}{Size of validation split in megabytes}
#'   \item{total_size_mb}{Total size across all splits in megabytes}
#'   \item{has_train}{Is training split available}
#'   \item{has_test}{Is test split available}
#'   \item{has_valid}{Is validation split available}
#'   \item{function_name}{R function name to load this dataset's collection}
#'   \item{roboflow_url}{URL to the collection on RoboFlow Universe}
#' }
#'
#' @examples
#' \dontrun{
#' # View the complete catalog
#' data(collection_catalog)
#' View(collection_catalog)
#'
#' # See all biology datasets
#' subset(collection_catalog, collection == "biology")
#'
#' # Find large datasets (> 100 MB)
#' subset(collection_catalog, total_size_mb > 100)
#' }
#'
#' @seealso [search_collection()], [get_collection_catalog()]
"collection_catalog"

#' Search Collection Catalog
#'
#' Search through all Collection datasets by keywords in name or description,
#' or filter by collection. This makes it easy to discover datasets relevant
#' to your task without browsing each collection individually.
#'
#' @param keyword Character string to search for (case-insensitive). Searches
#'   in both dataset names and descriptions. If NULL, returns all datasets
#'   (optionally filtered by collection).
#' @param collection Filter by collection name. One of: "biology", "medical",
#'   "infrared", "damage", "underwater", "document". If NULL, searches all collections.
#'
#' @return A data frame with matching datasets and their metadata. Returns NULL
#'   invisibly if no matches are found.
#'
#' @examples
#' \dontrun{
#' # Find all medical datasets
#' search_collection(collection = "medical")
#'
#' # Find datasets about cells
#' search_collection("cell")
#'
#' # Find photovoltaic/solar datasets
#' search_collection("solar")
#' search_collection("photovoltaic")
#'
#' # Find all biology datasets with "cell" in name/description
#' search_collection("cell", collection = "biology")
#'
#' # List all available datasets
#' search_collection()
#' }
#'
#' @seealso [get_collection_catalog()], [collection_catalog]
#' @export
search_collection <- function(keyword = NULL, collection = NULL) {
  # Load the catalog
  utils::data("collection_catalog", package = "torchvision", envir = environment())

  result <- collection_catalog

  # Filter by collection first
  if (!is.null(collection)) {
    valid_collections <- c("biology", "medical", "infrared", "damage", "underwater", "document", "mnist")
    if (!collection %in% valid_collections) {
      stop("Invalid collection. Must be one of: ", paste(valid_collections, collapse = ", "))
    }
    result <- result[result$collection == collection, ]
  }

  # Filter by keyword in dataset name or description
  if (!is.null(keyword)) {
    pattern <- tolower(keyword)
    matches <- grepl(pattern, tolower(result$dataset)) |
               grepl(pattern, tolower(result$description))
    result <- result[matches, ]
  }

  if (nrow(result) == 0) {
    message("No datasets found matching criteria")
    return(invisible(NULL))
  }

  # Reset row names for cleaner display
  rownames(result) <- NULL

  result
}

#' Get Complete Collection Catalog
#'
#' Returns the complete catalog of datasets in collections with their metadata.
#' This is a convenience function that loads and returns the collection_catalog data.
#'
#' @return A data frame with all datasets and their metadata.
#'
#' @examples
#' \dontrun{
#' # Get complete catalog
#' catalog <- get_collection_catalog()
#'
#' # View in RStudio
#' View(catalog)
#'
#' # Summary statistics
#' summary(catalog$total_size_mb)
#' table(catalog$collection)
#'
#' # Find smallest dataset
#' catalog[which.min(catalog$total_size_mb), ]
#'
#' # Find largest dataset
#' catalog[which.max(catalog$total_size_mb), ]
#' }
#'
#' @seealso [search_collection()], [collection_catalog]
#' @export
get_collection_catalog <- function() {
  utils::data("collection_catalog", package = "torchvision", envir = environment())
  collection_catalog
}

#' List Datasets in a Collection
#'
#' List all available datasets within a specific RF100 collection.
#'
#' @param collection Collection name. One of: "biology", "medical", "infrared",
#'   "damage", "underwater", "document".
#'
#' @return Character vector of dataset names in the collection.
#'
#' @examples
#' \dontrun{
#' # List all biology datasets
#' list_collection_datasets("biology")
#'
#' # List all medical datasets
#' list_collection_datasets("medical")
#' }
#'
#' @seealso [search_collection()], [get_collection_catalog()]
#' @export
list_collection_datasets <- function(collection) {
  valid_collections <- c("biology", "medical", "infrared", "damage", "underwater", "document", "mnist")
  if (!collection %in% valid_collections) {
    stop("Invalid collection. Must be one of: ", paste(valid_collections, collapse = ", "))
  }

  utils::data("collection_catalog", package = "torchvision", envir = environment())
  datasets <- collection_catalog[collection_catalog$collection == collection, "dataset"]
  sort(datasets)
}

