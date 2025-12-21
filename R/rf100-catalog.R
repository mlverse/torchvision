#' RF100 Dataset Catalog
#'
#' A comprehensive catalog of all RF100 (RoboFlow 100) datasets available in torchvision.
#' This data frame contains metadata about each dataset including descriptions, sizes,
#' available splits, and collection information.
#'
#' @format A data frame with datasets as rows and 13 columns:
#' \describe{
#'   \item{collection}{Collection name (biology, medical, infrared, damage, underwater, document)}
#'   \item{dataset}{Dataset identifier used in collection functions}
#'   \item{description}{Brief description of the dataset and its purpose}
#'   \item{task}{Machine learning task type (currently all "object_detection")}
#'   \item{train_size_mb}{Size of training split in megabytes}
#'   \item{test_size_mb}{Size of test split in megabytes}
#'   \item{valid_size_mb}{Size of validation split in megabytes}
#'   \item{total_size_mb}{Total size across all splits in megabytes}
#'   \item{has_train}{Logical indicating if training split is available}
#'   \item{has_test}{Logical indicating if test split is available}
#'   \item{has_valid}{Logical indicating if validation split is available}
#'   \item{function_name}{R function name to load this dataset's collection}
#'   \item{roboflow_url}{URL to the collection on RoboFlow Universe}
#' }
#'
#' @examples
#' \dontrun{
#' # View the complete catalog
#' data(rf100_catalog)
#' View(rf100_catalog)
#'
#' # See all biology datasets
#' subset(rf100_catalog, collection == "biology")
#'
#' # Find large datasets (> 100 MB)
#' subset(rf100_catalog, total_size_mb > 100)
#' }
#'
#' @seealso [search_rf100()], [get_rf100_catalog()]
"rf100_catalog"

#' Search RF100 Dataset Catalog
#'
#' Search through all RF100 datasets by keywords in name or description,
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
#' search_rf100(collection = "medical")
#'
#' # Find datasets about cells
#' search_rf100("cell")
#'
#' # Find photovoltaic/solar datasets
#' search_rf100("solar")
#' search_rf100("photovoltaic")
#'
#' # Find all biology datasets with "cell" in name/description
#' search_rf100("cell", collection = "biology")
#'
#' # List all available datasets
#' search_rf100()
#' }
#'
#' @seealso [get_rf100_catalog()], [rf100_catalog]
#' @export
search_rf100 <- function(keyword = NULL, collection = NULL) {
  # Load the catalog
  utils::data("rf100_catalog", package = "torchvision", envir = environment())

  result <- rf100_catalog

  # Filter by collection first
  if (!is.null(collection)) {
    valid_collections <- c("biology", "medical", "infrared", "damage", "underwater", "document")
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

#' Get Complete RF100 Catalog
#'
#' Returns the complete catalog of all RF100 datasets with their metadata.
#' This is a convenience function that loads and returns the rf100_catalog data.
#'
#' @return A data frame with all RF100 datasets and their metadata.
#'
#' @examples
#' \dontrun{
#' # Get complete catalog
#' catalog <- get_rf100_catalog()
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
#' @seealso [search_rf100()], [rf100_catalog]
#' @export
get_rf100_catalog <- function() {
  utils::data("rf100_catalog", package = "torchvision", envir = environment())
  rf100_catalog
}

#' List Datasets in an RF100 Collection
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
#' list_rf100_datasets("biology")
#'
#' # List all medical datasets
#' list_rf100_datasets("medical")
#' }
#'
#' @seealso [search_rf100()], [get_rf100_catalog()]
#' @export
list_rf100_datasets <- function(collection) {
  valid_collections <- c("biology", "medical", "infrared", "damage", "underwater", "document")
  if (!collection %in% valid_collections) {
    stop("Invalid collection. Must be one of: ", paste(valid_collections, collapse = ", "))
  }

  utils::data("rf100_catalog", package = "torchvision", envir = environment())
  datasets <- rf100_catalog[rf100_catalog$collection == collection, "dataset"]
  sort(datasets)
}

