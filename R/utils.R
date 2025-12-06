
download_and_cache <- function(url, redownload = FALSE, prefix = NULL) {

  cache_path <- rappdirs::user_cache_dir("torch")

  fs::dir_create(cache_path)
  if (!is.null(prefix)) {
    cache_path <- file.path(cache_path, prefix)
  }
  try(fs::dir_create(cache_path, recurse = TRUE), silent = TRUE)
  path <- file.path(cache_path, fs::path_sanitize(fs::path_file(url)))

  if (!file.exists(path) || redownload) {
    # we should first download to a temporary file because
    # download probalems could cause hard to debug errors.
    tmp <- tempfile(fileext = fs::path_ext(path))
    on.exit({try({fs::file_delete(tmp)}, silent = TRUE)}, add = TRUE)

    withr::with_options(
      list(timeout = max(600, getOption("timeout", default = 0))),
      utils::download.file(url, tmp, mode = "wb")
    )
    fs::file_move(tmp, path)
  }

  path
}


#' Validate num_classes parameter
#'
#' @param num_classes Number of classes to validate
#' @keywords internal
#' @noRd
validate_num_classes <- function(num_classes) {
  if (num_classes <= 0) {
    cli_abort("{.var num_classes} must be positive")
  }
}


# add additional checks to release issues created with usethis::use_release_issue()
# https://usethis.r-lib.org/reference/use_release_issue.html
release_bullets <- function() {
  c("Update `po/R-torchvision.pot` file with `potools::po_update()`",
    "Contact translators to collect their translation `.po` files",
    "Compile the translations with `potools::po_compile()`"
  )
}
