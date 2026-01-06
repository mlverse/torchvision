
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


#' Download a file with fallback to alternative mirror URLs
#'
#' Attempts to download a file from a list of URLs, trying each one in order
#' until a successful download occurs. Useful for datasets with unreliable
#' hosting or multiple mirror sites.
#'
#' @param urls A character vector of URLs to try, in order of preference.
#' @param expected_md5 Expected MD5 checksum (optional). If provided, the
#'   downloaded file's checksum will be verified.
#' @param prefix Subdirectory within the cache directory.
#' @param redownload Force redownload even if file exists in cache.
#' @param n_retries Number of times to retry each URL before moving to next.
#' @param delay Delay in seconds between retries.
#'
#' @return Path to the downloaded and cached file.
#' @keywords internal
download_with_fallback <- function(
    urls,
    expected_md5 = NULL,
    prefix = NULL,
    redownload = FALSE,
    n_retries = 2,
    delay = 1
) {
  cache_path <- rappdirs::user_cache_dir("torch")

  fs::dir_create(cache_path)
  if (!is.null(prefix)) {
    cache_path <- file.path(cache_path, prefix)
  }
  try(fs::dir_create(cache_path, recurse = TRUE), silent = TRUE)

  # Use the filename from the first URL for caching
  filename <- fs::path_sanitize(fs::path_file(urls[1]))
  path <- file.path(cache_path, filename)

  # Return cached file if it exists and redownload is FALSE

  if (file.exists(path) && !redownload) {
    if (!is.null(expected_md5)) {
      actual_md5 <- tools::md5sum(path)
      if (actual_md5 == expected_md5) {
        return(path)
      }
      # MD5 mismatch, need to redownload
      cli::cli_inform("Cached file checksum mismatch, redownloading...")
      fs::file_delete(path)
    } else {
      return(path)
    }
  }

  # Try each URL in sequence
  last_error <- NULL
  tried_urls <- character(0)

  for (url in urls) {
    tried_urls <- c(tried_urls, url)

    for (attempt in seq_len(n_retries)) {
      tryCatch({
        cli::cli_inform("Attempting download from: {.url {url}} (attempt {attempt}/{n_retries})")

        tmp <- tempfile(fileext = paste0(".", fs::path_ext(filename)))
        on.exit({try({fs::file_delete(tmp)}, silent = TRUE)}, add = TRUE)

        withr::with_options(
          list(timeout = max(600, getOption("timeout", default = 0))),
          utils::download.file(url, tmp, mode = "wb", quiet = FALSE)
        )

        # Verify MD5 if provided
        if (!is.null(expected_md5)) {
          actual_md5 <- tools::md5sum(tmp)
          if (actual_md5 != expected_md5) {
            cli::cli_warn("Checksum mismatch for {.url {url}}, trying next mirror...")
            next
          }
        }

        # Success! Move to cache
        fs::file_move(tmp, path)
        cli::cli_inform("Successfully downloaded from {.url {url}}")
        return(path)

      }, error = function(e) {
        last_error <<- e
        cli::cli_warn("Download failed from {.url {url}}: {conditionMessage(e)}")
        if (attempt < n_retries) {
          Sys.sleep(delay)
        }
      })
    }
  }

  # All URLs failed - provide helpful error message
  cli::cli_abort(c(
    "Failed to download file after trying all mirror URLs.",
    "i" = "Tried URLs:",
    paste("  -", tried_urls),
    "",
    "i" = "You can manually download the file and place it in:",
    "  {.path {cache_path}}",
    "",
    "i" = "Last error: {conditionMessage(last_error)}"
  ))
}

#' Validate num_classes parameter
#'
#' @param num_classes Number of classes to validate
#' @param pretrained shall the classes match a pretrained model
#' @param label the label family for the classification head. Shall be "coco" or "ade20k"
#' @keywords internal
#' @noRd
validate_num_classes <- function(num_classes, pretrained, label = "coco") {
  match.arg(label, c("coco","ade20k"))
  if (num_classes <= 0) {
    cli_abort("{.var num_classes} must be positive")
  }
  if (label == "coco" && pretrained && num_classes != 21) {
    cli_abort("Pretrained weights on COCO require {.var num_classes} to be {.val 21}.")
  }
  if (label == "ade20k" && pretrained && num_classes != 150) {
    cli_abort("Pretrained weights on ADE20K require {.var num_classes} to be {.val 150}.")
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
