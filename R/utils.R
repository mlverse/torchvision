
download_and_cache <- function(url, redownload = FALSE) {

  cache_path <- rappdirs::user_cache_dir("torch")

  fs::dir_create(cache_path)
  path <- file.path(cache_path, fs::path_file(url))

  if (!file.exists(path) || redownload) {
    # we should first download to a temporary file because
    # download probalems could cause hard to debug errors.
    tmp <- tempfile(fileext = fs::path_ext(path))
    on.exit({try({fs::file_delete(tmp)}, silent = TRUE)}, add = TRUE)

    withr::with_options(
      list(timeout = 600),
      utils::download.file(url, tmp, mode = "wb")
    )
    fs::file_move(tmp, path)
  }

  path
}
