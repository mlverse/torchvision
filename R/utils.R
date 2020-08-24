
download_and_cache <- function(url, redownload = FALSE) {

  cache_path <- rappdirs::user_cache_dir("torch")

  fs::dir_create(cache_path)
  path <- file.path(cache_path, fs::path_file(url))

  if (!file.exists(path) || redownload)
    utils::download.file(url, path)

  path
}
