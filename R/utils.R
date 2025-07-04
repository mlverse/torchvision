
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


#' Convert a magick image to a (3, H, W) torch_tensor
#'
#' Converts an image read with magick::image_read() into a normalized torch tensor
#' in channel-first (CHW) format with values in [0,1].
#'
#' @param img A magick image object
#'
#' @return A torch_tensor of shape (3, H, W)
#' @export
transform_to_tensor <- function(img) {
  img_data <- magick::image_data(img, channels = "rgb")
  arr <- as.numeric(img_data)
  arr <- arr / 255  # only ONCE
  arr <- aperm(arr, c(3, 2, 1))  # HWC to CHW
  torch::torch_tensor(arr, dtype = torch::torch_float())
}

visualize_tensor_image <- function(img_tensor) {
  img_array <- as.array(img_tensor$permute(c(2, 3, 1)))
  img_array <- img_array / max(img_array)
  img_array <- pmin(pmax(img_array, 0), 1)
  grid::grid.raster(img_array)
}

#' Auto-orient an image based on EXIF metadata
#'
#' Some images contain orientation metadata that causes them to display rotated
#' when loaded. This helper inspects the EXIF orientation tag and applies the
#' corresponding rotation and/or flip so that the image is returned in the
#' standard orientation.
#'
#' @param img A `magick-image` object as returned by [magick::image_read()].
#'
#' @return The reoriented `magick-image`.
#' @export
image_auto_orient <- function(img) {
  attrs <- magick::image_attributes(img)
  ori <- attrs$value[attrs$property == "exif:Orientation"]
  ori <- if (length(ori)) as.integer(ori) else 1L

  switch(ori,
         img,
         magick::image_flop(img),            # 2
         magick::image_rotate(img, 180),     # 3
         magick::image_flip(img),            # 4
         magick::image_transpose(img),       # 5
         magick::image_rotate(img, 90),      # 6
         magick::image_transverse(img),      # 7
         magick::image_rotate(img, 270),     # 8
         img
  )
}

