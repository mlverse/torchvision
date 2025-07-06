#' Vision Transformer models from
#' [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
#'
#' These models load pretrained ViT variants.
#' @param pretrained Whether to load TorchScript weights.
#' @param progress Show progress bar during download.
#' @param ... Ignored.
#' @name model_vit
NULL

vit_torchscript_urls <- list(
  vit_b_16 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_b_16.pt","463defcc1d7ea95f7258736904f895b7","330 MB"),
  vit_b_32 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_b_32.pt","8d3f0a714f3445fd8698987be7e83dcf","330 MB"),
  vit_l_16 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_l_16.pt","a6331544b2a65d80a36789b7dfe7ca26","1.2 GB"),
  vit_l_32 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_l_32.pt","a92dd74b52f94cf4fff1a855d83ec860","1.2 GB"),
  vit_h_14 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_h_14.pt","35493afb4d4ea402e7bb2581441d0bfc","2.4 GB")
)

load_vit_torchscript_model <- function(
    name,
    pretrained = FALSE,
    progress = TRUE,
    ...
) {

  r <- vit_torchscript_urls[[name]]

  cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")

  archive <- download_and_cache(r[1], prefix = name)

  if (tools::md5sum(archive) != r[2])
    runtime_error("Corrupt file! Delete the file in {archive} and try again.")

  jit_load(archive)

}

#' @rdname model_vit
#' @export
model_vit_b_16 <- function(
    name = "vit_b_16",
    pretrained = FALSE,
    progress = TRUE,
    ...
) {

  if (pretrained)
    load_vit_torchscript_model(name, pretrained, progress, ...)
}

#' @rdname model_vit
#' @export
model_vit_b_32 <- function(
    name = "vit_b_32",
    pretrained = FALSE,
    progress = TRUE,
    ...
) {

  if (pretrained)
    load_vit_torchscript_model(name, pretrained, progress, ...)
}

#' @rdname model_vit
#' @export
model_vit_l_16 <- function(
    name = "vit_l_16",
    pretrained = FALSE,
    progress = TRUE,
    ...
) {

  if (pretrained)
    load_vit_torchscript_model(name, pretrained, progress, ...)
}

#' @rdname model_vit
#' @export
model_vit_l_32 <- function(
    name = "vit_l_32",
    pretrained = FALSE,
    progress = TRUE,
    ...
) {

  if (pretrained)
    load_vit_torchscript_model(name, pretrained, progress, ...)
}

#' @rdname model_vit
#' @export
model_vit_h_14 <- function(
    name = "vit_h_14",
    pretrained = FALSE,
    progress = TRUE,
    ...
) {

  if (pretrained)
    load_vit_torchscript_model(name, pretrained, progress, ...)
}