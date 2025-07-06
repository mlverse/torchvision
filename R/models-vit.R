#' Vision Transformer Implementation 
#'
#' Vision Transformer models implementation on [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
#' 
#' @param pretrained Whether to load TorchScript weights.
#' @param progress Show progress bar during download.
#' @param ... Additional arguments passed to the model constructor.
#'
#' @family models
#'
#' @name model_vit
NULL

vit_torchscript_urls <- list(
  vit_b_16 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_b_16.pt","463defcc1d7ea95f7258736904f895b7","330 MB"),
  vit_b_32 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_b_32.pt","8d3f0a714f3445fd8698987be7e83dcf","330 MB"),
  vit_l_16 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_l_16.pt","a6331544b2a65d80a36789b7dfe7ca26","1.2 GB"),
  vit_l_32 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_l_32.pt","a92dd74b52f94cf4fff1a855d83ec860","1.2 GB"),
  vit_h_14 = c("https://huggingface.co/datasets/JimmyUnleashed/VisionTransformerModel/resolve/main/vit_h_14.pt","35493afb4d4ea402e7bb2581441d0bfc","2.4 GB")
)

load_vit_torchscript_model <- function(name) {

  r <- vit_torchscript_urls[[name]]

  cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")

  archive <- download_and_cache(r[1], prefix = name)

  if (tools::md5sum(archive) != r[2])
    runtime_error("Corrupt file! Delete the file in {archive} and try again.")

  jit_load(archive)

}

model_vit_base <- function(name, pretrained, progress, ...) {

  if (pretrained) {
    return(load_vit_torchscript_model(name))
  } else {
    args <- list(...)
    config <- switch(name,
      vit_b_16 = list(img_size = 224, patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, num_classes = 1000),
      vit_b_32 = list(img_size = 224, patch_size = 32, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, num_classes = 1000),
      vit_l_16 = list(img_size = 224, patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16, mlp_ratio = 4, num_classes = 1000),
      vit_l_32 = list(img_size = 224, patch_size = 32, embed_dim = 1024, depth = 24, num_heads = 16, mlp_ratio = 4, num_classes = 1000),
      vit_h_14 = list(img_size = 224, patch_size = 14, embed_dim = 1280, depth = 32, num_heads = 16, mlp_ratio = 4, num_classes = 1000),
    )

    config[names(args)] <- args
    do.call(vit_model, config)
  }
}

#' @describeIn model_vit ViT-B/16 model (Base, 16×16 patch size)
#' @export
model_vit_b_16 <- function(pretrained = FALSE, progress = TRUE, ...) {
  model_vit_base("vit_b_16", pretrained, progress, ...)
}

#' @describeIn model_vit ViT-B/32 model (Base, 32×32 patch size)
#' @export
model_vit_b_32 <- function(pretrained = FALSE, progress = TRUE, ...) {
  model_vit_base("vit_b_32", pretrained, progress, ...)
}

#' @describeIn model_vit ViT-L/16 model (Base, 16×16 patch size)
#' @export
model_vit_l_16 <- function(pretrained = FALSE, progress = TRUE, ...) {
  model_vit_base("vit_l_16", pretrained, progress, ...)
}

#' @describeIn model_vit ViT-L/32 model (Base, 32×32 patch size)
#' @export
model_vit_l_32 <- function(pretrained = FALSE, progress = TRUE, ...) {
  model_vit_base("vit_l_32", pretrained, progress, ...)
}

#' @describeIn model_vit ViT-H/14 model (Base, 14×14 patch size)
#' @export
model_vit_h_14 <- function(pretrained = FALSE, progress = TRUE, ...) {
  model_vit_base("vit_h_14", pretrained, progress, ...)
}

vit_model <- nn_module(
  classname = "vit_model",
  initialize = function(
    img_size = 224,
    patch_size = 16,
    in_chans = 3,
    num_classes = 1000,
    embed_dim = 768,
    depth = 12,
    num_heads = 12,
    mlp_ratio = 4,
    qkv_bias = TRUE,
    representation_size = NULL,
    dropout = 0.0
  ) {
    self$patch_embed <- patch_embed(img_size, patch_size, in_chans, embed_dim)
    num_patches <- self$patch_embed$num_patches

    self$class_token <- nn_parameter(torch_zeros(1, 1, embed_dim))
    self$pos_embed <- nn_parameter(torch_zeros(1, num_patches + 1, embed_dim))
    self$pos_drop <- nn_dropout(p = dropout)

    self$blocks <- nn_module_list(
      lapply(1:depth, function(i) {
        encoder_block(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
      })
    )

    self$norm <- nn_layer_norm(embed_dim)

    self$head <- if (is.null(num_classes) || num_classes == 0) {
      nn_identity()
    } else {
      nn_linear(embed_dim, num_classes)
    }

    self$init_weights()
  },

  init_weights = function() {
    nn_init_trunc_normal_(self$pos_embed, std = 0.02)
    nn_init_trunc_normal_(self$class_token, std = 0.02)
    for (m in self$modules) {
      if (inherits(m, "nn_linear")) {
        nn_init_trunc_normal_(m$weight, std = 0.02)
        if (!is.null(m$bias))
          nn_init_zeros_(m$bias)
      } else if (inherits(m, "nn_layer_norm")) {
        nn_init_zeros_(m$bias)
        nn_init_ones_(m$weight)
      }
    }
  },

  forward = function(x) {
    B <- x$size(1)
    x <- self$patch_embed(x)  # (B, num_patches, embed_dim)

    cls_tokens <- self$class_token$expand(c(B, -1, -1))
    x <- torch_cat(list(cls_tokens, x), dim = 2)
    x <- x + self$pos_embed
    x <- self$pos_drop(x)

    for (i in seq_along(self$blocks)) {
      blk <- model$blocks[[i]]
      x <- blk(x)
    }

    x <- self$norm(x)
    return(self$head(x[ ,1, ]))
  }
)

patch_embed <- nn_module(
  classname = "patch_embed",
  initialize = function(img_size, patch_size, in_chans, embed_dim) {
    self$img_size <- img_size
    self$patch_size <- patch_size
    self$grid_size <- as.integer(img_size / patch_size)
    self$num_patches <- self$grid_size^2

    self$proj <- nn_conv2d(
      in_channels = in_chans,
      out_channels = embed_dim,
      kernel_size = patch_size,
      stride = patch_size
    )
  },

  forward = function(x) {
    x <- self$proj(x)  # shape: (B, embed_dim, H/ps, W/ps)
    x <- torch_flatten(x, start_dim = 3, end_dim = 4)  # flatten spatial dimensions (H/ps, W/ps)
    x <- x$transpose(2, 3)  # shape: (B, num_patches, embed_dim)
    x
  }
)

encoder_block <- nn_module(
  classname = "encoder_block",
  initialize = function(embed_dim = 768, num_heads = 12, mlp_ratio = 4, qkv_bias = TRUE, dropout = 0.0) {
    self$norm1 <- nn_layer_norm(embed_dim)
    self$attn <- nn_multihead_attention(embed_dim, num_heads, bias = qkv_bias, dropout = dropout)
    self$norm2 <- nn_layer_norm(embed_dim)
    self$mlp <- nn_sequential(
      nn_linear(embed_dim, as.integer(embed_dim * mlp_ratio)),
      nn_gelu(),
      nn_dropout(p = dropout),
      nn_linear(as.integer(embed_dim * mlp_ratio), embed_dim),
      nn_dropout(p = dropout)
    )
  },

  forward = function(x) {
    x <- x + self$attn(self$norm1(x), self$norm1(x), self$norm1(x))[[1]]
    x <- x + self$mlp(self$norm2(x))
    return(x)
  }
)