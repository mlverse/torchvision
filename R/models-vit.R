#' Vision Transformer Implementation
#'
#' Vision Transformer (ViT) models implement the architecture proposed in the paper 
#' [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
#' These models are designed for image classification tasks and operate by treating 
#' image patches as tokens in a Transformer model.
#'
#' ## Model Variants and Performance (ImageNet-1k)
#' ```
#' | Model     | Top-1 Acc | Top-5 Acc | Params  | GFLOPS | File Size | Weights Used              | Notes                  |
#' |-----------|-----------|-----------|---------|--------|-----------|---------------------------|------------------------|
#' | vit_b_16  | 81.1%     | 95.3%     | 86.6M   | 17.56  | 346 MB    | IMAGENET1K_V1             | Base, 16x16 patches    |
#' | vit_b_32  | 75.9%     | 92.5%     | 88.2M   | 4.41   | 353 MB    | IMAGENET1K_V1             | Base, 32x32 patches    |
#' | vit_l_16  | 79.7%     | 94.6%     | 304.3M  | 61.55  | 1.22 GB   | IMAGENET1K_V1             | Large, 16x16 patches   |
#' | vit_l_32  | 77.0%     | 93.1%     | 306.5M  | 15.38  | 1.23 GB   | IMAGENET1K_V1             | Large, 32x32 patches   |
#' | vit_h_14  | 88.6%     | 98.7%     | 633.5M  | 1016.7 | 2.53 GB   | IMAGENET1K_SWAG_E2E_V1    | Huge, 14x14 patches    |
#' ```
#' - **TorchVision Recipe**: <https://github.com/pytorch/vision/tree/main/references/classification>
#' - **SWAG Recipe**: <https://github.com/facebookresearch/SWAG>
#'
#' **Weights Selection**:  
#' - All models use the default `IMAGENET1K_V1` weights for consistency, stability, and official support from TorchVision.  
#' - These are supervised weights trained on ImageNet-1k.  
#' - For `vit_h_14`, the default weight is `IMAGENET1K_SWAG_E2E_V1`, pretrained on SWAG and fine-tuned on ImageNet.
#'
#' @inheritParams model_mobilenet_v2
#'
#' @family models
#' @rdname model_vit
#' @name model_vit
NULL
vit_torchscript_urls <- list(
  vit_b_16 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/vit_b_16.pth", "dc04e8807c138d7ed4ae2754f59aec00", "330 MB"),
  vit_b_32 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/vit_b_32.pth", "6ea9578eb4e1cf9ffe1eec998ffebe14", "330 MB"),
  vit_l_16 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/vit_l_16.pth", "bc28a00cbcb48fb25318c7d542100915", "1.2 GB"),
  vit_l_32 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/vit_l_32.pth", "fc44a3ff3f6905a6868f55070d6106af", "1.2 GB"),
  vit_h_14 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/vit_h_14.pth", "1f8d098982a80ccbe8dab3a5ff45752b", "2.4 GB")
)

load_vit_torchscript_model <- function(name, ...) {

  r <- vit_torchscript_urls[[name]]

  cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")

  archive <- download_and_cache(r[1], prefix = name)

  if (tools::md5sum(archive) != r[2])
    runtime_error("Corrupt file! Delete the file in {archive} and try again.")
  
  model <- model_vit_base(name = name, pretrained = FALSE, progress = FALSE, ...)
  state_dict <- torch::load_state_dict(archive)

  names(state_dict) <- gsub("^conv_proj\\.", "patch_embed.proj.", names(state_dict))
  names(state_dict) <- gsub("^encoder\\.layers\\.encoder_layer_(\\d+)\\.ln_1\\.", "blocks.\\1.norm1.", names(state_dict))
  names(state_dict) <- gsub("^encoder\\.layers\\.encoder_layer_(\\d+)\\.self_attention\\.", "blocks.\\1.attn.", names(state_dict))
  names(state_dict) <- gsub("^encoder\\.layers\\.encoder_layer_(\\d+)\\.ln_2\\.", "blocks.\\1.norm2.", names(state_dict))
  names(state_dict) <- gsub("^encoder\\.layers\\.encoder_layer_(\\d+)\\.mlp\\.linear_1\\.", "blocks.\\1.mlp.0.", names(state_dict))
  names(state_dict) <- gsub("^encoder\\.layers\\.encoder_layer_(\\d+)\\.mlp\\.linear_2\\.", "blocks.\\1.mlp.3.", names(state_dict))
  names(state_dict) <- gsub("^encoder\\.pos_embedding", "encoder.pos_embedding", names(state_dict))
  names(state_dict) <- gsub("^encoder\\.ln\\.", "norm.", names(state_dict))
  names(state_dict) <- gsub("^heads\\.head\\.", "head.", names(state_dict))

  model$load_state_dict(state_dict)
  model
}

model_vit_base <- function(name, pretrained, progress, ...) {

  if (pretrained) {
    return(load_vit_torchscript_model(name,...))
  } else {
    args <- list(...)
    config <- switch(name,
      vit_b_16 = list(img_size = 224, patch_size = 16, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, num_classes = 1000),
      vit_b_32 = list(img_size = 224, patch_size = 32, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, num_classes = 1000),
      vit_l_16 = list(img_size = 224, patch_size = 16, embed_dim = 1024, depth = 24, num_heads = 16, mlp_ratio = 4, num_classes = 1000),
      vit_l_32 = list(img_size = 224, patch_size = 32, embed_dim = 1024, depth = 24, num_heads = 16, mlp_ratio = 4, num_classes = 1000),
      vit_h_14 = list(img_size = 518, patch_size = 14, embed_dim = 1280, depth = 32, num_heads = 16, mlp_ratio = 4, num_classes = 1000),
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

vit_model <- torch::nn_module(
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
    self$encoder$pos_embedding <- nn_parameter(torch_zeros(1, num_patches + 1, embed_dim))
    self$pos_drop <- nn_dropout(p = dropout)

    self$blocks <- torch::nn_module_list(
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
    nn_init_trunc_normal_(self$encoder$pos_embedding, std = 0.02)
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
      else if (inherits(m, "nn_multihead_attention")) {
        nn_init_xavier_uniform_(m$in_proj_weight)
        if (!is.null(m$in_proj_bias))
          nn_init_zeros_(m$in_proj_bias)
      }
    }
  },

  forward = function(x) {
    B <- x$size(1)
    x <- self$patch_embed(x)

    cls_tokens <- self$class_token$expand(c(B, -1, -1))
    x <- torch_cat(list(cls_tokens, x), dim = 2)
    x <- x + self$encoder$pos_embedding
    x <- self$pos_drop(x)

    for (i in seq_along(self$blocks)) {
      blk <- self$blocks[[i]]
      x <- blk(x)
    }

    x <- self$norm(x)
    return(self$head(x[ ,1, ]))
  }
)

patch_embed <- torch::nn_module(
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
    x <- self$proj(x)
    x <- torch_flatten(x, start_dim = 3, end_dim = 4)
    x <- x$transpose(2, 3)
    x
  }
)

encoder_block <- torch::nn_module(
  classname = "encoder_block",
  initialize = function(
    embed_dim = 768,
    num_heads = 12,
    mlp_ratio = 4,
    qkv_bias = TRUE,
    dropout = 0.0
  ) {

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