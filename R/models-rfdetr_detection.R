#' RF-DETR Implementation
#'
#' RF-DETR: Neural Architecture Search for Real-Time Detection Transformers
#' ([https://arxiv.org/abs/2511.09554](https://arxiv.org/abs/2511.09554))
#'
#' Object detection transformer models combining a DINOv2 backbone
#' (windowed attention, register tokens) with a deformable-attention
#' decoder and two-stage query proposal. Supports Nano, Small, Medium,
#' Base, Base with O365 pretraining, and Large variants.
#'
#' ## Model Variants
#' ```
#' | Variant   | Backbone           | Decoder Layers | Resolution | # Queries | Group DETR | Weights                     |
#' |-----------|--------------------|----------------|------------|-----------|------------|-----------------------------|
#' | nano      | DINOv2 Small (win) | 2              | 384        | 300       | 13         | COCO (91 classes)           |
#' | small     | DINOv2 Small (win) | 2              | 512        | 300       | 13         | COCO (91 classes)           |
#' | medium    | DINOv2 Small (win) | 2              | 640        | 300       | 13         | COCO (91 classes)           |
#' | base      | DINOv2 Small (win) | 3              | 640        | 300       | 13         | COCO (91 classes)           |
#' | base_2    | DINOv2 Small (win) | 3              | 640        | 300       | 13         | COCO (91 classes, alt run)  |
#' | base_o365 | DINOv2 Small (win) | 3              | 640        | 300       | 13         | Objects365 (366 classes)    |
#' | large     | DINOv2 Base (win)  | 3              | 560        | 300       | 13         | COCO (91 classes)           |
#' ```
#' - All models use group DETR (group_detr=13) with two-stage query proposal,
#'   Lite Refpoint Refine, and BBox reparameterisation.
#' - The `large` variant corresponds to the deprecated RF-DETR-Large config
#'   (DINOv2 Base encoder, hidden_dim=384).
#'
#' @inheritParams model_mobilenet_v2
#'
#' @family object_detection_model
#' @rdname model_rfdetr
#' @name model_rfdetr
#'
#' @examples
#' \dontrun{
#' url <- "https://upload.wikimedia.org/wikipedia/commons/6/6f/Toy_Poodle_wearing_clothes_in_Tokyo.jpg"
#' img_tensor <- transform_to_tensor(magick::image_read(url))
#'
#' x <- nnf_interpolate(img_tensor$unsqueeze(1), size = c(640, 640))
#'
#' model_fn <- get("model_rfdetr_large")
#' model <- model_fn(pretrained = TRUE)
#' model$eval()
#'
#' results <- model(x)$detections[[1]]
#'
#' scores <- as.numeric(results$scores)
#' labels <- as.numeric(results$labels)
#' boxes <- as.matrix(results$boxes)
#'
#' keep <- which(scores > 0.75)
#'
#' if (length(keep) > 0) {
#'   boxed <- draw_bounding_boxes(
#'     (img_tensor * 255)$to(dtype = torch_uint8()),
#'     torch_tensor(boxes[keep, , drop = FALSE], dtype = torch_int64()),
#'     labels = sprintf("%d:%.2f", labels[keep], scores[keep]),
#'     colors = "black",
#'     width = 3,
#'     font_size = 12
#'   )
#'   tensor_image_browse(boxed)
#' }
#' }
NULL

# This code is modified from https://github.com/roboflow/rf-detr/
rfdetr_torchscript_urls <- list(
  rfdetr_nano = c("https://torch-cdn.mlverse.org/models/vision/v2/models/rf_detr_nano.pth", "f995ded00af2036196c9d4148da1532d", "116 MB"),
  rfdetr_small = c("https://torch-cdn.mlverse.org/models/vision/v2/models/rf_detr_small.pth", "b9ea5f60a04f07efb51097db3815e397", "116 MB"),
  rfdetr_medium = c("https://torch-cdn.mlverse.org/models/vision/v2/models/rf_detr_medium.pth", "6d836fae193ca043f269b6c38b415791", "116 MB"),
  rfdetr_base = c("https://torch-cdn.mlverse.org/models/vision/v2/models/rf_detr_base.pth", "7652e8b739b9e59479eaface8f48a285", "123 MB"),
  rfdetr_base_2 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/rf_detr_base_2.pth", "a281798e2562ebb4ec38140a767cde7d", "123 MB"),
  rfdetr_base_o365 = c("https://torch-cdn.mlverse.org/models/vision/v2/models/rf_detr_base_o365.pth", "2d409686b30615418bf0849bcac11863", "127 MB"),
  rfdetr_large = c("https://torch-cdn.mlverse.org/models/vision/v2/models/rf_detr_large.pth", "d77db0c7d6df39c93b0077a136ba07a3", "518 MB")
)

get_clones <- function(module, n) {
  nn_module_list(lapply(seq_len(n), function(i) module$clone(deep = TRUE)))
}

dinov2_patch_embeddings <- nn_module(
  "dinov2_patch_embeddings",
  initialize = function(hidden_size, patch_size, num_channels = 3) {
    self$projection <- nn_conv2d(num_channels, hidden_size, kernel_size = patch_size, stride = patch_size)
  },
  forward = function(pixel_values) {
    embeddings <- self$projection(pixel_values)
    embeddings <- embeddings$flatten(start_dim = 3, end_dim = 4)
    embeddings <- embeddings$transpose(2, 3)
    embeddings
  }
)

windowed_dinov2_embeddings <- nn_module(
  "windowed_dinov2_embeddings",
  initialize = function(config) {
    self$cls_token <- nn_parameter(torch_randn(1, 1, config$hidden_size))
    self$mask_token <- nn_parameter(torch_zeros(1, config$hidden_size))
    if (!is.null(config$num_register_tokens) && config$num_register_tokens > 0) {
      self$register_tokens <- nn_parameter(torch_zeros(1, config$num_register_tokens, config$hidden_size))
    }
    self$patch_embeddings <- dinov2_patch_embeddings(
      config$hidden_size, config$patch_size, config$num_channels %||% 3
    )
    num_patches <- (config$image_size %/% config$patch_size)^2
    self$position_embeddings <- nn_parameter(torch_randn(1, num_patches + 1, config$hidden_size))
    self$dropout <- nn_dropout(p = config$hidden_dropout_prob %||% 0.0)
    self$patch_size <- config$patch_size
    self$num_windows <- config$num_windows %||% 1
    self$num_register_tokens <- config$num_register_tokens %||% 0
    self$config <- config
  },
  interpolate_pos_encoding = function(embeddings, height, width) {
    num_patches <- embeddings$size(2) - 1
    num_positions <- self$position_embeddings$size(2) - 1
    if (num_patches == num_positions && height == width) {
      return(self$position_embeddings)
    }
    class_pos_embed <- self$position_embeddings[, 1, ]
    patch_pos_embed <- self$position_embeddings[, 2:(num_positions + 1), , drop = FALSE]
    dim <- embeddings$size(3)
    h <- height %/% self$config$patch_size
    w <- width %/% self$config$patch_size
    sqrt_n <- as.integer(sqrt(num_positions))
    patch_pos_embed <- patch_pos_embed$reshape(c(1, sqrt_n, sqrt_n, dim))
    patch_pos_embed <- patch_pos_embed$permute(c(1, 4, 2, 3))
    target_dtype <- patch_pos_embed$dtype
    patch_pos_embed <- nnf_interpolate(
      patch_pos_embed$to(dtype = torch_float32()),
      size = c(h, w),
      mode = "bicubic",
      align_corners = FALSE
    )$to(dtype = target_dtype)
    patch_pos_embed <- patch_pos_embed$permute(c(1, 3, 4, 2))$reshape(c(1, -1, dim))
    torch_cat(list(class_pos_embed$unsqueeze(1), patch_pos_embed), dim = 2)
  },
  forward = function(pixel_values) {
    batch_size <- pixel_values$size(1)
    height <- pixel_values$size(3)
    width <- pixel_values$size(4)
    target_dtype <- self$patch_embeddings$projection$weight$dtype
    embeddings <- self$patch_embeddings(pixel_values$to(dtype = target_dtype))
    cls_tokens <- self$cls_token$expand(c(batch_size, -1, -1))
    embeddings <- torch_cat(list(cls_tokens, embeddings), dim = 2)
    embeddings <- embeddings + self$interpolate_pos_encoding(embeddings, height, width)
    if (self$num_windows > 1) {
      num_h_patches <- height %/% self$patch_size
      num_w_patches <- width %/% self$patch_size
      cls_token_with_pos <- embeddings[, 1, , drop = FALSE]
      pixel_tokens <- embeddings[, 2:embeddings$size(2), , drop = FALSE]
      pixel_tokens <- pixel_tokens$view(c(batch_size, num_h_patches, num_w_patches, -1))
      num_h_ppw <- num_h_patches %/% self$num_windows
      num_w_ppw <- num_w_patches %/% self$num_windows
      nw <- self$num_windows
      windowed_pixel <- pixel_tokens$reshape(c(
        batch_size * nw, num_h_ppw, nw, num_w_ppw, -1
      ))
      windowed_pixel <- windowed_pixel$permute(c(1, 3, 2, 4, 5))
      windowed_pixel <- windowed_pixel$reshape(c(
        batch_size * nw^2, num_h_ppw * num_w_ppw, -1
      ))
      windowed_cls <- cls_token_with_pos$'repeat'(c(nw^2, 1, 1))
      embeddings <- torch_cat(list(windowed_cls, windowed_pixel), dim = 2)
    }
    if (self$num_register_tokens > 0) {
      reg_tokens <- self$register_tokens$expand(c(embeddings$size(1), -1, -1))
      embeddings <- torch_cat(list(
        embeddings[, 1, , drop = FALSE], reg_tokens, embeddings[, 2:embeddings$size(2), , drop = FALSE]
      ), dim = 2)
    }
    embeddings <- self$dropout(embeddings)
    embeddings
  }
)

dinov2_self_attention <- nn_module(
  "dinov2_self_attention",
  initialize = function(config) {
    self$num_attention_heads <- config$num_attention_heads
    self$attention_head_size <- config$hidden_size %/% config$num_attention_heads
    self$all_head_size <- self$num_attention_heads * self$attention_head_size
    self$query <- nn_linear(config$hidden_size, self$all_head_size, bias = config$qkv_bias %||% TRUE)
    self$key <- nn_linear(config$hidden_size, self$all_head_size, bias = config$qkv_bias %||% TRUE)
    self$value <- nn_linear(config$hidden_size, self$all_head_size, bias = config$qkv_bias %||% TRUE)
    self$dropout <- nn_dropout(p = config$attention_probs_dropout_prob %||% 0.0)
  },
  transpose_for_scores = function(x) {
    new_shape <- c(x$size()[-length(x$size())], self$num_attention_heads, self$attention_head_size)
    x <- x$reshape(new_shape)
    x$permute(c(1, 3, 2, 4))
  },
  forward = function(hidden_states) {
    mixed_query_layer <- self$query(hidden_states)
    key_layer <- self$transpose_for_scores(self$key(hidden_states))
    value_layer <- self$transpose_for_scores(self$value(hidden_states))
    query_layer <- self$transpose_for_scores(mixed_query_layer)
    attention_scores <- query_layer$matmul(key_layer$transpose(-2, -1))
    attention_scores <- attention_scores / sqrt(self$attention_head_size)
    attention_probs <- nnf_softmax(attention_scores, dim = -1)
    attention_probs <- self$dropout(attention_probs)
    context_layer <- attention_probs$matmul(value_layer)
    context_layer <- context_layer$permute(c(1, 3, 2, 4))
    context_layer <- context_layer$reshape(c(
      context_layer$size()[1:2], self$all_head_size
    ))
    list(context_layer, attention_probs)
  }
)

dinov2_self_output <- nn_module(
  "dinov2_self_output",
  initialize = function(config) {
    self$dense <- nn_linear(config$hidden_size, config$hidden_size)
    self$dropout <- nn_dropout(p = config$hidden_dropout_prob %||% 0.0)
  },
  forward = function(hidden_states) {
    hidden_states <- self$dense(hidden_states)
    hidden_states <- self$dropout(hidden_states)
    hidden_states
  }
)

dinov2_attention <- nn_module(
  "dinov2_attention",
  initialize = function(config) {
    self$attention <- dinov2_self_attention(config)
    self$output <- dinov2_self_output(config)
  },
  forward = function(hidden_states) {
    self_outputs <- self$attention(hidden_states)
    attention_output <- self$output(self_outputs[[1]])
    list(attention_output, self_outputs[[2]])
  }
)

dinov2_layerscale <- nn_module(
  "dinov2_layerscale",
  initialize = function(config) {
    self$lambda1 <- nn_parameter(config$layerscale_value %||% 1.0 * torch_ones(config$hidden_size))
  },
  forward = function(hidden_state) {
    hidden_state * self$lambda1
  }
)

dinov2_mlp <- nn_module(
  "dinov2_mlp",
  initialize = function(config) {
    in_features <- out_features <- config$hidden_size
    hidden_features <- as.integer(config$hidden_size * config$mlp_ratio %||% 4)
    self$fc1 <- nn_linear(in_features, hidden_features, bias = TRUE)
    self$activation <- nn_gelu()
    self$fc2 <- nn_linear(hidden_features, out_features, bias = TRUE)
  },
  forward = function(hidden_state) {
    hidden_state <- self$fc1(hidden_state)
    hidden_state <- self$activation(hidden_state)
    hidden_state <- self$fc2(hidden_state)
    hidden_state
  }
)

windowed_dinov2_layer <- nn_module(
  "windowed_dinov2_layer",
  initialize = function(config) {
    self$num_windows <- config$num_windows %||% 1
    self$norm1 <- nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps %||% 1e-6)
    self$attention <- dinov2_attention(config)
    self$layer_scale1 <- dinov2_layerscale(config)
    self$norm2 <- nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps %||% 1e-6)
    self$mlp <- if (config$use_swiglu_ffn %||% FALSE) {
      dinov2_swiglu_mlp(config)
    } else {
      dinov2_mlp(config)
    }
    self$layer_scale2 <- dinov2_layerscale(config)
  },
  forward = function(hidden_states, run_full_attention = FALSE) {
    shortcut <- hidden_states
    if (run_full_attention) {
      bw <- hidden_states$size(1)
      tpw <- hidden_states$size(2)
      c <- hidden_states$size(3)
      nws <- self$num_windows^2
      hidden_states <- hidden_states$reshape(c(bw %/% nws, nws * tpw, c))
    }
    norm1_out <- self$norm1(hidden_states)
    attn_outputs <- self$attention(norm1_out)
    attention_output <- attn_outputs[[1]]
    if (run_full_attention) {
      bw <- hidden_states$size(1)
      tpw <- hidden_states$size(2)
      c <- hidden_states$size(3)
      nws <- self$num_windows^2
      attention_output <- attention_output$reshape(c(bw * nws, tpw %/% nws, c))
    }
    attention_output <- self$layer_scale1(attention_output)
    hidden_states <- attention_output + shortcut
    layer_output <- self$norm2(hidden_states)
    layer_output <- self$mlp(layer_output)
    layer_output <- self$layer_scale2(layer_output)
    layer_output <- layer_output + hidden_states
    list(layer_output)
  }
)

dinov2_swiglu_mlp <- nn_module(
  "dinov2_swiglu_mlp",
  initialize = function(config) {
    in_features <- out_features <- config$hidden_size
    hidden_features <- as.integer(config$hidden_size * config$mlp_ratio %||% 4)
    hidden_features <- as.integer((hidden_features * 2 / 3 + 7) %/% 8 * 8)
    self$weights_in <- nn_linear(in_features, 2 * hidden_features, bias = TRUE)
    self$weights_out <- nn_linear(hidden_features, out_features, bias = TRUE)
  },
  forward = function(hidden_state) {
    hidden_state <- self$weights_in(hidden_state)
    x1 <- hidden_state$chunk(2, dim = -1)[[1]]
    x2 <- hidden_state$chunk(2, dim = -1)[[2]]
    hidden <- nnf_silu(x1) * x2
    self$weights_out(hidden)
  }
)

windowed_dinov2_encoder <- nn_module(
  "windowed_dinov2_encoder",
  initialize = function(config) {
    self$config <- config
    self$layer <- nn_module_list(lapply(seq_len(config$num_hidden_layers), function(i) {
      windowed_dinov2_layer(config)
    }))
  },
  forward = function(hidden_states, output_hidden_states = FALSE) {
    all_hidden_states <- list()
    out_feat_idx <- as.integer(gsub("stage", "", self$config$out_features))
    for (i in seq_along(self$layer)) {
      if (output_hidden_states) {
        all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
      }
      if (i > max(out_feat_idx)) break
      run_full_attention <- !((i - 1) %in% (self$config$window_block_indexes %||% list()))
      layer_outputs <- self$layer[[i]](hidden_states, run_full_attention)
      hidden_states <- layer_outputs[[1]]
    }
    if (output_hidden_states) {
      all_hidden_states[[length(all_hidden_states) + 1]] <- hidden_states
    }
    list(hidden_states, all_hidden_states)
  }
)

windowed_dinov2_backbone_hf <- nn_module(
  "windowed_dinov2_backbone_hf",
  initialize = function(config) {
    self$embeddings <- windowed_dinov2_embeddings(config)
    self$encoder <- windowed_dinov2_encoder(config)
    self$layernorm <- nn_layer_norm(config$hidden_size, eps = config$layer_norm_eps %||% 1e-6)
    self$config <- config
    self$num_register_tokens <- config$num_register_tokens %||% 0
    self$stage_names <- c("stem", paste0("stage", 1:config$num_hidden_layers))
    self$out_features <- config$out_features
    self$apply_layernorm <- config$apply_layernorm %||% TRUE
    self$reshape_hidden_states <- config$reshape_hidden_states %||% TRUE
  },
  forward = function(pixel_values) {
    embedding_output <- self$embeddings(pixel_values)
    encoder_outputs <- self$encoder(embedding_output, output_hidden_states = TRUE)
    hidden_states <- encoder_outputs[[2]]
    feature_maps <- list()
    for (i in seq_along(self$stage_names)) {
      stage <- self$stage_names[i]
      if (stage %in% self$out_features && i <= length(hidden_states)) {
        hs <- hidden_states[[i]]
        if (self$apply_layernorm) {
          hs <- self$layernorm(hs)
        }
        if (self$reshape_hidden_states) {
          if (self$num_register_tokens > 0) {
            hs <- hs[, (2 + self$num_register_tokens):hs$size(2), , drop = FALSE]
          } else {
            hs <- hs[, 2:hs$size(2), , drop = FALSE]
          }
          batch_size <- pixel_values$size(1)
          height <- pixel_values$size(3)
          width <- pixel_values$size(4)
          patch_size <- self$config$patch_size
          num_h_patches <- height %/% patch_size
          num_w_patches <- width %/% patch_size
          num_windows <- self$config$num_windows %||% 1
          if (num_windows > 1) {
            nws <- num_windows^2
            bw <- hs$size(1)
            tpw <- hs$size(2)
            c <- hs$size(3)
            nh_ppw <- num_h_patches %/% num_windows
            nw_ppw <- num_w_patches %/% num_windows
            hs <- hs$reshape(c(bw %/% nws, nws * tpw, c))
            hs <- hs$reshape(c(
              (bw %/% nws) * num_windows, num_windows, nh_ppw, nw_ppw, c
            ))
            hs <- hs$permute(c(1, 3, 2, 4, 5))
          }
          hs <- hs$reshape(c(batch_size, num_h_patches, num_w_patches, -1))
          hs <- hs$permute(c(1, 4, 2, 3))$contiguous()
        }
        feature_maps[[length(feature_maps) + 1]] <- hs
      }
    }
    list(feature_maps)
  }
)

projector_layernorm <- nn_module(
  "projector_layernorm",
  initialize = function(normalized_shape, eps = 1e-6) {
    self$weight <- nn_parameter(torch_ones(normalized_shape))
    self$bias <- nn_parameter(torch_zeros(normalized_shape))
    self$eps <- eps
    self$normalized_shape <- c(normalized_shape)
  },
  forward = function(x) {
    x <- x$permute(c(1, 3, 4, 2))
    x <- nnf_layer_norm(x, self$normalized_shape, self$weight, self$bias, self$eps)
    x <- x$permute(c(1, 4, 2, 3))
    x
  }
)

convx <- nn_module(
  "convx",
  initialize = function(in_planes, out_planes, kernel = 3, stride = 1, groups = 1,
                        act = "silu", layer_norm = FALSE) {
    if (length(kernel) == 1) kernel <- c(kernel, kernel)
    padding <- c(kernel[1] %/% 2, kernel[2] %/% 2)
    self$conv <- nn_conv2d(in_planes, out_planes, kernel_size = kernel, stride = stride,
                           padding = padding, groups = groups, bias = FALSE)
    if (layer_norm) {
      self$bn <- projector_layernorm(out_planes)
    } else {
      self$bn <- nn_batch_norm2d(out_planes)
    }
    self$act <- if (act == "silu") nn_silu(inplace = TRUE) else nn_relu(inplace = TRUE)
  },
  forward = function(x) {
    self$act(self$bn(self$conv(x$contiguous())))
  }
)

rfdetr_bottleneck <- nn_module(
  "rfdetr_bottleneck",
  initialize = function(c1, c2, shortcut = TRUE, g = 1, k = c(3, 3), e = 0.5, act = "silu", layer_norm = FALSE) {
    c_ <- as.integer(c2 * e)
    self$cv1 <- convx(c1, c_, kernel = k[1], stride = 1, act = act, layer_norm = layer_norm)
    self$cv2 <- convx(c_, c2, kernel = k[2], stride = 1, groups = g, act = act, layer_norm = layer_norm)
    self$add <- shortcut && c1 == c2
  },
  forward = function(x) {
    out <- self$cv2(self$cv1(x))
    if (self$add) x + out else out
  }
)

c2f <- nn_module(
  "c2f",
  initialize = function(c1, c2, n = 1, shortcut = FALSE, g = 1, e = 0.5, act = "silu", layer_norm = FALSE) {
    c <- as.integer(c2 * e)
    self$c <- c
    self$cv1 <- convx(c1, 2 * c, kernel = 1, stride = 1, act = act, layer_norm = layer_norm)
    self$cv2 <- convx((2 + n) * c, c2, kernel = 1, act = act, layer_norm = layer_norm)
    self$m <- nn_module_list(lapply(seq_len(n), function(i) {
      rfdetr_bottleneck(c, c, shortcut, g, k = c(3, 3), e = 1.0, act = act, layer_norm = layer_norm)
    }))
  },
  forward = function(x) {
    y <- self$cv1(x)$split(self$c, dim = 2)
    y <- as.list(y)
    for (i in seq_along(self$m)) {
      y[[length(y) + 1]] <- self$m[[i]](y[[length(y)]])
    }
    self$cv2(torch_cat(y, dim = 2))
  }
)

multiscale_projector <- nn_module(
  "multiscale_projector",
  initialize = function(in_channels, out_channels, scale_factors, num_blocks = 3,
                        layer_norm = FALSE) {
    self$scale_factors <- scale_factors
    stages_sampling <- list()
    for (i in seq_along(scale_factors)) {
      scale <- scale_factors[i]
      level_modules <- list()
      for (j in seq_along(in_channels)) {
        in_dim <- in_channels[j]
        if (scale == 1.0) {
          level_modules[[j]] <- nn_sequential(nn_identity())
        } else if (scale == 2.0) {
          level_modules[[j]] <- nn_sequential(
            nn_conv_transpose2d(in_dim, in_dim %/% 2, kernel_size = 2, stride = 2)
          )
        } else if (scale == 0.5) {
          level_modules[[j]] <- nn_sequential(
            convx(in_dim, in_dim, kernel = 3, stride = 2, act = "silu", layer_norm = layer_norm)
          )
        } else {
          level_modules[[j]] <- nn_sequential(nn_identity())
        }
      }
      stages_sampling[[i]] <- nn_module_list(level_modules)
    }
    self$stages_sampling <- nn_module_list(stages_sampling)
    self$stages <- nn_module_list()
    for (scale in scale_factors) {
      in_dim <- as.integer(sum(in_channels / max(1, scale)))
      layers <- nn_sequential(
        c2f(in_dim, out_channels, num_blocks, layer_norm = layer_norm),
        projector_layernorm(out_channels)
      )
      self$stages$append(layers)
    }
  },
  forward = function(x) {
    results <- list()
    for (i in seq_along(self$stages)) {
      feat_fuse <- list()
      for (j in seq_along(x)) {
        feat_fuse[[length(feat_fuse) + 1]] <- self$stages_sampling[[i]][[j]](x[[j]])
      }
      if (length(feat_fuse) > 1) {
        feat_fuse <- torch_cat(feat_fuse, dim = 2)
      } else {
        feat_fuse <- feat_fuse[[1]]
      }
      results[[length(results) + 1]] <- self$stages[[i]](feat_fuse)
    }
    results
  }
)

position_embedding_sine <- nn_module(
  "position_embedding_sine",
  initialize = function(num_pos_feats = 128, temperature = 10000, normalize = TRUE, scale = NULL) {
    self$num_pos_feats <- num_pos_feats
    self$temperature <- temperature
    self$normalize <- normalize
    if (is.null(scale)) scale <- 2 * pi
    self$scale <- scale
  },
  forward = function(x, align_dim_orders = FALSE) {
    if (is.list(x) && !is.null(x$mask)) {
      mask <- x$mask
    } else {
      mask <- NULL
    }
    if (!is.null(mask)) {
      not_mask <- !mask
    } else {
      not_mask <- torch_ones(c(x$size(1), x$size(3), x$size(4)), device = x$device, dtype = torch_bool())
    }
    y_embed <- torch_cumsum(not_mask$to(dtype = torch_float32()), dim = 2)
    x_embed <- torch_cumsum(not_mask$to(dtype = torch_float32()), dim = 3)
    if (self$normalize) {
      eps <- 1e-6
      y_embed <- y_embed / (y_embed[, -1, , drop = FALSE] + eps) * self$scale
      x_embed <- x_embed / (x_embed[, , -1, drop = FALSE] + eps) * self$scale
    }
    dim_t <- torch_arange(0, self$num_pos_feats - 1, dtype = torch_float32(), device = x$device)
    dim_t <- self$temperature^(2 * (dim_t %/% 2) / self$num_pos_feats)
    pos_x <- x_embed$unsqueeze(4) / dim_t
    pos_y <- y_embed$unsqueeze(4) / dim_t
    pos_x <- torch_stack(list(pos_x[, , , seq(1, self$num_pos_feats, 2)]$sin(),
                               pos_x[, , , seq(2, self$num_pos_feats, 2)]$cos()), dim = 5)$flatten(start_dim = 4)
    pos_y <- torch_stack(list(pos_y[, , , seq(1, self$num_pos_feats, 2)]$sin(),
                               pos_y[, , , seq(2, self$num_pos_feats, 2)]$cos()), dim = 5)$flatten(start_dim = 4)
    if (align_dim_orders) {
      pos <- torch_cat(list(pos_y, pos_x), dim = 4)$permute(c(3, 4, 1, 2))
    } else {
      pos <- torch_cat(list(pos_y, pos_x), dim = 4)$permute(c(1, 4, 2, 3))
    }
    pos
  }
)

rfdetr_dinov2 <- nn_module(
  "rfdetr_dinov2",
  initialize = function(size = "small", out_feature_indexes = c(3, 6, 9, 12),
                        image_size = 518, patch_size = 14, num_windows = 2,
                        hidden_size = 384, num_attention_heads = 6,
                        num_hidden_layers = 12, num_register_tokens = 4,
                        drop_path_rate = 0.0) {
    config <- list(
      hidden_size = hidden_size,
      num_hidden_layers = num_hidden_layers,
      num_attention_heads = num_attention_heads,
      mlp_ratio = 4,
      hidden_act = "gelu",
      hidden_dropout_prob = 0.0,
      attention_probs_dropout_prob = 0.0,
      layer_norm_eps = 1e-6,
      image_size = image_size,
      patch_size = patch_size,
      num_channels = 3,
      qkv_bias = TRUE,
      layerscale_value = 1.0,
      drop_path_rate = drop_path_rate,
      use_swiglu_ffn = FALSE,
      num_register_tokens = num_register_tokens,
      num_windows = num_windows,
      window_block_indexes = setdiff(0:max(out_feature_indexes), out_feature_indexes),
      out_features = paste0("stage", out_feature_indexes),
      apply_layernorm = TRUE,
      reshape_hidden_states = TRUE
    )
    self$config <- config
    self$encoder <- windowed_dinov2_backbone_hf(config)
    self$out_feature_channels <- rep(hidden_size, length(out_feature_indexes))
  },
  forward = function(x) {
    out <- self$encoder(x)
    out[[1]]
  }
)

rfdetr_backbone <- nn_module(
  "rfdetr_backbone",
  initialize = function(encoder_name = "dinov2_windowed_small",
                        out_feature_indexes = c(3, 6, 9, 12),
                        hidden_dim = 256, layer_norm = TRUE,
                        target_shape = c(640, 640), patch_size = 16,
                        num_windows = 2, num_register_tokens = 0,
                        drop_path = 0.0, projector_scale = c(1.0)) {
    size_configs <- list(
      tiny = list(hidden_size = 192, num_heads = 3, num_layers = 12),
      small = list(hidden_size = 384, num_heads = 6, num_layers = 12),
      base = list(hidden_size = 768, num_heads = 12, num_layers = 12),
      large = list(hidden_size = 1024, num_heads = 16, num_layers = 24)
    )
    name_parts <- strsplit(encoder_name, "_")[[1]]
    size <- name_parts[length(name_parts)]
    sc <- size_configs[[size]]
    self$encoder <- rfdetr_dinov2(
      size = size,
      out_feature_indexes = out_feature_indexes,
      image_size = target_shape[1] %/% patch_size * patch_size,
      patch_size = patch_size,
      num_windows = num_windows,
      hidden_size = sc$hidden_size,
      num_attention_heads = sc$num_heads,
      num_hidden_layers = sc$num_layers,
      num_register_tokens = num_register_tokens,
      drop_path_rate = drop_path
    )
    self$patch_size <- patch_size
    self$num_windows <- num_windows
    self$projector <- multiscale_projector(
      in_channels = rep(sc$hidden_size, length(out_feature_indexes)),
      out_channels = hidden_dim,
      scale_factors = projector_scale,
      num_blocks = 3,
      layer_norm = layer_norm
    )
    self$out_channels <- hidden_dim
    self$hidden_dim <- hidden_dim
  },
  forward = function(x, mask = NULL) {
    divisor <- self$patch_size * self$num_windows
    h <- x$size(3) %/% divisor * divisor
    w <- x$size(4) %/% divisor * divisor
    if (h != x$size(3) || w != x$size(4)) {
      x <- nnf_interpolate(x, size = c(h, w), mode = "bilinear", align_corners = FALSE)
    }
    feats <- self$encoder(x)
    feats <- self$projector(feats)
    if (!is.null(mask)) {
      out <- list()
      for (feat in feats) {
        m <- nnf_interpolate(mask$unsqueeze(1)$float(), size = feat$shape[3:4])$to(dtype = torch_bool())[1, , ]
        out[[length(out) + 1]] <- list(tensors = feat, mask = m)
      }
      out
    } else {
      feats
    }
  }
)

rfdetr_joiner <- nn_module(
  "rfdetr_joiner",
  initialize = function(backbone, position_embedding) {
    self[["0"]] <- backbone
    self[["1"]] <- position_embedding
  },
  forward = function(x, mask = NULL) {
    if (is.list(x) && !is.null(x$tensors)) {
      mask <- x$mask
      x <- x$tensors
    }
    feats <- self[["0"]](x, mask)
    if (length(feats) > 0 && is.list(feats[[1]]) && !is.null(feats[[1]]$tensors)) {
      pos <- list()
      for (i in seq_along(feats)) {
        pos[[i]] <- self[["1"]](feats[[i]]$tensors, align_dim_orders = FALSE)
      }
      list(feats, pos)
    } else {
      pos <- list()
      for (feat in feats) {
        pos[[length(pos) + 1]] <- self[["1"]](feat, align_dim_orders = FALSE)
      }
      list(feats, pos)
    }
  }
)


gen_sineembed_for_position <- function(pos_tensor, dim = 128) {
  scale <- 2 * pi
  dim_t <- torch_arange(0, dim - 1, dtype = pos_tensor$dtype, device = pos_tensor$device)
  dim_t <- 10000^(2 * (dim_t %/% 2) / dim)
  x_embed <- pos_tensor[, , 1] * scale
  y_embed <- pos_tensor[, , 2] * scale
  pos_x <- x_embed$unsqueeze(3) / dim_t
  pos_y <- y_embed$unsqueeze(3) / dim_t
  pos_x <- torch_stack(list(pos_x[, , seq(1, dim, 2)]$sin(), pos_x[, , seq(2, dim, 2)]$cos()), dim = 4)$flatten(start_dim = 3)
  pos_y <- torch_stack(list(pos_y[, , seq(1, dim, 2)]$sin(), pos_y[, , seq(2, dim, 2)]$cos()), dim = 4)$flatten(start_dim = 3)
  if (pos_tensor$size(-1) == 4) {
    w_embed <- pos_tensor[, , 3] * scale
    h_embed <- pos_tensor[, , 4] * scale
    pos_w <- w_embed$unsqueeze(3) / dim_t
    pos_h <- h_embed$unsqueeze(3) / dim_t
    pos_w <- torch_stack(list(pos_w[, , seq(1, dim, 2)]$sin(), pos_w[, , seq(2, dim, 2)]$cos()), dim = 4)$flatten(start_dim = 3)
    pos_h <- torch_stack(list(pos_h[, , seq(1, dim, 2)]$sin(), pos_h[, , seq(2, dim, 2)]$cos()), dim = 4)$flatten(start_dim = 3)
    pos <- torch_cat(list(pos_y, pos_x, pos_w, pos_h), dim = 3)
  } else {
    pos <- torch_cat(list(pos_y, pos_x), dim = 3)
  }
  pos
}

gen_encoder_output_proposals <- function(memory, memory_padding_mask, spatial_shapes, unsigmoid = TRUE) {
  proposals <- list()
  cur <- 1
  for (lvl in seq_len(nrow(spatial_shapes))) {
    h <- as.integer(spatial_shapes[lvl, 1])
    w <- as.integer(spatial_shapes[lvl, 2])
    if (!is.null(memory_padding_mask)) {
      mask_flatten <- memory_padding_mask[, cur:(cur + h * w - 1)]
      mask_flatten <- mask_flatten$view(c(-1, h, w, 1))
      valid_height <- torch_sum((!mask_flatten[, , 1, 1])$to(dtype = torch_int()), dim = 2)
      valid_width <- torch_sum((!mask_flatten[, 1, , 1])$to(dtype = torch_int()), dim = 2)
    }
    grid_y <- torch_linspace(0, h - 1, h, dtype = torch_float32(), device = memory$device)
    grid_x <- torch_linspace(0, w - 1, w, dtype = torch_float32(), device = memory$device)
    grid <- torch_stack(torch_meshgrid(list(grid_x, grid_y), indexing = "ij"), dim = -1)
    scale <- torch_tensor(c(w, h), dtype = torch_float32(), device = grid$device)$unsqueeze(1)$unsqueeze(1)
    grid <- (grid$unsqueeze(1) + 0.5) / scale
    wh <- torch_ones_like(grid) * 0.05 * (2^(lvl - 1))
    proposal <- torch_cat(list(grid, wh), dim = -1)$reshape(c(-1, h * w, 4))
    proposals[[lvl]] <- proposal
    cur <- cur + h * w
  }
  output_proposals <- torch_cat(proposals, dim = 2)
  output_proposals_valid <- (output_proposals > 0.01 & output_proposals < 0.99)$all(dim = -1, keepdim = TRUE)
  if (unsigmoid) {
    output_proposals <- torch_log(output_proposals / (1 - output_proposals))
    if (!is.null(memory_padding_mask)) {
      output_proposals <- output_proposals$masked_fill(memory_padding_mask$unsqueeze(3), Inf)
    }
    output_proposals <- output_proposals$masked_fill(!output_proposals_valid, Inf)
  } else {
    if (!is.null(memory_padding_mask)) {
      output_proposals <- output_proposals$masked_fill(memory_padding_mask$unsqueeze(3), 0)
    }
    output_proposals <- output_proposals$masked_fill(!output_proposals_valid, 0)
  }
  output_memory <- memory
  if (!is.null(memory_padding_mask)) {
    output_memory <- output_memory$masked_fill(memory_padding_mask$unsqueeze(3), 0)
  }
  output_memory <- output_memory$masked_fill(!output_proposals_valid, 0)
  list(output_memory, output_proposals)
}

rfdetr_decoder_layer <- nn_module(
  "rfdetr_decoder_layer",
  initialize = function(d_model = 256, sa_nhead = 8, ca_nhead = 16,
                        dim_feedforward = 2048, dropout = 0.0,
                        group_detr = 1, num_feature_levels = 1,
                        dec_n_points = 2) {
    self$self_attn <- nn_multihead_attention(d_model, sa_nhead, dropout = dropout, batch_first = TRUE)
    self$dropout1 <- nn_dropout(dropout)
    self$norm1 <- nn_layer_norm(d_model)
    self$cross_attn <- detr_ms_deform_attn(d_model, n_levels = num_feature_levels, n_heads = ca_nhead, n_points = dec_n_points)
    self$linear1 <- nn_linear(d_model, dim_feedforward)
    self$dropout <- nn_dropout(dropout)
    self$linear2 <- nn_linear(dim_feedforward, d_model)
    self$norm2 <- nn_layer_norm(d_model)
    self$norm3 <- nn_layer_norm(d_model)
    self$dropout2 <- nn_dropout(dropout)
    self$dropout3 <- nn_dropout(dropout)
    self$activation <- nn_relu()
    self$group_detr <- group_detr
  },
  with_pos_embed = function(tensor, pos) {
    if (is.null(pos)) tensor else tensor + pos
  },
  forward = function(tgt, memory, tgt_mask = NULL, memory_key_padding_mask = NULL,
                     pos = NULL, query_pos = NULL, query_sine_embed = NULL,
                     is_first = FALSE, reference_points = NULL,
                     spatial_shapes = NULL, level_start_index = NULL) {
    bs <- tgt$size(1)
    num_queries <- tgt$size(2)
    q <- k <- tgt + query_pos
    v <- tgt
    if (self$training) {
      q <- torch_cat(q$split(num_queries %/% self$group_detr, dim = 2), dim = 1)
      k <- torch_cat(k$split(num_queries %/% self$group_detr, dim = 2), dim = 1)
      v <- torch_cat(v$split(num_queries %/% self$group_detr, dim = 2), dim = 1)
    }
    tgt2 <- self$self_attn(q, k, v, attn_mask = tgt_mask, need_weights = FALSE)[[1]]
    if (self$training) {
      tgt2 <- torch_cat(tgt2$split(bs, dim = 1), dim = 2)
    }
    tgt <- tgt + self$dropout1(tgt2)
    tgt <- self$norm1(tgt)
    tgt2 <- self$cross_attn(
      self$with_pos_embed(tgt, query_pos),
      reference_points,
      memory,
      spatial_shapes,
      level_start_index,
      mask = if (!is.null(memory_key_padding_mask)) memory_key_padding_mask$unsqueeze(3) else NULL
    )
    tgt <- tgt + self$dropout2(tgt2)
    tgt <- self$norm2(tgt)
    tgt2 <- self$linear2(self$dropout(self$activation(self$linear1(tgt))))
    tgt <- tgt + self$dropout3(tgt2)
    tgt <- self$norm3(tgt)
    tgt
  }
)

rfdetr_decoder <- nn_module(
  "rfdetr_decoder",
  initialize = function(decoder_layer, num_layers, norm = NULL,
                        return_intermediate = FALSE, d_model = 256,
                        lite_refpoint_refine = FALSE, bbox_reparam = FALSE) {
    self$layers <- get_clones(decoder_layer, num_layers)
    self$num_layers <- num_layers
    self$norm <- norm
    self$return_intermediate <- return_intermediate
    self$lite_refpoint_refine <- lite_refpoint_refine
    self$bbox_reparam <- bbox_reparam
    self$ref_point_head <- detr_mlp_layer(2 * d_model, d_model, d_model, 2)
  },
  refpoints_refine = function(refpoints_unsigmoid, new_refpoints_delta) {
    if (self$bbox_reparam) {
      new_cxcy <- new_refpoints_delta[, , 1:2] * refpoints_unsigmoid[, , 3:4] + refpoints_unsigmoid[, , 1:2]
      new_wh <- new_refpoints_delta[, , 3:4]$exp() * refpoints_unsigmoid[, , 3:4]
      torch_cat(list(new_cxcy, new_wh), dim = -1)
    } else {
      refpoints_unsigmoid + new_refpoints_delta
    }
  },
  get_reference = function(refpoints, valid_ratios, d_model_half) {
    obj_center <- refpoints[, , 1:4, drop = FALSE]
    refpoints_input <- obj_center$unsqueeze(3) * torch_cat(list(valid_ratios, valid_ratios), dim = -1)$unsqueeze(2)
    query_sine_embed <- gen_sineembed_for_position(refpoints_input[, , 1, ], d_model_half)
    query_pos <- self$ref_point_head(query_sine_embed)
    list(obj_center, refpoints_input, query_pos, query_sine_embed)
  },
  forward = function(tgt, memory, memory_key_padding_mask = NULL, pos = NULL,
                     refpoints_unsigmoid = NULL, level_start_index = NULL,
                     spatial_shapes = NULL, valid_ratios = NULL) {
    output <- tgt
    intermediate <- list()
    hs_refpoints <- list(refpoints_unsigmoid)
    d_model_half <- as.integer(memory$size(3) / 2)
    if (self$lite_refpoint_refine) {
      if (self$bbox_reparam) {
        ref_info <- self$get_reference(refpoints_unsigmoid, valid_ratios, d_model_half)
      } else {
        ref_info <- self$get_reference(refpoints_unsigmoid$sigmoid(), valid_ratios, d_model_half)
      }
    }
    for (layer_id in seq_len(self$num_layers)) {
      if (!self$lite_refpoint_refine) {
        if (self$bbox_reparam) {
          ref_info <- self$get_reference(refpoints_unsigmoid, valid_ratios, d_model_half)
        } else {
          ref_info <- self$get_reference(refpoints_unsigmoid$sigmoid(), valid_ratios, d_model_half)
        }
      }
      query_pos <- ref_info[[3]]
      output <- self$layers[[layer_id]](
        output, memory,
        tgt_mask = NULL,
        memory_key_padding_mask = memory_key_padding_mask,
        pos = pos,
        query_pos = query_pos,
        query_sine_embed = ref_info[[4]],
        is_first = (layer_id == 1),
        reference_points = ref_info[[2]],
        spatial_shapes = spatial_shapes,
        level_start_index = level_start_index
      )
      if (self$return_intermediate) {
        intermediate[[layer_id]] <- if (!is.null(self$norm)) self$norm(output) else output
      }
    }
    if (!is.null(self$norm)) {
      output <- self$norm(output)
    }
    if (self$return_intermediate) {
      intermediate[[self$num_layers]] <- output
    }
    if (self$return_intermediate) {
      list(torch_stack(intermediate, dim = 1), refpoints_unsigmoid$unsqueeze(1))
    } else {
      list(output$unsqueeze(1))
    }
  }
)

rfdetr_transformer <- nn_module(
  "rfdetr_transformer",
  initialize = function(d_model = 256, sa_nhead = 8, ca_nhead = 16,
                        num_queries = 300, num_decoder_layers = 6,
                        dim_feedforward = 2048, dropout = 0.0,
                        return_intermediate_dec = TRUE,
                        group_detr = 1, two_stage = FALSE,
                        num_feature_levels = 1, dec_n_points = 2,
                        lite_refpoint_refine = FALSE,
                        bbox_reparam = FALSE) {
    self$d_model <- d_model
    self$dec_layers <- num_decoder_layers
    self$group_detr <- group_detr
    self$num_feature_levels <- num_feature_levels
    self$bbox_reparam <- bbox_reparam
    decoder_layer <- rfdetr_decoder_layer(
      d_model, sa_nhead, ca_nhead, dim_feedforward, dropout,
      group_detr, num_feature_levels, dec_n_points
    )
    self$decoder <- rfdetr_decoder(
      decoder_layer, num_decoder_layers,
      norm = nn_layer_norm(d_model),
      return_intermediate = return_intermediate_dec,
      d_model = d_model,
      lite_refpoint_refine = lite_refpoint_refine,
      bbox_reparam = bbox_reparam
    )
    if (two_stage) {
      self$enc_output <- nn_module_list(lapply(seq_len(group_detr), function(i) nn_linear(d_model, d_model)))
      self$enc_output_norm <- nn_module_list(lapply(seq_len(group_detr), function(i) nn_layer_norm(d_model)))
    }
    self$two_stage <- two_stage
  },
  get_valid_ratio = function(mask) {
    mask <- mask$to(dtype = torch_float32())
    valid_height <- torch_sum((1 - mask[, , 1]), dim = 2)
    valid_width <- torch_sum((1 - mask[, 1, ]), dim = 2)
    valid_ratio_h <- valid_height$float() / mask$size(2)
    valid_ratio_w <- valid_width$float() / mask$size(3)
    torch_stack(list(valid_ratio_w, valid_ratio_h), dim = -1)
  },
  forward = function(srcs, masks, pos_embeds, refpoint_embed, query_feat) {
    src_flatten <- list()
    mask_flatten <- list()
    lvl_pos_embed_flatten <- list()
    spatial_shapes <- list()
    for (lvl in seq_along(srcs)) {
      src <- srcs[[lvl]]
      c <- src$size(2)
      h <- src$size(3)
      w <- src$size(4)
      spatial_shapes[[lvl]] <- c(h, w)
      src <- src$flatten(start_dim = 3)$transpose(2, 3)
      pos_embed <- pos_embeds[[lvl]]$flatten(start_dim = 3)$transpose(2, 3)
      lvl_pos_embed_flatten[[lvl]] <- pos_embed
      src_flatten[[lvl]] <- src
      if (!is.null(masks)) {
        mask_flatten[[lvl]] <- masks[[lvl]]$flatten(start_dim = 2)
      }
    }
    memory <- torch_cat(src_flatten, dim = 2)
    spatial_shapes_t <- torch_tensor(
      matrix(unlist(spatial_shapes), ncol = 2, byrow = TRUE),
      dtype = torch_int64(), device = memory$device
    )
    n_spatial <- nrow(spatial_shapes_t)
    level_start_index <- torch_cat(list(
      spatial_shapes_t$new_zeros(1),
      if (n_spatial > 1) cumsum(spatial_shapes_t[, 1] * spatial_shapes_t[, 2])[1:(n_spatial - 1)] else spatial_shapes_t$new_zeros(0)
    ), dim = 1)$to(dtype = torch_int64())
    if (!is.null(masks)) {
      mask_flatten <- torch_cat(mask_flatten, dim = 2)
      valid_ratios <- torch_stack(lapply(masks, function(m) self$get_valid_ratio(m)), dim = 2)
    } else {
      valid_ratios <- torch_stack(
        lapply(seq_len(n_spatial), function(i)
          torch_ones(c(memory$size(1), 2), dtype = torch_float32(), device = memory$device)
        ), dim = 2
      )
    }
    lvl_pos_embed_flatten <- torch_cat(lvl_pos_embed_flatten, dim = 2)
    hs <- NULL
    references <- NULL
    hs_enc <- NULL
    ref_enc <- NULL
    if (self$two_stage) {
      proposals <- gen_encoder_output_proposals(
        memory, if (!is.null(masks)) mask_flatten else NULL,
        spatial_shapes_t, unsigmoid = !self$bbox_reparam
      )
      output_memory <- proposals[[1]]
      output_proposals <- proposals[[2]]
      refpoint_embed_ts <- list()
      memory_ts <- list()
      boxes_ts <- list()
      gd <- if (self$training) self$group_detr else 1
      for (g_idx in seq_len(gd)) {
        om <- self$enc_output_norm[[g_idx]](self$enc_output[[g_idx]](output_memory))
        cls_enc_g <- self$enc_out_class_embed[[g_idx]](om)
        if (self$bbox_reparam) {
          delta_g <- self$enc_out_bbox_embed[[g_idx]](om)
          cxcy_g <- delta_g[, , 1:2] * output_proposals[, , 3:4] + output_proposals[, , 1:2]
          wh_g <- delta_g[, , 3:4]$exp() * output_proposals[, , 3:4]
          coord_g <- torch_cat(list(cxcy_g, wh_g), dim = -1)
        }
        topk <- refpoint_embed$size(1) %/% gd
        topk_idx <- cls_enc_g$max(dim = -1, keepdim = FALSE)[[1]]$topk(topk, dim = 2)[[2]]
        ref_g <- torch_gather(coord_g, 2, topk_idx$unsqueeze(3)$'repeat'(c(1, 1, 4)))
        tgt_g <- torch_gather(om, 2, topk_idx$unsqueeze(3)$'repeat'(c(1, 1, memory$size(3))))
        refpoint_embed_ts[[g_idx]] <- ref_g$detach()
        memory_ts[[g_idx]] <- tgt_g
        boxes_ts[[g_idx]] <- ref_g
      }
      refpoint_embed_ts <- torch_cat(refpoint_embed_ts, dim = 2)
      memory_ts <- torch_cat(memory_ts, dim = 2)
      boxes_ts <- torch_cat(boxes_ts, dim = 2)
    }
    if (self$dec_layers > 0) {
      tgt <- query_feat$unsqueeze(1)$expand(c(memory$size(1), -1, -1))$contiguous()
      refpoint_embed_exp <- refpoint_embed$unsqueeze(1)$expand(c(memory$size(1), -1, -1))$contiguous()
      if (self$two_stage) {
        ts_len <- refpoint_embed_ts$size(2)
        ref_subset <- refpoint_embed_exp[, 1:ts_len, , drop = FALSE]
        ref_remain <- refpoint_embed_exp[, (ts_len + 1):refpoint_embed_exp$size(2), , drop = FALSE]
        if (self$bbox_reparam) {
          cxcy <- ref_subset[, , 1:2] * refpoint_embed_ts[, , 3:4] + refpoint_embed_ts[, , 1:2]
          wh <- ref_subset[, , 3:4]$exp() * refpoint_embed_ts[, , 3:4]
          ref_subset <- torch_cat(list(cxcy, wh), dim = -1)
        }
        refpoint_embed_exp <- torch_cat(list(ref_subset, ref_remain), dim = 2)
      }
      dec_out <- self$decoder(
        tgt, memory,
        memory_key_padding_mask = if (!is.null(masks)) mask_flatten else NULL,
        pos = lvl_pos_embed_flatten,
        refpoints_unsigmoid = refpoint_embed_exp,
        level_start_index = level_start_index,
        spatial_shapes = spatial_shapes_t,
        valid_ratios = valid_ratios
      )
      hs <- dec_out[[1]]
      references <- dec_out[[2]]
    }
    if (self$two_stage) {
      list(hs, references, memory_ts, boxes_ts)
    } else {
      list(hs, references, NULL, NULL)
    }
  }
)

rfdetr_model <- nn_module(
  "rfdetr_model",
  initialize = function(backbone, transformer, num_classes = 91,
                        num_queries = 300, aux_loss = TRUE,
                        group_detr = 13, two_stage = TRUE,
                        lite_refpoint_refine = TRUE,
                        bbox_reparam = TRUE) {
    self$num_queries <- num_queries
    self$transformer <- transformer
    hidden_dim <- transformer$d_model
    self$class_embed <- nn_linear(hidden_dim, num_classes)
    self$bbox_embed <- detr_mlp_layer(hidden_dim, hidden_dim, 4, 3)
    query_dim <- 4
    self$refpoint_embed <- nn_embedding(num_queries * group_detr, query_dim)
    self$query_feat <- nn_embedding(num_queries * group_detr, hidden_dim)
    nn_init_constant_(self$refpoint_embed$weight, 0)
    self$backbone <- backbone
    self$aux_loss <- aux_loss
    self$group_detr <- group_detr
    self$lite_refpoint_refine <- lite_refpoint_refine
    if (!self$lite_refpoint_refine) {
      self$transformer$decoder$bbox_embed <- self$bbox_embed
    } else {
      self$transformer$decoder$bbox_embed <- NULL
    }
    self$bbox_reparam <- bbox_reparam
    prior_prob <- 0.01
    bias_value <- -log((1 - prior_prob) / prior_prob)
    self$class_embed$bias <- nn_parameter(torch_ones(num_classes) * bias_value)
    nn_init_constant_(self$bbox_embed$layers[[3]]$weight, 0)
    nn_init_constant_(self$bbox_embed$layers[[3]]$bias, 0)
    if (two_stage) {
      self$transformer$enc_out_bbox_embed <- nn_module_list(
        lapply(seq_len(group_detr), function(i) self$bbox_embed$clone(deep = TRUE))
      )
      self$transformer$enc_out_class_embed <- nn_module_list(
        lapply(seq_len(group_detr), function(i) self$class_embed$clone(deep = TRUE))
      )
    }
    self$two_stage <- two_stage
  },
  forward = function(x, mask = NULL) {
    if (is.list(x) && !is.null(x$tensors)) {
      mask <- x$mask
      x <- x$tensors
    }
    backbone_out <- self$backbone(x, mask)
    features <- backbone_out[[1]]
    poss <- backbone_out[[2]]
    srcs <- list()
    masks <- list()
    for (feat in features) {
      srcs[[length(srcs) + 1]] <- feat
      if (!is.null(mask)) {
        m <- nnf_interpolate(mask$unsqueeze(1)$float(), size = feat$shape[3:4])$squeeze(1)$to(dtype = torch_bool())
        masks[[length(masks) + 1]] <- m
      }
    }
    refpoint_embed_weight <- self$refpoint_embed$weight
    query_feat_weight <- self$query_feat$weight
    if (!self$training) {
      refpoint_embed_weight <- refpoint_embed_weight[1:self$num_queries, ]
      query_feat_weight <- query_feat_weight[1:self$num_queries, ]
    }
    trans_out <- self$transformer(
      srcs, if (length(masks) > 0) masks else NULL, poss,
      refpoint_embed_weight, query_feat_weight
    )
    hs <- trans_out[[1]]
    ref_unsigmoid <- trans_out[[2]]
    hs_enc <- trans_out[[3]]
    ref_enc <- trans_out[[4]]
    out <- list()
    if (!is.null(hs)) {
      if (self$bbox_reparam) {
        outputs_coord_delta <- self$bbox_embed(hs)
        outputs_coord_cxcy <- outputs_coord_delta[, , , 1:2] * ref_unsigmoid[, , , 3:4] + ref_unsigmoid[, , , 1:2]
        outputs_coord_wh <- outputs_coord_delta[, , , 3:4]$exp() * ref_unsigmoid[, , , 3:4]
        outputs_coord <- torch_cat(list(outputs_coord_cxcy, outputs_coord_wh), dim = -1)
      } else {
        outputs_coord <- (self$bbox_embed(hs) + ref_unsigmoid)$sigmoid()
      }
      outputs_class <- self$class_embed(hs)
      out$pred_logits <- outputs_class[hs$size(1), , , ]
      out$pred_boxes <- outputs_coord[hs$size(1), , , ]
      if (self$aux_loss) {
        aux <- list()
        for (i in seq_len(hs$size(1) - 1)) {
          aux[[i]] <- list(
            pred_logits = outputs_class[i, , , ],
            pred_boxes = outputs_coord[i, , , ]
          )
        }
        out$aux_outputs <- aux
      }
    }
    if (self$two_stage) {
      gd <- if (self$training) self$group_detr else 1
      hs_enc_list <- hs_enc$split(hs_enc$size(2) %/% gd, dim = 2)
      cls_enc_list <- list()
      for (g_idx in seq_len(gd)) {
        cls_enc_list[[g_idx]] <- self$transformer$enc_out_class_embed[[g_idx]](hs_enc_list[[g_idx]])
      }
      cls_enc <- torch_cat(cls_enc_list, dim = 2)
      if (!is.null(hs)) {
        out$enc_outputs <- list(pred_logits = cls_enc, pred_boxes = ref_enc)
      } else {
        out$pred_logits <- cls_enc
        out$pred_boxes <- ref_enc
      }
    }
    if (!self$training) {
      out_logits <- out$pred_logits
      out_bbox <- out$pred_boxes
      prob <- out_logits$sigmoid()
      num_select <- min(self$num_queries, out_logits$size(2))
      topk_values <- prob$view(c(out_logits$size(1), -1))$topk(num_select, dim = 2)
      scores <- topk_values[[1]]
      topk_indexes <- topk_values[[2]]
      flat_idx_0 <- topk_indexes - 1L
      topk_boxes <- flat_idx_0 %/% out_logits$size(3)
      labels <- flat_idx_0 %% out_logits$size(3)
      boxes_cxcywh <- out_bbox
      x1 <- boxes_cxcywh[, , 1] - boxes_cxcywh[, , 3] / 2
      y1 <- boxes_cxcywh[, , 2] - boxes_cxcywh[, , 4] / 2
      x2 <- boxes_cxcywh[, , 1] + boxes_cxcywh[, , 3] / 2
      y2 <- boxes_cxcywh[, , 2] + boxes_cxcywh[, , 4] / 2
      boxes_xyxy <- torch_stack(list(x1, y1, x2, y2), dim = -1)
      gather_idx <- (topk_boxes + 1L)$unsqueeze(3)$'repeat'(c(1, 1, 4))
      boxes_xyxy <- torch_gather(boxes_xyxy, 2, gather_idx)
      h <- x$size(3)
      w <- x$size(4)
      scale_fct <- torch_tensor(c(w, h, w, h), device = boxes_xyxy$device, dtype = boxes_xyxy$dtype)
      boxes_xyxy <- boxes_xyxy * scale_fct
      clamp_max <- max(h, w)
      x1 <- boxes_xyxy[, , 1]$clamp(min = 0)
      y1 <- boxes_xyxy[, , 2]$clamp(min = 0)
      x2 <- torch_maximum(boxes_xyxy[, , 3], x1 + 2)
      y2 <- torch_maximum(boxes_xyxy[, , 4], y1 + 2)
      boxes_xyxy <- torch_stack(list(x1, y1, x2, y2), dim = -1)$clamp(max = clamp_max)
      detections <- list()
      for (i in seq_len(boxes_xyxy$size(1))) {
        detections[[i]] <- list(
          scores = scores[i, ],
          labels = labels[i, ],
          boxes = boxes_xyxy[i, , ]
        )
      }
      return(list(detections = detections))
    }
    out
  }
)

rfdetr_postprocess <- nn_module(
  "rfdetr_postprocess",
  initialize = function(num_select = 300) {
    self$num_select <- num_select
  },
  forward = function(outputs, target_sizes) {
    out_logits <- outputs$pred_logits
    out_bbox <- outputs$pred_boxes
    prob <- out_logits$sigmoid()
    topk_values <- prob$view(c(out_logits$size(1), -1))$topk(self$num_select, dim = 2)
    scores <- topk_values[[1]]
    topk_indexes <- topk_values[[2]]
    flat_idx_0 <- topk_indexes - 1L
    topk_boxes <- flat_idx_0 %/% out_logits$size(3)
    labels <- flat_idx_0 %% out_logits$size(3)
    boxes <- out_bbox
    boxes_cxcywh <- boxes
    x1 <- boxes_cxcywh[, , 1] - boxes_cxcywh[, , 3] / 2
    y1 <- boxes_cxcywh[, , 2] - boxes_cxcywh[, , 4] / 2
    x2 <- boxes_cxcywh[, , 1] + boxes_cxcywh[, , 3] / 2
    y2 <- boxes_cxcywh[, , 2] + boxes_cxcywh[, , 4] / 2
    boxes_xyxy <- torch_stack(list(x1, y1, x2, y2), dim = -1)
    gather_idx <- (topk_boxes + 1L)$unsqueeze(3)$'repeat'(c(1, 1, 4))
    boxes_xyxy <- torch_gather(boxes_xyxy, 2, gather_idx)
    img_h <- target_sizes[, 1]
    img_w <- target_sizes[, 2]
    scale_fct <- torch_stack(list(img_w, img_h, img_w, img_h), dim = 2)
    boxes_xyxy <- boxes_xyxy * scale_fct$unsqueeze(2)
    x1 <- boxes_xyxy[, , 1]$clamp(min = 0)
    y1 <- boxes_xyxy[, , 2]$clamp(min = 0)
    x2 <- torch_maximum(boxes_xyxy[, , 3], x1 + 2)
    y2 <- torch_maximum(boxes_xyxy[, , 4], y1 + 2)
    boxes_xyxy <- torch_stack(list(x1, y1, x2, y2), dim = -1)$clamp(min = 0)
    results <- list()
    for (i in seq_len(boxes_xyxy$size(1))) {
      results[[i]] <- list(
        scores = scores[i, ],
        labels = labels[i, ],
        boxes = boxes_xyxy[i, , ]
      )
    }
    results
  }
)

rfdetr_configs <- list(
  nano = list(
    encoder = "dinov2_windowed_small",
    hidden_dim = 256,
    num_queries = 300,
    dec_layers = 2,
    sa_nheads = 8,
    ca_nheads = 16,
    dim_feedforward = 2048,
    dec_n_points = 2,
    group_detr = 13,
    out_feature_indexes = c(3, 6, 9, 12),
    patch_size = 16,
    num_windows = 2,
    num_classes = 91,
    num_register_tokens = 0,
    resolution = 384
  ),
  small = list(
    encoder = "dinov2_windowed_small",
    hidden_dim = 256,
    num_queries = 300,
    dec_layers = 2,
    sa_nheads = 8,
    ca_nheads = 16,
    dim_feedforward = 2048,
    dec_n_points = 2,
    group_detr = 13,
    out_feature_indexes = c(3, 6, 9, 12),
    patch_size = 16,
    num_windows = 2,
    num_classes = 91,
    num_register_tokens = 0,
    resolution = 512
  ),
  medium = list(
    encoder = "dinov2_windowed_small",
    hidden_dim = 256,
    num_queries = 300,
    dec_layers = 2,
    sa_nheads = 8,
    ca_nheads = 16,
    dim_feedforward = 2048,
    dec_n_points = 2,
    group_detr = 13,
    out_feature_indexes = c(3, 6, 9, 12),
    patch_size = 16,
    num_windows = 2,
    num_classes = 91,
    resolution = 640
  ),
  base = list(
    encoder = "dinov2_windowed_small",
    hidden_dim = 256,
    num_queries = 300,
    dec_layers = 3,
    sa_nheads = 8,
    ca_nheads = 16,
    dim_feedforward = 2048,
    dec_n_points = 2,
    group_detr = 13,
    out_feature_indexes = c(2, 5, 8, 11),
    patch_size = 14,
    num_windows = 4,
    num_classes = 91,
    resolution = 640
  ),
  large = list(
    encoder = "dinov2_windowed_base",
    hidden_dim = 384,
    num_queries = 300,
    dec_layers = 3,
    sa_nheads = 12,
    ca_nheads = 24,
    dim_feedforward = 2048,
    dec_n_points = 4,
    group_detr = 13,
    out_feature_indexes = c(2, 5, 8, 11),
    patch_size = 14,
    num_windows = 4,
    num_classes = 91,
    resolution = 560,
    projector_scale = c(2.0, 0.5)
  ),
  base_2 = list(
    encoder = "dinov2_windowed_small",
    hidden_dim = 256,
    num_queries = 300,
    dec_layers = 3,
    sa_nheads = 8,
    ca_nheads = 16,
    dim_feedforward = 2048,
    dec_n_points = 2,
    group_detr = 13,
    out_feature_indexes = c(2, 5, 8, 11),
    patch_size = 14,
    num_windows = 4,
    num_classes = 91,
    resolution = 640
  ),
  base_o365 = list(
    encoder = "dinov2_windowed_small",
    hidden_dim = 256,
    num_queries = 300,
    dec_layers = 3,
    sa_nheads = 8,
    ca_nheads = 16,
    dim_feedforward = 2048,
    dec_n_points = 2,
    group_detr = 13,
    out_feature_indexes = c(2, 5, 8, 11),
    patch_size = 14,
    num_windows = 4,
    num_classes = 366,
    resolution = 640
  )
)

#' @importFrom torch nn_parameter nn_linear nn_embedding nn_layer_norm nn_dropout
#' @importFrom torch nn_multihead_attention nn_conv2d nn_conv_transpose2d nn_batch_norm2d
#' @importFrom torch nn_relu nn_gelu nn_silu nnf_relu nnf_gelu nnf_silu nnf_softmax
#' @importFrom torch nnf_interpolate nnf_grid_sample nnf_layer_norm
#' @importFrom torch nn_init_trunc_normal_ nn_init_constant_ nn_init_xavier_uniform_ nn_init_zeros_ nn_init_ones_
#' @importFrom torch torch_randn torch_zeros torch_ones torch_arange torch_linspace torch_cat
#' @importFrom torch torch_stack torch_sum torch_cumsum torch_log
build_rfdetr <- function(cfg, pretrained = FALSE, progress = TRUE, name = NULL) {
  projector_scale <- cfg$projector_scale %||% c(1.0)
  backbone <- rfdetr_backbone(
    encoder_name = cfg$encoder,
    out_feature_indexes = cfg$out_feature_indexes,
    hidden_dim = cfg$hidden_dim,
    layer_norm = TRUE,
    target_shape = c(cfg$resolution, cfg$resolution),
    patch_size = cfg$patch_size,
    num_windows = cfg$num_windows,
    num_register_tokens = cfg$num_register_tokens %||% 0,
    projector_scale = projector_scale
  )
  pos_embed <- position_embedding_sine(
    num_pos_feats = as.integer(cfg$hidden_dim / 2),
    normalize = TRUE
  )
  joiner <- rfdetr_joiner(backbone, pos_embed)
  transformer <- rfdetr_transformer(
    d_model = cfg$hidden_dim,
    sa_nhead = cfg$sa_nheads,
    ca_nhead = cfg$ca_nheads,
    num_queries = cfg$num_queries,
    num_decoder_layers = cfg$dec_layers,
    dim_feedforward = cfg$dim_feedforward,
    dropout = 0.0,
    return_intermediate_dec = TRUE,
    group_detr = cfg$group_detr,
    two_stage = TRUE,
    num_feature_levels = length(projector_scale),
    dec_n_points = cfg$dec_n_points,
    lite_refpoint_refine = TRUE,
    bbox_reparam = TRUE
  )
  model <- rfdetr_model(
    backbone = joiner,
    transformer = transformer,
    num_classes = cfg$num_classes,
    num_queries = cfg$num_queries,
    aux_loss = TRUE,
    group_detr = cfg$group_detr,
    two_stage = TRUE,
    lite_refpoint_refine = TRUE,
    bbox_reparam = TRUE
  )
  if (pretrained) {
    if (is.null(name)) {
      runtime_error("Internal error: variant name required for pretrained weights")
    }
    r <- rfdetr_torchscript_urls[[name]]
    if (is.null(r)) {
      runtime_error("Pretrained weights not available for this variant")
    }
    cli_inform("Model weights for {.cls {name}} (~{.emph {r[3]}}) will be downloaded and processed if not already available.")
    archive <- download_and_cache(r[1], prefix = name)
    if (tools::md5sum(archive) != r[2]) {
      runtime_error("Corrupt file! Delete the file in {archive} and try again.")
    }
    state_dict <- torch::load_state_dict(archive)
    model_sd <- model$state_dict()
    loaded <- 0
    skipped <- 0
    local({
      with_no_grad({
        for (k in names(state_dict)) {
          if (is.null(model_sd[[k]])) {
            skipped <<- skipped + 1
            next
          }
          if (!identical(dim(state_dict[[k]]), dim(model_sd[[k]]))) {
            if (grepl("position_embeddings", k)) {
              sd_w <- state_dict[[k]]
              m_w <- model_sd[[k]]
              cls_pos <- sd_w[, 1, , drop = FALSE]
              patch_pos <- sd_w[, 2:sd_w$size(2), , drop = FALSE]
              dim <- patch_pos$size(3)
              num_patches <- patch_pos$size(2)
              h <- w <- as.integer(sqrt(num_patches))
              patch_pos <- patch_pos$reshape(c(1, h, w, dim))
              patch_pos <- patch_pos$permute(c(1, 4, 2, 3))
              target_h <- m_w$size(2) - 1
              target_w <- target_h
              target_size <- as.integer(sqrt(target_h))
              patch_pos <- nnf_interpolate(
                patch_pos$to(dtype = torch_float32()),
                size = c(target_size, target_size),
                mode = "bicubic",
                align_corners = FALSE
              )$to(dtype = sd_w$dtype)
              patch_pos <- patch_pos$permute(c(1, 3, 4, 2))$reshape(c(1, -1, dim))
              interpolated <- torch_cat(list(cls_pos, patch_pos), dim = 2)
              m_w$copy_(interpolated)
              loaded <<- loaded + 1
            } else {
              skipped <<- skipped + 1
            }
            next
          }
          model_sd[[k]]$copy_(state_dict[[k]])
          loaded <<- loaded + 1
        }
      })
    })
    model$load_state_dict(model_sd, strict = FALSE)
    total <- length(model_sd)
    cli_inform("Loaded pretrained weights for {.cls {name}} ({loaded}/{total} keys, {skipped} skipped).")
  }
  model
}

#' @describeIn model_rfdetr RF-DETR Nano (fastest, COCO, 384px)
#' @export
model_rfdetr_nano <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfg <- rfdetr_configs$nano
  cfg[names(list(...))] <- list(...)
  build_rfdetr(cfg, pretrained, progress, name = "rfdetr_nano")
}

#' @describeIn model_rfdetr RF-DETR Small (lightweight, COCO, 512px)
#' @export
model_rfdetr_small <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfg <- rfdetr_configs$small
  cfg[names(list(...))] <- list(...)
  build_rfdetr(cfg, pretrained, progress, name = "rfdetr_small")
}

#' @describeIn model_rfdetr RF-DETR Medium (balanced speed/accuracy, COCO, 640px)
#' @export
model_rfdetr_medium <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfg <- rfdetr_configs$medium
  cfg[names(list(...))] <- list(...)
  build_rfdetr(cfg, pretrained, progress, name = "rfdetr_medium")
}

#' @describeIn model_rfdetr RF-DETR Base (COCO pretrained, 640px)
#' @export
model_rfdetr_base <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfg <- rfdetr_configs$base
  cfg[names(list(...))] <- list(...)
  build_rfdetr(cfg, pretrained, progress, name = "rfdetr_base")
}

#' @describeIn model_rfdetr RF-DETR Base variant 2 (alternative COCO training run)
#' @export
model_rfdetr_base_2 <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfg <- rfdetr_configs$base_2
  cfg[names(list(...))] <- list(...)
  build_rfdetr(cfg, pretrained, progress, name = "rfdetr_base_2")
}

#' @describeIn model_rfdetr RF-DETR Base O365 (Objects365, 366 classes)
#' @export
model_rfdetr_base_o365 <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfg <- rfdetr_configs$base_o365
  cfg[names(list(...))] <- list(...)
  build_rfdetr(cfg, pretrained, progress, name = "rfdetr_base_o365")
}

#' @describeIn model_rfdetr RF-DETR Large (DINOv2 Base backbone, COCO, 560px)
#' @export
model_rfdetr_large <- function(pretrained = FALSE, progress = TRUE, ...) {
  cfg <- rfdetr_configs$large
  cfg[names(list(...))] <- list(...)
  build_rfdetr(cfg, pretrained, progress, name = "rfdetr_large")
}
