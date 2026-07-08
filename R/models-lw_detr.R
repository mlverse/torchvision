# LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection
# Paper: https://arxiv.org/abs/2406.03459
# Reference impl: https://github.com/Atten4Vis/LW-DETR

# Utility helpers

# Channel-wise LayerNorm for (B, C, H, W)
#' @importFrom torch torch_ones_like with_no_grad torch_topk torch_gather torch_div torch_bool
lw_detr_channel_layer_norm <- nn_module(
  initialize = function(channels, eps = 1e-6) {
    self$weight <- nn_parameter(torch_ones(channels))
    self$bias <- nn_parameter(torch_zeros(channels))
    self$eps <- eps
  },
  forward = function(x) {
    u <- x$mean(2L, keepdim = TRUE)
    s <- (x - u)$pow(2)$mean(2L, keepdim = TRUE)
    x <- (x - u) / (s + self$eps)$sqrt()
    self$weight[, NULL, NULL] * x + self$bias[, NULL, NULL]
  }
)

# Sinusoidal embedding for 4D reference points
lw_detr_gen_sineembed <- function(pos, dim = 128L) {
  scale <- 2 * pi
  dim_t <- torch_arange(dim, dtype = torch_float32(), device = pos$device)
  dim_t <- 10000^(2 * torch_div(dim_t, 2L, rounding_mode = "floor") / dim)

  coords <- list()
  for (c_idx in seq_len(pos$size(-1))) {
    v <- pos[,, c_idx] * scale
    pe <- v$unsqueeze(3) / dim_t
    pe_s <- torch_stack(list(pe[,, seq(1, dim, 2)]$sin(), pe[,, seq(2, dim, 2)]$cos()), dim = 4)$flatten(
      start_dim = 3L
    )
    coords[[c_idx]] <- pe_s
  }
  # DETR convention: swap the x and y embeddings before concatenating
  if (length(coords) >= 2L) {
    coords[c(1L, 2L)] <- coords[c(2L, 1L)]
  }
  torch_cat(coords, dim = 3)
}

# Generate anchor proposals from the feature grid for two-stage selection.
# `masks` is a list of (B, H_l, W_l) logical tensors (TRUE = valid pixel); the
# proposal centres are normalised by the valid extent so padding is ignored.
lw_detr_gen_proposals <- function(spatial_shapes, masks, bs, device) {
  proposals <- list()
  for (lvl in seq_len(nrow(spatial_shapes))) {
    h_l <- as.integer(spatial_shapes[lvl, 1])
    w_l <- as.integer(spatial_shapes[lvl, 2])
    m <- masks[[lvl]]
    valid_h <- m[,, 1]$to(dtype = torch_float32())$sum(dim = 2L) # (B,)
    valid_w <- m[, 1, ]$to(dtype = torch_float32())$sum(dim = 2L) # (B,)

    gy <- torch_linspace(0, h_l - 1, h_l, device = device)
    gx <- torch_linspace(0, w_l - 1, w_l, device = device)
    grids <- torch_meshgrid(list(gy, gx), indexing = "ij")
    grid <- torch_stack(list(grids[[2]]$flatten(), grids[[1]]$flatten()), dim = 2L)
    grid <- grid$unsqueeze(1L)$expand(c(bs, -1L, -1L)) # (B, HW, 2) x,y

    scale <- torch_stack(list(valid_w, valid_h), dim = -1L)$view(c(bs, 1L, 2L))
    grid <- (grid + 0.5) / scale
    wh <- torch_ones_like(grid) * (0.05 * (2.0^(lvl - 1L)))
    proposals[[lvl]] <- torch_cat(list(grid, wh), dim = -1L) # (B, HW, 4)
  }
  torch_cat(proposals, dim = 2L)
}


# ViT backbone

# Inner QKV attention: $query, $key (no bias), $value
.lw_detr_inner_attention <- nn_module(
  initialize = function(dim, num_heads) {
    self$query <- nn_linear(dim, dim)
    self$key <- nn_linear(dim, dim, bias = FALSE)
    self$value <- nn_linear(dim, dim)
    self$num_heads <- num_heads
    self$head_dim <- dim %/% num_heads
    self$scale <- (dim %/% num_heads)^(-0.5)
  },
  forward = function(x) {
    c(B, N, C) %<-% x$shape
    nh <- self$num_heads
    hd <- self$head_dim

    q <- self$query(x)$reshape(c(B, N, nh, hd))$permute(c(1L, 3L, 2L, 4L))
    k <- self$key(x)$reshape(c(B, N, nh, hd))$permute(c(1L, 3L, 2L, 4L))
    v <- self$value(x)$reshape(c(B, N, nh, hd))$permute(c(1L, 3L, 2L, 4L))

    attn <- (q * self$scale)$matmul(k$transpose(-2L, -1L))
    attn <- nnf_softmax(attn, dim = -1L)
    (attn$matmul(v))$transpose(2L, 3L)$reshape(c(B, N, C))
  }
)

# Outer attention
.lw_detr_outer_attention <- nn_module(
  initialize = function(dim, num_heads) {
    self$attention <- .lw_detr_inner_attention(dim, num_heads)
    self$output <- nn_linear(dim, dim)
  },
  forward = function(x) {
    self$output(self$attention(x))
  }
)

# FFN stored as $intermediate with $fc1 and $fc2
.lw_detr_vit_ffn <- nn_module(
  initialize = function(dim, mlp_ratio = 4.0) {
    hidden <- as.integer(dim * mlp_ratio)
    self$fc1 <- nn_linear(dim, hidden)
    self$fc2 <- nn_linear(hidden, dim)
  },
  forward = function(x) {
    self$fc2(nnf_gelu(self$fc1(x)))
  }
)

# ViT block
.lw_detr_vit_block <- nn_module(
  initialize = function(dim, num_heads, window = FALSE) {
    self$attention <- .lw_detr_outer_attention(dim, num_heads)
    self$gamma_1 <- nn_parameter(torch_ones(dim) * 0.1)
    self$gamma_2 <- nn_parameter(torch_ones(dim) * 0.1)
    self$intermediate <- .lw_detr_vit_ffn(dim)
    self$layernorm_before <- nn_layer_norm(dim, eps = 1e-6)
    self$layernorm_after <- nn_layer_norm(dim, eps = 1e-6)
    self$window <- window
  },
  forward = function(x) {
    shortcut <- x

    if (!self$window) {
      c(bw, N, C) %<-% x$shape
      B <- bw %/% 16L
      x_norm <- self$layernorm_before(x)$reshape(c(B, 16L * N, C))
      attn_out <- self$attention(x_norm)$reshape(c(bw, N, C))
    } else {
      attn_out <- self$attention(self$layernorm_before(x))
    }

    x <- shortcut + self$gamma_1 * attn_out
    x <- x + self$gamma_2 * self$intermediate(self$layernorm_after(x))
    x
  }
)

# ViT encoder
.lw_detr_vit_encoder <- nn_module(
  initialize = function(embed_dim, depth, num_heads, window_block_indexes) {
    self$layer <- nn_module_list(lapply(seq_len(depth), function(i) {
      .lw_detr_vit_block(embed_dim, num_heads, window = (i - 1L) %in% window_block_indexes)
    }))
    self$depth <- depth
  },
  forward = function(x, out_flags) {
    out <- vector("list", sum(out_flags))
    j <- 0L
    for (i in seq_len(self$depth)) {
      x <- self$layer[[i]](x)
      if (out_flags[i]) {
        j <- j + 1L
        out[[j]] <- x
      }
    }
    out
  }
)

# ViT embeddings
.lw_detr_vit_embeddings <- nn_module(
  initialize = function(embed_dim = 192L, patch_size = 16L, pretrain_img_size = 224L) {
    num_patches <- (pretrain_img_size %/% patch_size)^2L
    self$projection <- nn_conv2d(3L, embed_dim, patch_size, stride = patch_size)
    self$position_embeddings <- nn_parameter(
      torch_zeros(1L, num_patches + 1L, embed_dim)
    )
    self$pretrain_size <- pretrain_img_size %/% patch_size
  },
  forward = function(x) {
    x <- self$projection(x)
    c(B, C, H, W) %<-% x$shape

    pos <- self$position_embeddings[, 2:self$position_embeddings$size(2), ]
    ps <- self$pretrain_size
    if (ps != H || ps != W) {
      pos <- pos$reshape(c(1L, ps, ps, C))$permute(c(1L, 4L, 2L, 3L))
      pos <- nnf_interpolate(pos, size = c(H, W), mode = "bicubic", align_corners = FALSE)
      pos <- pos$permute(c(1L, 3L, 4L, 2L))
    } else {
      pos <- pos$reshape(c(1L, H, W, C))
    }
    x$permute(c(1L, 3L, 4L, 2L)) + pos
  }
)

# Full ViT backbone
.lw_detr_vit_backbone <- nn_module(
  initialize = function(embed_dim, depth, num_heads, window_block_indexes, out_feature_indexes) {
    self$embeddings <- .lw_detr_vit_embeddings(embed_dim)
    self$encoder <- .lw_detr_vit_encoder(embed_dim, depth, num_heads, window_block_indexes)
    self$out_flags <- (seq_len(depth) - 1L) %in% out_feature_indexes
  },
  forward = function(x) {
    patches <- self$embeddings(x)
    c(B, H, W, C) %<-% patches$shape

    h <- H %/% 4L
    w <- W %/% 4L
    xw <- patches$reshape(c(B, 4L, h, 4L, w, C))$permute(c(1L, 2L, 4L, 3L, 5L, 6L))
    xw <- xw$reshape(c(B * 16L, h * w, C))

    win_feats <- self$encoder(xw, self$out_flags)

    lapply(win_feats, function(f) {
      f$reshape(c(B, 4L, 4L, h, w, C))$permute(c(1L, 6L, 2L, 4L, 3L, 5L))$reshape(c(B, C, H, W))
    })
  }
)


# C2f projector

# ConvX
.lw_detr_conv_x <- nn_module(
  initialize = function(in_ch, out_ch, kernel = 3L, stride = 1L) {
    pad <- kernel %/% 2L
    self$conv <- nn_conv2d(in_ch, out_ch, kernel, stride = stride, padding = pad, bias = FALSE)
    self$norm <- nn_batch_norm2d(out_ch)
  },
  forward = function(x) {
    nnf_silu(self$norm(self$conv(x)))
  }
)

# Bottleneck
.lw_detr_bottleneck <- nn_module(
  initialize = function(c) {
    self$conv1 <- .lw_detr_conv_x(c, c, 3L)
    self$conv2 <- .lw_detr_conv_x(c, c, 3L)
  },
  forward = function(x) {
    x + self$conv2(self$conv1(x))
  }
)

# C2f projector_layer
.lw_detr_c2f <- nn_module(
  initialize = function(c1, c2, n = 3L) {
    c <- c2 %/% 2L
    self$conv1 <- .lw_detr_conv_x(c1, 2L * c, 1L)
    self$conv2 <- .lw_detr_conv_x((2L + n) * c, c2, 1L)
    self$bottlenecks <- nn_module_list(lapply(seq_len(n), function(i) {
      .lw_detr_bottleneck(c)
    }))
    self$n <- n
  },
  forward = function(x) {
    halves <- torch_chunk(self$conv1(x), 2L, dim = 2L)
    y <- list(halves[[1]], halves[[2]])
    for (i in seq_len(self$n)) {
      y[[length(y) + 1L]] <- self$bottlenecks[[i]](y[[length(y)]])
    }
    self$conv2(torch_cat(y, dim = 2L))
  }
)

# Sampling layer wrapper
.lw_detr_sampling_layer <- nn_module(
  initialize = function(op) {
    self$layers <- nn_module_list(list(op))
  },
  forward = function(x) {
    self$layers[[1]](x)
  }
)

# Scale layer
.lw_detr_scale_layer <- nn_module(
  initialize = function(total_in_ch, out_ch, n_blocks, sampling_ops = NULL) {
    if (!is.null(sampling_ops)) {
      self$sampling_layers <- nn_module_list(lapply(sampling_ops, function(op) {
        .lw_detr_sampling_layer(op)
      }))
    }
    self$projector_layer <- .lw_detr_c2f(total_in_ch, out_ch, n_blocks)
    self$layer_norm <- lw_detr_channel_layer_norm(out_ch)
    self$has_sampling <- !is.null(sampling_ops)
  },
  forward = function(feats) {
    if (self$has_sampling) {
      feats <- lapply(seq_along(feats), function(j) self$sampling_layers[[j]](feats[[j]]))
    }
    x <- torch_cat(feats, dim = 2L)
    self$layer_norm(self$projector_layer(x))
  }
)

# Full projector
.lw_detr_projector <- nn_module(
  initialize = function(scale_layers_list) {
    self$scale_layers <- nn_module_list(scale_layers_list)
    self$n_scales <- length(scale_layers_list)
  },
  forward = function(feats) {
    lapply(seq_len(self$n_scales), function(s) self$scale_layers[[s]](feats))
  }
)


# Multi-scale deformable attention

# Adapted from atten4vis/lw-detr
detr_ms_deform_attn <- nn_module(
  initialize = function(d_model = 256L, n_levels = 4L, n_heads = 8L, n_points = 4L) {
    self$n_levels <- n_levels
    self$n_heads <- n_heads
    self$n_points <- n_points
    self$head_dim <- d_model %/% n_heads

    self$sampling_offsets <- nn_linear(d_model, n_heads * n_levels * n_points * 2L)
    self$attention_weights <- nn_linear(d_model, n_heads * n_levels * n_points)
    self$value_proj <- nn_linear(d_model, d_model)
    self$output_proj <- nn_linear(d_model, d_model)

    with_no_grad({
      nn_init_constant_(self$sampling_offsets$weight, 0)
      thetas <- torch_arange(n_heads, dtype = torch_float32()) * (2 * pi / n_heads)
      grid_init <- torch_stack(list(thetas$cos(), thetas$sin()), dim = -1L)
      grid_init <- (grid_init / grid_init$abs()$amax(-1L, keepdim = TRUE))
      grid_init <- grid_init$reshape(c(n_heads, 1L, 1L, 2L))$`repeat`(c(1L, n_levels, n_points, 1L))
      for (i in seq_len(n_points)) {
        grid_init[,, i, ] <- grid_init[,, i, ] * i
      }
      self$sampling_offsets$bias <- nn_parameter(grid_init$reshape(c(-1L)))

      nn_init_constant_(self$attention_weights$weight, 0)
      nn_init_constant_(self$attention_weights$bias, 0)
      nn_init_xavier_uniform_(self$value_proj$weight)
      nn_init_constant_(self$value_proj$bias, 0)
      nn_init_xavier_uniform_(self$output_proj$weight)
      nn_init_constant_(self$output_proj$bias, 0)
    })
  },
  forward = function(query, reference_points, input_flatten, spatial_shapes, level_start_index, mask = NULL) {
    bs <- query$size(1L)
    lenq <- query$size(2L)
    nh <- self$n_heads
    nl <- self$n_levels
    np <- self$n_points
    hd <- self$head_dim

    value <- self$value_proj(input_flatten)
    if (!is.null(mask)) {
      value <- value$masked_fill(mask, 0)
    }
    offsets <- self$sampling_offsets(query)$reshape(c(bs, lenq, nh, nl, np, 2L))
    attn_w <- nnf_softmax(
      self$attention_weights(query)$reshape(c(bs, lenq, nh, nl * np)),
      dim = -1L
    )

    if (reference_points$size(-1L) == 2L) {
      offset_normalizer <- torch_stack(list(
        spatial_shapes[, 2L], spatial_shapes[, 1L]
      ), dim = -1L)
      sampling_locs <- reference_points$unsqueeze(3L)$unsqueeze(5L) +
        offsets / offset_normalizer$unsqueeze(1L)$unsqueeze(1L)$unsqueeze(4L)
    } else {
      ref_xy <- reference_points[,,, 1:2]
      ref_wh <- reference_points[,,, 3:4]
      sampling_locs <- ref_xy$unsqueeze(3L)$unsqueeze(5L) +
        offsets / np * ref_wh$unsqueeze(3L)$unsqueeze(5L) * 0.5
    }

    val_split <- list()
    for (lvl in seq_len(nl)) {
      h_l <- as.integer(spatial_shapes[lvl, 1])
      w_l <- as.integer(spatial_shapes[lvl, 2])
      s <- as.integer(level_start_index[lvl]) + 1L
      e <- s + h_l * w_l - 1L
      val_l <- value[, s:e, ]$reshape(c(bs, h_l, w_l, nh, hd))
      val_l <- val_l$permute(c(1L, 4L, 5L, 2L, 3L))$reshape(c(bs * nh, hd, h_l, w_l))
      val_split[[lvl]] <- val_l
    }

    sampling_grids <- 2 * sampling_locs - 1

    out_list <- list()
    for (lvl in seq_len(nl)) {
      grid_l <- sampling_grids[,,, lvl, , ]
      grid_l <- grid_l$permute(c(1L, 3L, 2L, 4L, 5L))
      grid_l <- grid_l$reshape(c(bs * nh, lenq, np, 2L))

      sampled <- nnf_grid_sample(
        val_split[[lvl]],
        grid_l,
        mode = "bilinear",
        padding_mode = "zeros",
        align_corners = FALSE
      )
      out_list[[lvl]] <- sampled
    }

    out_vals <- torch_cat(out_list, dim = -1L)
    attn_w2 <- attn_w$permute(c(1L, 3L, 2L, 4L))$reshape(c(bs * nh, 1L, lenq, nl * np))
    output <- (out_vals * attn_w2)$sum(-1L)$reshape(c(bs, nh * hd, lenq))
    self$output_proj(output$permute(c(1L, 3L, 2L)))
  }
)


# Decoder

# MLP with $layers nn_module_list
detr_mlp_layer <- nn_module(
  initialize = function(input_dim, hidden_dim, output_dim, num_layers) {
    dims_in <- c(input_dim, rep(hidden_dim, num_layers - 1L))
    dims_out <- c(rep(hidden_dim, num_layers - 1L), output_dim)
    self$layers <- nn_module_list(mapply(
      function(di, do) nn_linear(di, do),
      dims_in,
      dims_out,
      SIMPLIFY = FALSE
    ))
    self$n <- num_layers
  },
  forward = function(x) {
    for (i in seq_len(self$n)) {
      x <- self$layers[[i]](x)
      if (i < self$n) x <- nnf_relu(x)
    }
    x
  }
)

# Decoder FFN
.lw_detr_dec_ffn <- nn_module(
  initialize = function(d_model, dim_feedforward) {
    self$fc1 <- nn_linear(d_model, dim_feedforward)
    self$fc2 <- nn_linear(dim_feedforward, d_model)
  },
  forward = function(x) {
    self$fc2(nnf_relu(self$fc1(x)))
  }
)

# Decoder self-attention
.lw_detr_dec_self_attn <- nn_module(
  initialize = function(d_model, n_heads) {
    self$q_proj <- nn_linear(d_model, d_model)
    self$k_proj <- nn_linear(d_model, d_model)
    self$v_proj <- nn_linear(d_model, d_model)
    self$o_proj <- nn_linear(d_model, d_model)
    self$n_heads <- n_heads
    self$head_dim <- d_model %/% n_heads
    self$scale <- (d_model %/% n_heads)^(-0.5)
  },
  forward = function(x, x_value = NULL) {
    if (is.null(x_value)) {
      x_value <- x
    }
    c(B, N, C) %<-% x$shape
    nh <- self$n_heads
    hd <- self$head_dim

    q <- self$q_proj(x)$reshape(c(B, N, nh, hd))$permute(c(1L, 3L, 2L, 4L))
    k <- self$k_proj(x)$reshape(c(B, N, nh, hd))$permute(c(1L, 3L, 2L, 4L))
    v <- self$v_proj(x_value)$reshape(c(B, N, nh, hd))$permute(c(1L, 3L, 2L, 4L))

    attn <- (q * self$scale)$matmul(k$transpose(-2L, -1L))
    attn <- nnf_softmax(attn, dim = -1L)
    out <- (attn$matmul(v))$transpose(2L, 3L)$reshape(c(B, N, C))
    self$o_proj(out)
  }
)

# Decoder layer
.lw_detr_decoder_layer <- nn_module(
  initialize = function(d_model, sa_nhead, ca_nhead, dim_feedforward = 2048L, n_levels = 1L, n_points = 4L) {
    self$self_attn <- .lw_detr_dec_self_attn(d_model, sa_nhead)
    self$self_attn_layer_norm <- nn_layer_norm(d_model)
    self$cross_attn <- detr_ms_deform_attn(d_model, n_levels, ca_nhead, n_points)
    self$cross_attn_layer_norm <- nn_layer_norm(d_model)
    self$mlp <- .lw_detr_dec_ffn(d_model, dim_feedforward)
    self$layer_norm <- nn_layer_norm(d_model)
  },
  forward = function(tgt, query_pos, memory, reference_points, spatial_shapes, level_start_index, mask = NULL) {
    sa_qk <- tgt + query_pos
    tgt <- self$self_attn_layer_norm(tgt + self$self_attn(sa_qk, tgt))

    ca_out <- self$cross_attn(
      tgt + query_pos, reference_points, memory, spatial_shapes, level_start_index,
      mask = if (!is.null(mask)) mask$logical_not()$unsqueeze(-1L) else NULL
    )
    tgt <- self$cross_attn_layer_norm(tgt + ca_out)
    tgt <- self$layer_norm(tgt + self$mlp(tgt))
    tgt
  }
)

# Decoder
.lw_detr_decoder <- nn_module(
  initialize = function(
    d_model,
    num_layers,
    sa_nhead,
    ca_nhead,
    dim_feedforward = 2048L,
    n_levels = 1L,
    n_points = 4L
  ) {
    self$layers <- nn_module_list(lapply(seq_len(num_layers), function(i) {
      .lw_detr_decoder_layer(d_model, sa_nhead, ca_nhead, dim_feedforward, n_levels, n_points)
    }))
    self$layernorm <- nn_layer_norm(d_model)

    self$ref_point_head <- detr_mlp_layer(2L * d_model, d_model, d_model, 2L)

    self$num_layers <- num_layers
    self$d_model <- d_model
  },
  forward = function(tgt, memory, refpoints, spatial_shapes, level_start_index, valid_ratios, mask = NULL) {
    vr2 <- torch_cat(list(valid_ratios, valid_ratios), dim = -1L)
    ref_pts <- refpoints$unsqueeze(3L) * vr2$unsqueeze(2L)

    sine_emb <- lw_detr_gen_sineembed(ref_pts[,, 1, ], self$d_model %/% 2L)
    query_pos <- self$ref_point_head(sine_emb)

    for (i in seq_len(self$num_layers)) {
      tgt <- self$layers[[i]](tgt, query_pos, memory, ref_pts, spatial_shapes, level_start_index, mask)
    }
    self$layernorm(tgt)
  }
)


# Inner LW-DETR model

.lw_detr_inner_model <- nn_module(
  initialize = function(
    embed_dim,
    depth,
    num_heads,
    window_block_indexes,
    out_feature_indexes,
    scale_layers_list,
    d_model,
    sa_nhead,
    ca_nhead,
    num_queries,
    num_decoder_layers,
    dim_feedforward,
    n_levels,
    n_points,
    num_classes,
    group_detr = 13L
  ) {
    self$backbone <- nn_module(
      initialize = function() {
        self$backbone <- .lw_detr_vit_backbone(
          embed_dim,
          depth,
          num_heads,
          window_block_indexes,
          out_feature_indexes
        )
        self$projector <- .lw_detr_projector(scale_layers_list)
      },
      forward = function(x) {
        self$projector(self$backbone(x))
      }
    )()

    self$decoder <- .lw_detr_decoder(
      d_model,
      num_decoder_layers,
      sa_nhead,
      ca_nhead,
      dim_feedforward,
      n_levels,
      n_points
    )

    self$enc_out_class_embed <- nn_module_list(lapply(seq_len(group_detr), function(g) {
      nn_linear(d_model, num_classes)
    }))
    self$enc_out_bbox_embed <- nn_module_list(lapply(seq_len(group_detr), function(g) {
      detr_mlp_layer(d_model, d_model, 4L, 3L)
    }))
    self$enc_output <- nn_module_list(lapply(seq_len(group_detr), function(g) {
      nn_linear(d_model, d_model)
    }))
    self$enc_output_norm <- nn_module_list(lapply(seq_len(group_detr), function(g) {
      nn_layer_norm(d_model)
    }))

    total_q <- num_queries * group_detr
    self$query_feat <- nn_embedding(total_q, d_model)
    self$reference_point_embed <- nn_embedding(total_q, 4L)

    self$d_model <- d_model
    self$num_queries <- num_queries
  },
  forward = function(images, class_embed_fn, bbox_embed_fn, pixel_mask) {
    bs <- images$size(1L)
    device <- images$device

    feats <- self$backbone(images)
    pm_f <- pixel_mask$unsqueeze(2L)$to(dtype = torch_float32()) # (B, 1, H, W)

    n_lvl <- length(feats)
    src_flat <- vector("list", n_lvl)
    masks_lvl <- vector("list", n_lvl)
    mask_list <- vector("list", n_lvl)
    shapes <- vector("list", n_lvl)
    lvl_start <- integer(n_lvl)
    cur <- 0L
    for (i in seq_along(feats)) {
      f <- feats[[i]]
      h_i <- as.integer(f$size(3L))
      w_i <- as.integer(f$size(4L))
      shapes[[i]] <- c(h_i, w_i)
      lvl_start[i] <- cur
      cur <- cur + h_i * w_i
      src_flat[[i]] <- f$flatten(start_dim = 3L)$permute(c(1L, 3L, 2L))
      m <- (nnf_interpolate(pm_f, size = c(h_i, w_i)) > 0.5)$squeeze(2L) # (B, H_l, W_l)
      masks_lvl[[i]] <- m
      mask_list[[i]] <- m$flatten(start_dim = 2L)
    }
    memory <- torch_cat(src_flat, dim = 2L)
    mask_flat <- torch_cat(mask_list, dim = 2L)
    spatial_shapes <- do.call(rbind, shapes)

    valid_ratios <- torch_stack(
      lapply(masks_lvl, function(m) {
        vh <- m[,, 1]$to(dtype = torch_float32())$sum(dim = 2L) / m$size(2L)
        vw <- m[, 1, ]$to(dtype = torch_float32())$sum(dim = 2L) / m$size(3L)
        torch_stack(list(vw, vh), dim = -1L)
      }),
      dim = 2L
    )

    out_proposals <- lw_detr_gen_proposals(spatial_shapes, masks_lvl, bs, device)
    prop_valid <- ((out_proposals > 0.01) & (out_proposals < 0.99))$all(-1L)
    invalid_mask <- (mask_flat$logical_not() | prop_valid$logical_not())$unsqueeze(-1L)

    out_mem_g0 <- self$enc_output_norm[[1]](self$enc_output[[1]](memory))
    out_mem_g0 <- out_mem_g0$masked_fill(invalid_mask, 0)

    cls_g0 <- self$enc_out_class_embed[[1]](out_mem_g0)$masked_fill(invalid_mask, -Inf)
    bbox_g0 <- self$enc_out_bbox_embed[[1]](out_mem_g0)

    enc_cxcy <- bbox_g0[,, 1:2] * out_proposals[,, 3:4] + out_proposals[,, 1:2]
    enc_wh <- torch_exp(bbox_g0[,, 3:4]) * out_proposals[,, 3:4]
    enc_boxes <- torch_cat(list(enc_cxcy, enc_wh), dim = -1L)

    topk_idx <- torch_topk(cls_g0$amax(-1L), self$num_queries, dim = 2L)[[2]]
    ref_from_enc <- torch_gather(
      enc_boxes,
      2L,
      topk_idx$unsqueeze(-1L)$expand(c(-1L, -1L, 4L))
    )

    qfeat <- self$query_feat$weight[1:self$num_queries, ]
    qref <- self$reference_point_embed$weight[1:self$num_queries, ]

    tgt <- qfeat$unsqueeze(1L)$expand(c(bs, -1L, -1L))
    qref_exp <- qref$unsqueeze(1L)$expand(c(bs, -1L, -1L))

    ref_cxcy <- qref_exp[,, 1:2] * ref_from_enc[,, 3:4] + ref_from_enc[,, 1:2]
    ref_wh <- torch_exp(qref_exp[,, 3:4]) * ref_from_enc[,, 3:4]
    refpoints_dec <- torch_cat(list(ref_cxcy, ref_wh), dim = -1L)

    hs <- self$decoder(tgt, memory, refpoints_dec, spatial_shapes, lvl_start, valid_ratios, mask_flat)
    pred_logits <- class_embed_fn(hs)
    pred_boxes <- bbox_embed_fn(hs)

    final_cxcy <- pred_boxes[,, 1:2] * refpoints_dec[,, 3:4] + refpoints_dec[,, 1:2]
    final_wh <- torch_exp(pred_boxes[,, 3:4]) * refpoints_dec[,, 3:4]
    final_boxes <- torch_cat(list(final_cxcy, final_wh), dim = -1L)

    list(logits = pred_logits, boxes = final_boxes)
  }
)


# Full LW-DETR model

lw_detr_model <- nn_module(
  "lw_detr",
  initialize = function(
    embed_dim,
    depth,
    num_heads,
    window_block_indexes,
    out_feature_indexes,
    scale_layers_list,
    d_model,
    sa_nhead,
    ca_nhead,
    num_queries,
    num_decoder_layers,
    dim_feedforward,
    n_levels,
    n_points,
    num_classes = 91L,
    num_select = 300L,
    group_detr = 13L
  ) {
    self$class_embed <- nn_linear(d_model, num_classes)
    self$bbox_embed <- detr_mlp_layer(d_model, d_model, 4L, 3L)

    self$model <- .lw_detr_inner_model(
      embed_dim,
      depth,
      num_heads,
      window_block_indexes,
      out_feature_indexes,
      scale_layers_list,
      d_model,
      sa_nhead,
      ca_nhead,
      num_queries,
      num_decoder_layers,
      dim_feedforward,
      n_levels,
      n_points,
      num_classes,
      group_detr
    )

    bias_value <- -log((1 - 0.01) / 0.01)
    with_no_grad({
      self$class_embed$bias$fill_(bias_value)
      for (g in seq_len(group_detr)) {
        self$model$enc_out_class_embed[[g]]$bias$fill_(bias_value)
      }
    })

    self$num_select <- num_select
    self$num_classes <- num_classes
    self$d_model <- d_model
  },
  forward = function(images, pixel_mask = NULL) {
    bs <- images$size(1L)
    img_h <- images$size(3L)
    img_w <- images$size(4L)
    if (is.null(pixel_mask)) {
      pixel_mask <- torch_ones(c(bs, img_h, img_w), dtype = torch_bool(), device = images$device)
    }

    out <- self$model(
      images,
      class_embed_fn = function(h) self$class_embed(h),
      bbox_embed_fn = function(h) self$bbox_embed(h),
      pixel_mask = pixel_mask
    )

    pred_logits <- out$logits
    pred_boxes <- out$boxes

    detections <- lapply(seq_len(bs), function(b) {
      valid_h <- pixel_mask[b, , 1]$sum()$item()
      valid_w <- pixel_mask[b, 1, ]$sum()$item()
      .lw_detr_postprocess(
        pred_logits[b],
        pred_boxes[b],
        valid_size = c(valid_h, valid_w),
        num_select = self$num_select
      )
    })
    list(detections = detections)
  }
)


# Post-processing

.lw_detr_postprocess <- function(logits, boxes, valid_size, num_select) {
  num_classes <- logits$size(2L)
  prob <- torch_sigmoid(logits)

  prob_flat <- prob$reshape(c(-1L))
  actual_k <- min(num_select, as.integer(prob_flat$numel()))
  topk_res <- torch_topk(prob_flat, actual_k, dim = 1L)
  scores <- topk_res[[1]]
  topk_idx <- topk_res[[2]]

  query_idx <- torch_div(topk_idx - 1L, num_classes, rounding_mode = "floor") + 1L
  class_idx <- (topk_idx - 1L) %% num_classes

  sel_boxes <- boxes[query_idx, ]
  h <- valid_size[1]
  w <- valid_size[2]
  boxes_xyxy <- box_cxcywh_to_xyxy(sel_boxes)
  scale <- torch_tensor(c(w, h, w, h), dtype = torch_float32(), device = boxes$device)$unsqueeze(1)
  boxes_xyxy <- (boxes_xyxy * scale)$clamp(min = 0)

  list(boxes = boxes_xyxy, labels = class_idx, scores = scores)
}


# Build scale layers from config

.lw_detr_build_scale_layers <- function(embed_dim, n_features, projector_scales, out_channels, n_blocks) {
  lapply(projector_scales, function(scale) {
    if (scale == 1.0) {
      total_in <- n_features * embed_dim
      .lw_detr_scale_layer(total_in, out_channels, n_blocks, sampling_ops = NULL)
    } else if (scale == 2.0) {
      out_per_feat <- embed_dim %/% 2L
      total_in <- n_features * out_per_feat
      ops <- lapply(seq_len(n_features), function(j) {
        nn_conv_transpose2d(embed_dim, out_per_feat, kernel_size = 2L, stride = 2L)
      })
      .lw_detr_scale_layer(total_in, out_channels, n_blocks, sampling_ops = ops)
    } else if (scale == 0.5) {
      total_in <- n_features * embed_dim
      ops <- lapply(seq_len(n_features), function(j) {
        .lw_detr_conv_x(embed_dim, embed_dim, 3L, stride = 2L)
      })
      .lw_detr_scale_layer(total_in, out_channels, n_blocks, sampling_ops = ops)
    } else {
      stop(paste("Unsupported projector scale:", scale))
    }
  })
}


# Model URLs and exported builder functions

.lw_detr_model_urls <- list(
  lw_detr_coco_tiny = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/lw_detr_coco_tiny.pth",
    NA_character_,
    "~46 MB"
  ),
  lw_detr_coco_small = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/lw_detr_coco_small.pth",
    NA_character_,
    "~56 MB"
  ),
  lw_detr_coco_medium = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/lw_detr_coco_medium.pth",
    NA_character_,
    "~108 MB"
  ),
  lw_detr_coco_large = c(
    "https://torch-cdn.mlverse.org/models/vision/v2/models/lw_detr_coco_large.pth",
    NA_character_,
    "~179 MB"
  )
)

.build_lw_detr <- function(
  embed_dim,
  depth,
  num_heads,
  window_block_indexes,
  out_feature_indexes,
  projector_scales,
  proj_out_channels,
  proj_num_blocks,
  d_model,
  sa_nhead,
  ca_nhead,
  num_queries,
  n_points,
  num_classes,
  num_select,
  pretrained,
  model_key
) {
  n_features <- length(out_feature_indexes)
  n_levels <- length(projector_scales)
  scale_layers <- .lw_detr_build_scale_layers(
    embed_dim,
    n_features,
    projector_scales,
    proj_out_channels,
    proj_num_blocks
  )

  model <- lw_detr_model(
    embed_dim = embed_dim,
    depth = depth,
    num_heads = num_heads,
    window_block_indexes = window_block_indexes,
    out_feature_indexes = out_feature_indexes,
    scale_layers_list = scale_layers,
    d_model = d_model,
    sa_nhead = sa_nhead,
    ca_nhead = ca_nhead,
    num_queries = num_queries,
    num_decoder_layers = 3L,
    dim_feedforward = 2048L,
    n_levels = n_levels,
    n_points = n_points,
    num_classes = num_classes,
    num_select = num_select,
    group_detr = 13L
  )

  if (pretrained) {
    if (num_classes != 91L) {
      cli::cli_abort("Pretrained weights require num_classes = 91 (COCO).")
    }

    r <- .lw_detr_model_urls[[model_key]]
    cli::cli_inform("Downloading LW-DETR weights ({r[3]})...")
    state_dict_path <- download_and_cache(r[1], prefix = "lw_detr")
    state_dict <- load_state_dict(state_dict_path)
    model$load_state_dict(state_dict, strict = FALSE)
  }

  model
}

#' LW-DETR Object Detection Models
#'
#' Construct LW-DETR model variants for real-time object detection.
#' LW-DETR is a lightweight Detection Transformer that combines a ViT encoder,
#' a C2f multi-scale projector, and a shallow DETR decoder with deformable
#' cross-attention.
#'
#' @param pretrained Logical. If TRUE, loads COCO pretrained weights.
#' @param progress Logical. Show progress bar during download (unused).
#' @param num_classes Integer. Number of object classes (default: 91 for COCO).
#' @param num_select Integer. Number of top-scoring detections to return per image.
#' @param ... Additional arguments (unused).
#' @return An `lw_detr` nn_module.
#'
#' @section Input Format:
#' The `forward` method is `model(images, pixel_mask = NULL)`, where `images` is
#' an ImageNet-normalised `torch_tensor` of shape `(batch_size, 3, H, W)` with
#' `H = W` divisible by 64 (recommended 640). Normalise with
#' `mean = c(0.485, 0.456, 0.406)`, `std = c(0.229, 0.224, 0.225)`.
#'
#' For non-square images, resize the longest side to 640 keeping the aspect
#' ratio, pad to 640×640, and pass a `pixel_mask` of shape `(batch_size, H, W)`
#' (logical, `TRUE` over real pixels and `FALSE` over padding). The padded region
#' is then excluded from attention and boxes are returned in the coordinates of
#' the unpadded image. This matches the reference preprocessing and gives the best
#' accuracy. When `pixel_mask` is omitted the whole frame is treated as valid,
#' which is only appropriate for square, unpadded inputs.
#'
#' @section Output Format:
#' Returns a list with element `detections`: a list (one per image) of
#' lists with:
#' \itemize{
#'   \item `boxes`  — tensor (k, 4) in xyxy pixel coordinates
#'   \item `labels` — integer tensor (k,) of COCO category ids (e.g. 17 = cat);
#'     pass to [coco_classes()] for names
#'   \item `scores` — float tensor (k,) — confidence scores
#' }
#'
#' @examples
#' \dontrun{
#' norm_mean <- c(0.485, 0.456, 0.406)
#' norm_std  <- c(0.229, 0.224, 0.225)
#'
#' # A non-square demo image from the LW-DETR repository
#' url <- "https://raw.githubusercontent.com/Atten4Vis/LW-DETR/main/demo/000000496954.jpg"
#' img <- base_loader(url) |> transform_to_tensor()
#' h <- img$shape[2]
#' w <- img$shape[3]
#'
#' # Letterbox the longest side to 640 and build the matching pixel mask
#' s  <- 640 / max(h, w)
#' nh <- round(h * s)
#' nw <- round(w * s)
#' resized <- img |> transform_resize(c(nh, nw))
#' canvas  <- torch::torch_zeros(c(3, 640, 640))
#' canvas[, 1:nh, 1:nw] <- resized
#' mask <- torch::torch_zeros(c(640, 640), dtype = torch::torch_bool())
#' mask[1:nh, 1:nw] <- TRUE
#'
#' input <- canvas |> transform_normalize(norm_mean, norm_std)
#'
#' model <- model_lw_detr_tiny(pretrained = TRUE)
#' model$eval()
#' pred <- torch::with_no_grad(
#'   model(input$unsqueeze(1), pixel_mask = mask$unsqueeze(1))
#' )$detections[[1]]
#'
#' # Draw the most confident detections on the letterboxed image
#' topk   <- pred$scores$topk(k = 5L)[[2]]
#' boxes  <- pred$boxes[topk, ]
#' labels <- coco_classes(as.integer(pred$labels[topk]))
#' boxed  <- draw_bounding_boxes(canvas, boxes, labels = labels)
#' tensor_image_browse(boxed)
#' }
#'
#' @references
#' Chen et al. (2024). LW-DETR: A Transformer Replacement to YOLO for
#' Real-Time Detection. \url{https://arxiv.org/abs/2406.03459}
#'
#' @family object_detection_model
#' @name model_lw_detr
#' @rdname model_lw_detr
NULL

#' @describeIn model_lw_detr LW-DETR tiny — ViT-tiny, 6 layers, 100 queries
#' @export
model_lw_detr_tiny <- function(pretrained = FALSE, progress = TRUE, num_classes = 91L, num_select = 100L, ...) {
  .build_lw_detr(
    embed_dim = 192L,
    depth = 6L,
    num_heads = 12L,
    window_block_indexes = c(0L, 2L, 4L),
    out_feature_indexes = c(1L, 3L, 5L),
    projector_scales = c(1.0),
    proj_out_channels = 256L,
    proj_num_blocks = 3L,
    d_model = 256L,
    sa_nhead = 8L,
    ca_nhead = 16L,
    num_queries = 100L,
    n_points = 2L,
    num_classes = num_classes,
    num_select = num_select,
    pretrained = pretrained,
    model_key = "lw_detr_coco_tiny"
  )
}

#' @describeIn model_lw_detr LW-DETR small — ViT-tiny, 10 layers, 300 queries
#' @export
model_lw_detr_small <- function(pretrained = FALSE, progress = TRUE, num_classes = 91L, num_select = 300L, ...) {
  .build_lw_detr(
    embed_dim = 192L,
    depth = 10L,
    num_heads = 12L,
    window_block_indexes = c(0L, 1L, 3L, 6L, 7L, 9L),
    out_feature_indexes = c(2L, 4L, 5L, 9L),
    projector_scales = c(1.0),
    proj_out_channels = 256L,
    proj_num_blocks = 3L,
    d_model = 256L,
    sa_nhead = 8L,
    ca_nhead = 16L,
    num_queries = 300L,
    n_points = 2L,
    num_classes = num_classes,
    num_select = num_select,
    pretrained = pretrained,
    model_key = "lw_detr_coco_small"
  )
}

#' @describeIn model_lw_detr LW-DETR medium — ViT-small, 10 layers, 300 queries
#' @export
model_lw_detr_medium <- function(pretrained = FALSE, progress = TRUE, num_classes = 91L, num_select = 300L, ...) {
  .build_lw_detr(
    embed_dim = 384L,
    depth = 10L,
    num_heads = 12L,
    window_block_indexes = c(0L, 1L, 3L, 6L, 7L, 9L),
    out_feature_indexes = c(2L, 4L, 5L, 9L),
    projector_scales = c(1.0),
    proj_out_channels = 256L,
    proj_num_blocks = 3L,
    d_model = 256L,
    sa_nhead = 8L,
    ca_nhead = 16L,
    num_queries = 300L,
    n_points = 2L,
    num_classes = num_classes,
    num_select = num_select,
    pretrained = pretrained,
    model_key = "lw_detr_coco_medium"
  )
}

#' @describeIn model_lw_detr LW-DETR large — ViT-small, 10 layers, 2-scale, 300 queries
#' @export
model_lw_detr_large <- function(pretrained = FALSE, progress = TRUE, num_classes = 91L, num_select = 300L, ...) {
  .build_lw_detr(
    embed_dim = 384L,
    depth = 10L,
    num_heads = 12L,
    window_block_indexes = c(0L, 1L, 3L, 6L, 7L, 9L),
    out_feature_indexes = c(2L, 4L, 5L, 9L),
    projector_scales = c(2.0, 0.5),
    proj_out_channels = 384L,
    proj_num_blocks = 3L,
    d_model = 384L,
    sa_nhead = 12L,
    ca_nhead = 24L,
    num_queries = 300L,
    n_points = 4L,
    num_classes = num_classes,
    num_select = num_select,
    pretrained = pretrained,
    model_key = "lw_detr_coco_large"
  )
}
