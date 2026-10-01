# R/models-maxvit.R - port of the MaxViT implementation of PyTorch's torchvision
# (torchvision/models/maxvit.py). Module names follow the reference, so that its
# weights can be loaded without renaming.

#' @importFrom torch nn_module nn_sequential nn_conv2d nn_batch_norm2d nn_gelu nn_identity
#' @importFrom torch nn_adaptive_avg_pool2d nn_avg_pool2d nn_linear nn_layer_norm nn_dropout
#' @importFrom torch nn_module_list nn_module_dict nn_flatten nn_tanh nn_parameter nn_buffer nnf_silu nnf_softmax
#' @importFrom torch torch_empty torch_sigmoid torch_stack torch_meshgrid torch_arange torch_long
#' @importFrom torch torch_flatten torch_chunk torch_matmul load_state_dict
#' @importFrom torch nn_init_normal_ nn_init_zeros_ nn_init_constant_ nn_init_trunc_normal_
#' @noRd
maxvit_bn <- function(num_features) {
  nn_batch_norm2d(num_features, eps = 1e-3, momentum = 0.01)
}

# conv -> batch norm -> GELU, as torchvision's Conv2dNormActivation
maxvit_conv_bn_act <- function(in_channels, out_channels, kernel_size, stride = 1,
                               padding = (kernel_size - 1) %/% 2, groups = 1) {
  nn_sequential(
    "0" = nn_conv2d(in_channels, out_channels, kernel_size, stride = stride,
      padding = padding, groups = groups, bias = FALSE),
    "1" = maxvit_bn(out_channels),
    "2" = nn_gelu()
  )
}

maxvit_stochastic_depth <- nn_module(
  "maxvit_stochastic_depth",
  initialize = function(p) {
    self$p <- p
  },
  forward = function(x) {
    if (!self$training || self$p == 0) {
      return(x)
    }
    survival_rate <- 1 - self$p
    size <- c(x$size(1), rep(1L, x$dim() - 1L))
    noise <- torch_empty(size, dtype = x$dtype, device = x$device)$bernoulli_(survival_rate)
    if (survival_rate > 0) {
      noise <- noise / survival_rate
    }
    x * noise
  }
)

maxvit_squeeze_excitation <- nn_module(
  "maxvit_squeeze_excitation",
  initialize = function(in_channels, squeeze_channels) {
    self$avgpool <- nn_adaptive_avg_pool2d(1)
    self$fc1 <- nn_conv2d(in_channels, squeeze_channels, 1)
    self$fc2 <- nn_conv2d(squeeze_channels, in_channels, 1)
  },
  forward = function(x) {
    scale <- self$avgpool(x)
    scale <- self$fc1(scale)
    scale <- nnf_silu(scale)
    scale <- self$fc2(scale)
    x * torch_sigmoid(scale)
  }
)

maxvit_mbconv <- nn_module(
  "maxvit_mbconv",
  initialize = function(in_channels, out_channels, expansion_ratio, squeeze_ratio, stride,
                        p_stochastic_dropout = 0) {
    if (stride != 1 || in_channels != out_channels) {
      proj <- list(nn_conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = TRUE))
      if (stride == 2) {
        proj <- c(list(nn_avg_pool2d(kernel_size = 3, stride = stride, padding = 1)), proj)
      }
      names(proj) <- as.character(seq_along(proj) - 1L)
      self$proj <- do.call(nn_sequential, proj)
    } else {
      self$proj <- nn_identity()
    }
    mid_channels <- as.integer(out_channels * expansion_ratio)
    sqz_channels <- as.integer(out_channels * squeeze_ratio)
    self$stochastic_depth <- if (p_stochastic_dropout > 0) {
      maxvit_stochastic_depth(p_stochastic_dropout)
    } else {
      nn_identity()
    }
    self$layers <- nn_module_dict(list(
      pre_norm = maxvit_bn(in_channels),
      conv_a = maxvit_conv_bn_act(in_channels, mid_channels, kernel_size = 1, padding = 0),
      conv_b = maxvit_conv_bn_act(mid_channels, mid_channels, kernel_size = 3, stride = stride,
        padding = 1, groups = mid_channels),
      squeeze_excitation = maxvit_squeeze_excitation(mid_channels, sqz_channels),
      conv_c = nn_conv2d(mid_channels, out_channels, kernel_size = 1, bias = TRUE)
    ))
  },
  forward = function(x) {
    out <- x
    for (nm in c("pre_norm", "conv_a", "conv_b", "squeeze_excitation", "conv_c")) {
      out <- self$layers[[nm]](out)
    }
    self$proj(x) + self$stochastic_depth(out)
  }
)

# 0-based index into the relative position bias table, as in the reference
maxvit_relative_position_index <- function(height, width) {
  coords <- torch_stack(torch_meshgrid(list(torch_arange(0, height - 1, dtype = torch_long()),
    torch_arange(0, width - 1, dtype = torch_long())), indexing = "ij"))
  coords_flat <- torch_flatten(coords, start_dim = 2)
  relative_coords <- coords_flat[, , NULL] - coords_flat[, NULL, ]
  relative_coords <- relative_coords$permute(c(2, 3, 1))$contiguous()
  relative_coords[, , 1] <- relative_coords[, , 1] + (height - 1)
  relative_coords[, , 2] <- relative_coords[, , 2] + (width - 1)
  relative_coords[, , 1] <- relative_coords[, , 1] * (2 * width - 1)
  relative_coords$sum(dim = -1)
}

maxvit_relative_attention <- nn_module(
  "maxvit_relative_attention",
  initialize = function(feat_dim, head_dim, max_seq_len) {
    if (feat_dim %% head_dim != 0) {
      value_error("feat_dim ({feat_dim}) must be divisible by head_dim ({head_dim}).")
    }
    self$n_heads <- feat_dim %/% head_dim
    self$head_dim <- head_dim
    self$size <- as.integer(sqrt(max_seq_len))
    self$max_seq_len <- max_seq_len
    self$to_qkv <- nn_linear(feat_dim, self$n_heads * self$head_dim * 3)
    # the reference scales by the feature dimension, not by the head dimension
    self$scale_factor <- feat_dim^-0.5
    self$merge <- nn_linear(self$head_dim * self$n_heads, feat_dim)
    self$relative_position_bias_table <- nn_parameter(
      torch_empty((2 * self$size - 1) * (2 * self$size - 1), self$n_heads)
    )
    self$relative_position_index <- nn_buffer(maxvit_relative_position_index(self$size, self$size))
    nn_init_trunc_normal_(self$relative_position_bias_table, std = 0.02)
  },
  get_relative_positional_bias = function() {
    bias_index <- self$relative_position_index$view(-1) + 1L
    relative_bias <- self$relative_position_bias_table[bias_index, ]
    relative_bias <- relative_bias$view(c(self$max_seq_len, self$max_seq_len, -1))
    relative_bias$permute(c(3, 1, 2))$contiguous()$unsqueeze(1)
  },
  forward = function(x) {
    # x: (batch, partitions, tokens, features)
    c(B, G, P, D) %<-% x$shape
    H <- self$n_heads
    DH <- self$head_dim
    qkv <- torch_chunk(self$to_qkv(x), 3, dim = -1)
    q <- qkv[[1]]$reshape(c(B, G, P, H, DH))$permute(c(1, 2, 4, 3, 5))
    k <- qkv[[2]]$reshape(c(B, G, P, H, DH))$permute(c(1, 2, 4, 3, 5))
    v <- qkv[[3]]$reshape(c(B, G, P, H, DH))$permute(c(1, 2, 4, 3, 5))
    k <- k * self$scale_factor
    dot_prod <- torch_matmul(q, k$transpose(-2, -1))
    dot_prod <- nnf_softmax(dot_prod + self$get_relative_positional_bias(), dim = -1)
    out <- torch_matmul(dot_prod, v)
    out <- out$permute(c(1, 2, 4, 3, 5))$reshape(c(B, G, P, D))
    self$merge(out)
  }
)

# (B, C, H, W) -> (B, H/p * W/p, p * p, C)
maxvit_window_partition <- function(x, p) {
  c(B, C, H, W) %<-% x$shape
  x <- x$reshape(c(B, C, H %/% p, p, W %/% p, p))
  x <- x$permute(c(1, 3, 5, 4, 6, 2))
  x$reshape(c(B, (H %/% p) * (W %/% p), p * p, C))
}

# (B, hp * wp, p * p, C) -> (B, C, hp * p, wp * p)
maxvit_window_departition <- function(x, p, hp, wp) {
  B <- x$size(1)
  C <- x$size(4)
  x <- x$reshape(c(B, hp, wp, p, p, C))
  x <- x$permute(c(1, 6, 2, 4, 3, 5))
  x$reshape(c(B, C, hp * p, wp * p))
}

maxvit_partition_attention <- nn_module(
  "maxvit_partition_attention",
  initialize = function(in_channels, head_dim, partition_size, partition_type, mlp_ratio,
                        attention_dropout, mlp_dropout, p_stochastic_dropout) {
    if (!partition_type %in% c("grid", "window")) {
      value_error("partition_type must be either 'grid' or 'window'.")
    }
    self$partition_size <- partition_size
    self$partition_type <- partition_type
    self$attn_layer <- nn_sequential(
      "0" = nn_layer_norm(in_channels),
      "1" = maxvit_relative_attention(in_channels, head_dim, partition_size^2),
      "2" = nn_dropout(attention_dropout)
    )
    self$mlp_layer <- nn_sequential(
      "0" = nn_layer_norm(in_channels),
      "1" = nn_linear(in_channels, in_channels * mlp_ratio),
      "2" = nn_gelu(),
      "3" = nn_linear(in_channels * mlp_ratio, in_channels),
      "4" = nn_dropout(mlp_dropout)
    )
    self$stochastic_dropout <- maxvit_stochastic_depth(p_stochastic_dropout)
  },
  forward = function(x) {
    c(H, W) %<-% x$shape[3:4]
    # window attention attends within partition_size x partition_size windows, grid attention
    # within a sparse partition_size x partition_size grid that spans the whole feature map
    p <- if (self$partition_type == "window") self$partition_size else H %/% self$partition_size
    if (H %% p != 0 || W %% p != 0) {
      value_error("The feature map ({H}x{W}) must be divisible by the partition size ({p}).")
    }
    gh <- H %/% p
    gw <- W %/% p
    x <- maxvit_window_partition(x, p)
    if (self$partition_type == "grid") x <- x$transpose(2, 3)
    x <- x + self$stochastic_dropout(self$attn_layer(x))
    x <- x + self$stochastic_dropout(self$mlp_layer(x))
    if (self$partition_type == "grid") x <- x$transpose(2, 3)
    maxvit_window_departition(x, p, gh, gw)
  }
)

maxvit_layer <- nn_module(
  "maxvit_layer",
  initialize = function(in_channels, out_channels, squeeze_ratio, expansion_ratio, stride,
                        head_dim, mlp_ratio, mlp_dropout, attention_dropout,
                        p_stochastic_dropout, partition_size) {
    self$layers <- nn_module_dict(list(
      MBconv = maxvit_mbconv(in_channels, out_channels, expansion_ratio, squeeze_ratio,
        stride, p_stochastic_dropout),
      window_attention = maxvit_partition_attention(out_channels, head_dim, partition_size,
        "window", mlp_ratio, attention_dropout, mlp_dropout, p_stochastic_dropout),
      grid_attention = maxvit_partition_attention(out_channels, head_dim, partition_size,
        "grid", mlp_ratio, attention_dropout, mlp_dropout, p_stochastic_dropout)
    ))
  },
  forward = function(x) {
    x <- self$layers$MBconv(x)
    x <- self$layers$window_attention(x)
    self$layers$grid_attention(x)
  }
)

maxvit_block <- nn_module(
  "maxvit_block",
  initialize = function(in_channels, out_channels, squeeze_ratio, expansion_ratio, head_dim,
                        mlp_ratio, mlp_dropout, attention_dropout, partition_size, p_stochastic) {
    self$layers <- nn_module_list(lapply(seq_along(p_stochastic), function(i) {
      maxvit_layer(
        in_channels = if (i == 1) in_channels else out_channels,
        out_channels = out_channels,
        squeeze_ratio = squeeze_ratio,
        expansion_ratio = expansion_ratio,
        stride = if (i == 1) 2 else 1,
        head_dim = head_dim,
        mlp_ratio = mlp_ratio,
        mlp_dropout = mlp_dropout,
        attention_dropout = attention_dropout,
        p_stochastic_dropout = p_stochastic[[i]],
        partition_size = partition_size
      )
    }))
  },
  forward = function(x) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x)
    }
    x
  }
)

maxvit_impl <- nn_module(
  "maxvit",
  initialize = function(num_classes = 1000, stem_channels = 64,
                        block_channels = c(64, 128, 256, 512), block_layers = c(2, 2, 5, 2),
                        head_dim = 32, stochastic_depth_prob = 0.2, partition_size = 7,
                        squeeze_ratio = 0.25, expansion_ratio = 4, mlp_ratio = 4,
                        mlp_dropout = 0, attention_dropout = 0) {
    self$stem <- nn_sequential(
      "0" = maxvit_conv_bn_act(3, stem_channels, kernel_size = 3, stride = 2),
      "1" = nn_sequential(
        "0" = nn_conv2d(stem_channels, stem_channels, 3, stride = 1, padding = 1, bias = TRUE)
      )
    )
    in_channels <- c(stem_channels, block_channels[-length(block_channels)])
    p_stochastic <- seq(0, stochastic_depth_prob, length.out = sum(block_layers))
    ends <- cumsum(block_layers)
    self$blocks <- nn_module_list(lapply(seq_along(block_channels), function(i) {
      maxvit_block(
        in_channels = in_channels[[i]],
        out_channels = block_channels[[i]],
        squeeze_ratio = squeeze_ratio,
        expansion_ratio = expansion_ratio,
        head_dim = head_dim,
        mlp_ratio = mlp_ratio,
        mlp_dropout = mlp_dropout,
        attention_dropout = attention_dropout,
        partition_size = partition_size,
        p_stochastic = p_stochastic[(ends[[i]] - block_layers[[i]] + 1):ends[[i]]]
      )
    }))
    d <- block_channels[[length(block_channels)]]
    self$classifier <- nn_sequential(
      "0" = nn_adaptive_avg_pool2d(1),
      "1" = nn_flatten(),
      "2" = nn_layer_norm(d),
      "3" = nn_linear(d, d),
      "4" = nn_tanh(),
      "5" = nn_linear(d, num_classes, bias = FALSE)
    )
    self$init_weights()
  },
  init_weights = function() {
    for (m in self$modules) {
      if (inherits(m, "nn_conv2d") || inherits(m, "nn_linear")) {
        nn_init_normal_(m$weight, std = 0.02)
        if (!is.null(m$bias)) nn_init_zeros_(m$bias)
      } else if (inherits(m, "nn_batch_norm2d")) {
        nn_init_constant_(m$weight, 1)
        nn_init_constant_(m$bias, 0)
      }
    }
  },
  forward = function(x) {
    x <- self$stem(x)
    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](x)
    }
    self$classifier(x)
  }
)

#' MaxViT Model
#'
#' Implementation of the MaxViT architecture described in
#' [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697).
#' The model performs image classification and by default returns logits for
#' 1000 ImageNet classes.
#'
#' @inheritParams model_resnet18
#' @param num_classes (integer) Number of output classes.
#' @param ... Additional parameters passed to the model initializer.
#'
#' @family classification_model
#'
#' @examples
#' \dontrun{
#' library(magrittr)
#' # 1. Load the basketball image
#' img_url <- "https://upload.wikimedia.org/wikipedia/commons/7/7a/Basketball.png"
#' img <- base_loader(img_url)
#'
#' # 2. Define normalization (ImageNet)
#' norm_mean <- c(0.485, 0.456, 0.406)
#' norm_std <- c(0.229, 0.224, 0.225)
#'
#' # 3. Preprocess: convert to tensor, resize, Normalize
#' input <- img %>%
#'   transform_to_tensor() %>%
#'   transform_resize(c(400, 400)) %>%
#'   transform_normalize(norm_mean, norm_std)
#' batch <- input$unsqueeze(1)    # Add batch dimension (1, 3, H, W)
#'
#' # 4. Display the image before normalization
#' tensor_image_browse(input)
#'
#' # 5. Load MaxViT model
#' model <- model_maxvit(pretrained = TRUE)
#' model$eval()
#'
#' # 6. Run inference
#' output <- model(batch)
#' topk <- output$topk(k = 5, dim = 2)
#' indices <- as.integer(topk[[2]][1, ])
#' scores <- as.numeric(topk[[1]][1, ])
#'
#' # 7. Show Top-5 predictions
#' glue::glue("{seq_along(indices)}. {imagenet_classes(indices)} ({round(scores, 2)}%)")
#' }
#'
#' @export
model_maxvit <- function(pretrained = FALSE, progress = TRUE, num_classes = 1000, ...) {
  if (!pretrained) {
    return(maxvit_impl(num_classes = num_classes, ...))
  }
  model <- maxvit_impl(num_classes = 1000, ...)
  path <- download_and_cache("https://torch-cdn.mlverse.org/models/vision/v2/models/maxvit.pth")
  model$load_state_dict(torch::load_state_dict(path))
  if (num_classes != 1000) {
    model$classifier$`5` <- nn_linear(model$classifier$`5`$in_features, num_classes, bias = FALSE)
  }
  model
}
