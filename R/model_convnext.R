#' ConvNeXt Implementation
#'
#' Implements the ConvNeXt architecture from [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545)
#' @inheritParams model_resnet18
#' @param ... Other parameters passed to the model implementation.
#'
#' @family classification_model
#' @name model_resnext
NULL

#' @importFrom torch nn_module nn_parameter torch_ones torch_zeros nnf_layer_norm
LayerNorm <- nn_module(
  "LayerNorm",
  initialize = function(normalized_shape,
                        eps = 1e-6,
                        data_format = "channels_last") {
    self$weight <- nn_parameter(torch_ones(normalized_shape))
    self$bias <- nn_parameter(torch_zeros(normalized_shape))
    self$eps <- eps
    self$data_format <- data_format
    self$normalized_shape <- normalized_shape
  },
  forward = function(x) {
    if (self$data_format == "channels_last") {
      nnf_layer_norm(x, self$normalized_shape, self$weight, self$bias, self$eps)
    } else if (self$data_format == "channels_first") {
      u <- x$mean(dim = 2, keepdim = TRUE)
      s <- ((x - u)$pow(2))$mean(dim = 2, keepdim = TRUE)
      x <- (x - u) / (s + self$eps)$sqrt()
      x <- self$weight$unsqueeze(2)$unsqueeze(3) * x + self$bias$unsqueeze(2)$unsqueeze(3)
      x
    } else {
      stop("Unsupported data format")
    }
  }
)

#' @importFrom torch nn_conv2d nn_linear torch_ones nn_gelu nn_identity
Block <- nn_module(
  "Block",
  initialize = function(dim,
                        drop_path = 0,
                        layer_scale_init_value = 1e-6) {
    self$dwconv <- nn_conv2d(dim, dim, kernel_size = 7, padding = 3, groups = dim)
    self$norm <- LayerNorm(dim, eps = 1e-6)
    self$pwconv1 <- nn_linear(dim, 4 * dim)
    self$act <- nn_gelu()
    self$pwconv2 <- nn_linear(4 * dim, dim)
    self$gamma <- if (layer_scale_init_value > 0) {
      nn_parameter(layer_scale_init_value * torch_ones(dim))
    } else {
      NULL
    }
    self$drop_path <- nn_identity()
  },
  forward = function(x) {
    input <- x
    x <- self$dwconv(x)
    x <- x$permute(c(1, 3, 4, 2))
    x <- self$norm(x)
    x <- self$pwconv1(x)
    x <- self$act(x)
    x <- self$pwconv2(x)
    if (!is.null(self$gamma)) {
      x <- self$gamma * x
    }
    x <- x$permute(c(1, 4, 2, 3))
    x <- input + self$drop_path(x)
    x
  }
)


#' @importFrom torch nn_conv2d nn_linear nn_sequential nn_module_list torch_linspace nn_layer_norm
ConvNeXt <- nn_module(
  "ConvNeXt",
  initialize = function(in_chans = 3,
                        num_classes = 1000,
                        depths = c(3, 3, 9, 3),
                        dims = c(96, 192, 384, 768),
                        drop_path_rate = 0.,
                        layer_scale_init_value = 1e-6,
                        head_init_scale = 1.) {
    self$downsample_layers <- nn_module_list()

    # Stem
    stem <- nn_sequential(
      nn_conv2d(in_chans, dims[1], kernel_size = 4, stride = 4),
      LayerNorm(dims[1], eps = 1e-6, data_format = "channels_first")
    )
    self$downsample_layers$append(stem)

    for (i in 1:3) {
      self$downsample_layers$append(nn_sequential(
        LayerNorm(dims[i], eps = 1e-6, data_format = "channels_first"),
        nn_conv2d(dims[i], dims[i + 1], kernel_size = 2, stride = 2)
      ))
    }

    self$stages <- nn_module_list()
    dp_rates <- as.numeric(torch_linspace(0, drop_path_rate, sum(depths)))
    cur <- 1
    for (i in 1:4) {
      blocks <- lapply(1:depths[i], function(j) {
        Block(dims[i],
              drop_path = dp_rates[cur],
              layer_scale_init_value = layer_scale_init_value)
      })
      self$stages$append(nn_sequential(!!!blocks))
      cur <- cur + depths[i]
    }

    self$norm <- nn_layer_norm(dims[4], eps = 1e-6)
    self$head <- nn_linear(dims[4], num_classes)
  },

  forward_features = function(x) {
    for (i in 1:4) {
      x <- self$downsample_layers[[i]](x)
      x <- self$stages[[i]](x)
    }
    x$mean(dim = c(3, 4)) %>% self$norm()
  },

  forward = function(x) {
    x <- self$forward_features(x)
    self$head(x)
  }
)


convnext_model_urls <- c(
  "convnext_tiny_1k" = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_tiny_1k.pth",
  "convnext_tiny_22k" = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_tiny_22k.pth",
  "convnext_small_22k" = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_small_22k.pth",
  'convnext_small_22k1k' = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_small_22k1k.pth",
  'convnext_base_1k' = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_base_1k.pth",
  'convnext_base_22k' = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_base_22k.pth",
  'convnext_large_1k' = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_large_1k.pth",
  'convnext_large_22k' = "https://torch-cdn.mlverse.org/models/vision/v2/models/convnext_large_22k.pth"
)



.convnext <- function(arch, channels, depths, dims, num_classes, pretrained, progress, ...) {
  if (!is.character(arch) || length(arch) != 1) {
    stop("arch must be a single character string.")
  }
  depths <- as.integer(depths)
  dims <- as.integer(dims)
  channels <- as.integer(channels)
  num_classes <- as.integer(num_classes)

  if (length(depths) != 4 || length(dims) != 4) {
    stop("depths and dims must be vectors of length 4.")
  }
  model <- ConvNeXt(
    in_chans = channels,
    depths = depths,
    dims = dims,
    num_classes = num_classes
  )

  if (pretrained) {
    if (!arch %in% names(convnext_model_urls)) {
      stop(paste("Pretrained model for", arch, "is not available."))
    }
    state_dict_path <- download_and_cache(convnext_model_urls[arch], prefix = "convnext")
    state_dict <- torch::load_state_dict(state_dict_path)
    new_names <- names(state_dict)
    new_names <- gsub("^features\\.([0-9])\\.([0-1])\\.w", "downsample_layers.\\1.\\2.w", new_names)
    new_names <- gsub("^features\\.([0-9])\\.([0-1])\\.bias", "downsample_layers.\\1.\\2.bias", new_names)
    names(state_dict) <- new_names

    # Interpolate stem weights if input channels differ.sample use cases - satellite images
    conv1_weight <- state_dict[["downsample_layers.0.0.weight"]]

    if (dim(conv1_weight)[2] != channels) {
      old_in_channels <- dim(conv1_weight)[2]
      mean_weight <- conv1_weight$mean(dim = 2, keepdim = TRUE)

      # Repeat manually using torch_cat
      new_weight_list <- rep(list(mean_weight), channels)
      new_weight <- torch_cat(new_weight_list, dim = 2)

      new_weight <- new_weight * (old_in_channels / channels)
      state_dict[["downsample_layers.0.0.weight"]] <- new_weight
    }
    model$load_state_dict(state_dict, strict = FALSE)
  }
  model
}


#' @describeIn model_convnext ConvNeXt Tiny model with 28 M parameters
#' 224 trained on Imagenet 1k.
#' @export
model_convnext_tiny <- function(pretrained = FALSE,
                                progress = TRUE,
                                channels = 3,
                                num_classes = 1000,
                                ...) {
  .convnext(
    arch = "convnext_tiny_224_1k" ,
    channels = channels,
    depths = c(3, 3, 9, 3),
    dims = c(96, 192, 384, 768),
    num_classes = num_classes,
    pretrained,
    progress,
    ...
  )
}


# #' @describeIn model_convnext ConvNeXt Tiny model.
## ' @export
# model_convnextTiny_224_22k <- function(pretrained = FALSE,
#                                        progress = TRUE,
#                                        channels = 3,
#                                        ...) {
#   .convnext(
#     "convnext_tiny_224_22k",
#     channels = channels,
#     depths = c(3, 3, 9, 3),
#     dims = c(96, 192, 384, 768),
#     num_classes = 21841,
#     pretrained,
#     progress,
#     ...
#   )
# }


#' @describeIn model_convnext ConvNeXt Small model 224 22k_1k.
#' @export
model_convnext_small <- function(pretrained = FALSE,
                                 progress = TRUE,
                                 channels = 3,
                                 num_classes = 1000,
                                 ...) {
  .convnext(
    arch = "convnextsmall_224_22k_1k" ,
    channels = channels,
    depths = c(3, 3, 27, 3),
    dims = c(96, 192, 384, 768),
    num_classes = num_classes,
    pretrained,
    progress,
    ...
  )
}


# #' @describeIn model_convnext ConvNeXt Small model.
# #' @export
# model_convnextSmall_224_22k <- function(pretrained = FALSE,
#                                         progress = TRUE,
#                                         channels = 3,
#                                         ...) {
#   .convnext(
#     "convnextsmall_224_22k",
#     channels = channels,
#     depths = c(3, 3, 27, 3),
#     dims = c(96, 192, 384, 768),
#     num_classes = 21841,
#     pretrained,
#     progress,
#     ...
#   )
# }
