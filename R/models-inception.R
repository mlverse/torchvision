Inception3 <- torch::nn_module(
  "Inception3",
  initialize = function(num_classes = 1000,
                        aux_logits = TRUE,
                        transform_input = FALSE,
                        inception_blocks = NULL,
                        init_weights = NULL,
                        dropout = 0.5) {
    if (is.null(inception_blocks))
      inception_blocks <- list(
        BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux
      )

    if (is.null(init_weights)) {
      cli::cli_warn(paste(collapse = "", gettext(c(
        "The default weight initialization of inception_v3 will be changed in future releases of ",
        "torchvision. If you wish to keep the old behavior (which leads to long initialization times",
        " due to scipy/scipy#11299), please set init_weights={.val TRUE}."
      ))))
      init_weights <- TRUE
    }

    if (length(inception_blocks) != 7) {
      cli::cli_abort(gettext("length of {.arg inception_blocks} should be 7 instead of {length(inception_blocks)}"))
    }

    conv_block <- inception_blocks[[0+1]]
    inception_a <- inception_blocks[[1+1]]
    inception_b <- inception_blocks[[2+1]]
    inception_c <- inception_blocks[[3+1]]
    inception_d <- inception_blocks[[4+1]]
    inception_e <- inception_blocks[[5+1]]
    inception_aux <- inception_blocks[[6+1]]

    self$aux_logits <- aux_logits
    self$transform_input <- transform_input
    self$Conv2d_1a_3x3 <- conv_block(3, 32, kernel_size=3, stride=2)
    self$Conv2d_2a_3x3 <- conv_block(32, 32, kernel_size=3)
    self$Conv2d_2b_3x3 <- conv_block(32, 64, kernel_size=3, padding=1)
    self$maxpool1 <- torch::nn_max_pool2d(kernel_size=3, stride=2)
    self$Conv2d_3b_1x1 <- conv_block(64, 80, kernel_size=1)
    self$Conv2d_4a_3x3 <- conv_block(80, 192, kernel_size=3)
    self$maxpool2 <- torch::nn_max_pool2d(kernel_size=3, stride=2)
    self$Mixed_5b <- inception_a(192, pool_features=32)
    self$Mixed_5c <- inception_a(256, pool_features=64)
    self$Mixed_5d <- inception_a(288, pool_features=64)
    self$Mixed_6a <- inception_b(288)
    self$Mixed_6b <- inception_c(768, channels_7x7=128)
    self$Mixed_6c <- inception_c(768, channels_7x7=160)
    self$Mixed_6d <- inception_c(768, channels_7x7=160)
    self$Mixed_6e <- inception_c(768, channels_7x7=192)

    self$AuxLogits <- NULL

    if (aux_logits)
      self$AuxLogits <- inception_aux(768, num_classes)

    self$Mixed_7a <- inception_d(768)
    self$Mixed_7b <- inception_e(1280)
    self$Mixed_7c <- inception_e(2048)
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(1, 1))
    self$dropout <- torch::nn_dropout(p=dropout)
    self$fc <- torch::nn_linear(2048, num_classes)

    if (init_weights) {
      for (m in self$modules) {
        if (inherits(m, "nn_conv2d") || inherits(m, "nn_linear")) {
          stddev <- if (!is.null(m$stddev)) m$stddev else 0.1
          torch::nn_init_trunc_normal_(m$weight, mean = 0, std = stddev, a = -2, b = -2)
        } else if (inherits(m, "nn_batch_norm2d")) {
          torch::nn_init_constant_(m$weight, 1)
          torch::nn_init_constant_(m$bias, 0)
        }
      }
    }

  },
  .transform_input = function(x) {
    if (self$transform_input) {
      x_ch0 <- torch::torch_unsqueeze(x[,1], 2) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
      x_ch1 <- torch::torch_unsqueeze(x[,2], 2) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
      x_ch2 <- torch::torch_unsqueeze(x[,3], 2) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
      x <- torch_cat(list(x_ch0, x_ch1, x_ch2), 2)
    }
    x
  },
  .forward = function(x) {
    # N x 3 x 299 x 299
    x <- self$Conv2d_1a_3x3(x)
    # N x 32 x 149 x 149
    x <- self$Conv2d_2a_3x3(x)
    # N x 32 x 147 x 147
    x <- self$Conv2d_2b_3x3(x)
    # N x 64 x 147 x 147
    x <- self$maxpool1(x)
    # N x 64 x 73 x 73
    x <- self$Conv2d_3b_1x1(x)
    # N x 80 x 73 x 73
    x <- self$Conv2d_4a_3x3(x)
    # N x 192 x 71 x 71
    x <- self$maxpool2(x)
    # N x 192 x 35 x 35
    x <- self$Mixed_5b(x)
    # N x 256 x 35 x 35
    x <- self$Mixed_5c(x)
    # N x 288 x 35 x 35
    x <- self$Mixed_5d(x)
    # N x 288 x 35 x 35
    x <- self$Mixed_6a(x)
    # N x 768 x 17 x 17
    x <- self$Mixed_6b(x)
    # N x 768 x 17 x 17
    x <- self$Mixed_6c(x)
    # N x 768 x 17 x 17
    x <- self$Mixed_6d(x)
    # N x 768 x 17 x 17
    x <- self$Mixed_6e(x)
    # N x 768 x 17 x 17
    if (!is.null(self$AuxLogits) && self$training) {
      aux <- self$AuxLogits(x)
    } else {
      aux <- NULL
    }
    # N x 768 x 17 x 17
    x <- self$Mixed_7a(x)
    # N x 1280 x 8 x 8
    x <- self$Mixed_7b(x)
    # N x 2048 x 8 x 8
    x <- self$Mixed_7c(x)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x <- self$avgpool(x)
    # N x 2048 x 1 x 1
    x <- self$dropout(x)
    # N x 2048 x 1 x 1
    x <- torch::torch_flatten(x, start_dim = 2)
    # N x 2048
    x <- self$fc(x)
    # N x 1000 (num_classes)

    list(logits = x, aux_logits = aux)
  },
  forward = function(x) {
    x <- self$.transform_input(x)
    out <- self$.forward(x)

    if (self$training && self$aux_logits) {
      out
    } else {
      out$logits
    }
  }
)

InceptionA <- torch::nn_module(
  "InceptionA",
  initialize = function(in_channels, pool_features, conv_block = NULL) {
    if (is.null(conv_block)) {
      conv_block <- BasicConv2d
    }

    self$branch1x1 <- conv_block(in_channels, 64, kernel_size=1)
    self$branch5x5_1 <- conv_block(in_channels, 48, kernel_size=1)
    self$branch5x5_2 <- conv_block(48, 64, kernel_size=5, padding=2)
    self$branch3x3dbl_1 <- conv_block(in_channels, 64, kernel_size=1)
    self$branch3x3dbl_2 <- conv_block(64, 96, kernel_size=3, padding=1)
    self$branch3x3dbl_3 <- conv_block(96, 96, kernel_size=3, padding=1)
    self$branch_pool <- conv_block(in_channels, pool_features, kernel_size=1)
  },
  .forward = function(x) {
    branch1x1 <- self$branch1x1(x)

    branch5x5 <- self$branch5x5_1(x)
    branch5x5 <- self$branch5x5_2(branch5x5)

    branch3x3dbl <- self$branch3x3dbl_1(x)
    branch3x3dbl <- self$branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl <- self$branch3x3dbl_3(branch3x3dbl)

    branch_pool <- torch::nnf_avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool <- self$branch_pool(branch_pool)

    outputs <- list(branch1x1, branch5x5, branch3x3dbl, branch_pool)
    outputs
  },
  forward = function(x) {
    outputs <- self$.forward(x)
    torch::torch_cat(outputs, 2)
  }
)

InceptionB <- torch::nn_module(
  "InceptionB",
  initialize = function(in_channels, conv_block = NULL) {
    if (is.null(conv_block)) {
      conv_block <- BasicConv2d
    }

    self$branch3x3 <- conv_block(in_channels, 384, kernel_size=3, stride=2)
    self$branch3x3dbl_1 <- conv_block(in_channels, 64, kernel_size=1)
    self$branch3x3dbl_2 <- conv_block(64, 96, kernel_size=3, padding=1)
    self$branch3x3dbl_3 <- conv_block(96, 96, kernel_size=3, stride=2)
  },
  .forward = function(x) {
    branch3x3 <- self$branch3x3(x)

    branch3x3dbl <- self$branch3x3dbl_1(x)
    branch3x3dbl <- self$branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl <- self$branch3x3dbl_3(branch3x3dbl)

    branch_pool <- torch::nnf_max_pool2d(x, kernel_size=3, stride=2)

    list(branch3x3, branch3x3dbl, branch_pool)
  },
  forward = function(x) {
    outputs <- self$.forward(x)
    torch::torch_cat(outputs, 2)
  }
)

InceptionC <- torch::nn_module(
  "InceptionC",
  initialize = function(in_channels, channels_7x7, conv_block = NULL) {
    if (is.null(conv_block)) {
      conv_block <- BasicConv2d
    }

    self$branch1x1 <- conv_block(in_channels, 192, kernel_size=1)
    c7 <- channels_7x7
    self$branch7x7_1 <- conv_block(in_channels, c7, kernel_size=1)
    self$branch7x7_2 <- conv_block(c7, c7, kernel_size=c(1, 7), padding=c(0, 3))
    self$branch7x7_3 <- conv_block(c7, 192, kernel_size=c(7, 1), padding=c(3, 0))

    self$branch7x7dbl_1 <- conv_block(in_channels, c7, kernel_size=1)
    self$branch7x7dbl_2 <- conv_block(c7, c7, kernel_size=c(7, 1), padding=c(3, 0))
    self$branch7x7dbl_3 <- conv_block(c7, c7, kernel_size=c(1, 7), padding=c(0, 3))
    self$branch7x7dbl_4 <- conv_block(c7, c7, kernel_size=c(7, 1), padding=c(3, 0))
    self$branch7x7dbl_5 <- conv_block(c7, 192, kernel_size=c(1, 7), padding=c(0, 3))

    self$branch_pool = conv_block(in_channels, 192, kernel_size=1)
  },
  .forward = function(x) {
    branch1x1 <- self$branch1x1(x)

    branch7x7 <- self$branch7x7_1(x)
    branch7x7 <- self$branch7x7_2(branch7x7)
    branch7x7 <- self$branch7x7_3(branch7x7)

    branch7x7dbl <- self$branch7x7dbl_1(x)
    branch7x7dbl <- self$branch7x7dbl_2(branch7x7dbl)
    branch7x7dbl <- self$branch7x7dbl_3(branch7x7dbl)
    branch7x7dbl <- self$branch7x7dbl_4(branch7x7dbl)
    branch7x7dbl <- self$branch7x7dbl_5(branch7x7dbl)

    branch_pool <- torch::nnf_avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool <- self$branch_pool(branch_pool)

    list(branch1x1, branch7x7, branch7x7dbl, branch_pool)
  },
  forward = function(x) {
    outputs <- self$.forward(x)
    torch::torch_cat(outputs, 2)
  }
)

InceptionD <- torch::nn_module(
  "InceptionD",
  initialize = function(in_channels, conv_block = NULL) {

    if (is.null(conv_block)) {
      conv_block <- BasicConv2d
    }

    self$branch3x3_1 <- conv_block(in_channels, 192, kernel_size=1)
    self$branch3x3_2 <- conv_block(192, 320, kernel_size=3, stride=2)

    self$branch7x7x3_1 <- conv_block(in_channels, 192, kernel_size=1)
    self$branch7x7x3_2 <- conv_block(192, 192, kernel_size=c(1, 7), padding=c(0, 3))
    self$branch7x7x3_3 <- conv_block(192, 192, kernel_size=c(7, 1), padding=c(3, 0))
    self$branch7x7x3_4 <- conv_block(192, 192, kernel_size=3, stride=2)
  },
  .forward = function(x) {
    branch3x3 <- self$branch3x3_1(x)
    branch3x3 <- self$branch3x3_2(branch3x3)

    branch7x7x3 <- self$branch7x7x3_1(x)
    branch7x7x3 <- self$branch7x7x3_2(branch7x7x3)
    branch7x7x3 <- self$branch7x7x3_3(branch7x7x3)
    branch7x7x3 <- self$branch7x7x3_4(branch7x7x3)

    branch_pool <- torch::nnf_max_pool2d(x, kernel_size=3, stride=2)
    list(branch3x3, branch7x7x3, branch_pool)
  },
  forward = function(x) {
    outputs <- self$.forward(x)
    torch::torch_cat(outputs, 2)
  }
)

InceptionE <- torch::nn_module(
  "InceptionE",
  initialize = function(in_channels, conv_block = NULL) {
    if (is.null(conv_block)) {
      conv_block <- BasicConv2d
    }

    self$branch1x1 <- conv_block(in_channels, 320, kernel_size=1)

    self$branch3x3_1 <- conv_block(in_channels, 384, kernel_size=1)
    self$branch3x3_2a <- conv_block(384, 384, kernel_size=c(1, 3), padding=c(0, 1))
    self$branch3x3_2b <- conv_block(384, 384, kernel_size=c(3, 1), padding=c(1, 0))

    self$branch3x3dbl_1 <- conv_block(in_channels, 448, kernel_size=1)
    self$branch3x3dbl_2 <- conv_block(448, 384, kernel_size=3, padding=1)
    self$branch3x3dbl_3a <- conv_block(384, 384, kernel_size=c(1, 3), padding=c(0, 1))
    self$branch3x3dbl_3b <- conv_block(384, 384, kernel_size=c(3, 1), padding=c(1, 0))

    self$branch_pool <- conv_block(in_channels, 192, kernel_size=1)
  },
  .forward = function(x) {
    branch1x1 <- self$branch1x1(x)

    branch3x3 <- self$branch3x3_1(x)
    branch3x3 <- list(
      self$branch3x3_2a(branch3x3),
      self$branch3x3_2b(branch3x3)
    )
    branch3x3 <- torch::torch_cat(branch3x3, 2)

    branch3x3dbl <- self$branch3x3dbl_1(x)
    branch3x3dbl <- self$branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl <- list(
      self$branch3x3dbl_3a(branch3x3dbl),
      self$branch3x3dbl_3b(branch3x3dbl)
    )
    branch3x3dbl <- torch::torch_cat(branch3x3dbl, 2)

    branch_pool <- torch::nnf_avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool <- self$branch_pool(branch_pool)

    list(branch1x1, branch3x3, branch3x3dbl, branch_pool)
  },
  forward = function(x) {
    outputs <- self$.forward(x)
    torch::torch_cat(outputs, 2)
  }
)

InceptionAux <- torch::nn_module(
  "InceptionAux",
  initialize = function(in_channels, num_classes, conv_block = NULL) {
    if (is.null(conv_block)) {
      conv_block <- BasicConv2d
    }

    self$conv0 <- conv_block(in_channels, 128, kernel_size=1)
    self$conv1 <- conv_block(128, 768, kernel_size=5)
    self$conv1$stddev = 0.01
    self$fc <- torch::nn_linear(768, num_classes)
    self$fc$stddev <- 0.001
  },
  forward = function(x) {
    # N x 768 x 17 x 17
    x <- torch::nnf_avg_pool2d(x, kernel_size=5, stride=3)
    # N x 768 x 5 x 5
    x <- self$conv0(x)
    # N x 128 x 5 x 5
    x <- self$conv1(x)
    # N x 768 x 1 x 1
    # Adaptive average pooling
    x <- torch::nnf_adaptive_avg_pool2d(x, c(1, 1))
    # N x 768 x 1 x 1
    x <- torch::torch_flatten(x, start_dim = 2)
    # N x 768
    x <- self$fc(x)
    # N x 1000
    x
  }
)

BasicConv2d <- torch::nn_module(
  "BasicConv2d",
  initialize = function(in_channels, out_channels, ...) {
    self$conv <- torch::nn_conv2d(in_channels, out_channels, bias=FALSE, ...)
    self$bn <- torch::nn_batch_norm2d(out_channels, eps=0.001)
  },
  forward = function(x) {
    x <- self$conv(x)
    x <- self$bn(x)
    torch::nnf_relu(x, inplace=TRUE)
  }
)

inception_model_urls <- list(
  inception_v3_google =c("https://torch-cdn.mlverse.org/models/vision/v2/models/inception_v3_google.pth", "8d60b4fcf263f2a8d2ed21f0f9690e3b", "~110 MB")
)

#' Inception v3 model
#'
#' Architecture from [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
#' The required minimum input size of the model is 75x75.
#' @note
#' **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
#' N x 3 x 299 x 299, so ensure your images are sized accordingly.
#' @param pretrained (bool): If `TRUE`, returns a model pre-trained on ImageNet
#' @param progress (bool): If `TRUE`, displays a progress bar of the download to stderr
#' @param ... Used to pass keyword arguments to the Inception module:
#'  - aux_logits (bool): If `TRUE`, add an auxiliary branch that can improve training.
#'  Default: *TRUE*
#'  - transform_input (bool): If `TRUE`, preprocess the input according to the method with which it
#'  was trained on ImageNet. Default: *FALSE*
#'
#' @family classification_model
#' @export
model_inception_v3 <-function(pretrained = FALSE, progress = TRUE, ...) {
  args <- rlang::list2(...)

  if (pretrained) {

    if (is.null(args$transform_input)) {
      args$transform_input <- TRUE
    }
    if (!is.null(args$aux_logits)) {
      original_aux_logits <- args$aux_logits
      args$aux_logits <- TRUE
    } else {
      original_aux_logits <- TRUE
    }

    args$init_weights <- FALSE  # we are loading weights from a pretrained model

    model <- do.call(Inception3, args)
    r <- inception_model_urls[['inception_v3_google']]
    cli_inform("Model weights for {.cls {class(model)[1]}} ({.emph {r[3]}}) will be downloaded and processed if not already available.")
    state_dict_path <- download_and_cache(r[1])
    if (!tools::md5sum(state_dict_path) == r[2])
      runtime_error("Corrupt file! Delete the file in {state_dict_path} and try again.")

    state_dict <- torch::load_state_dict(state_dict_path)
    model$load_state_dict(state_dict)
    return(model)
  }

  Inception3(...)
}
