# Texture Generation with Neural Cellular Automata
# This script is a port to torch for R of
# https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/texture_nca_pytorch.ipynb
# by Alex Mordvintsev (@zzznah)

# Packages ----------------------------------------------------------------

library(torch)
library(torchvision)
library(purrr)
library(zeallot)

device <- if(cuda_is_available()) "cuda" else "cpu"

# Function definitions ----------------------------------------------------

gram_matrix <- function(input) {
  size <- input$size()
  G <- torch_einsum("bchw, bdhw -> bcd", list(input, input))
  G$div(prod(tail(size, 2)))
}

style_loss <- function(content, style) {
  (content - style)$square()$mean()
}

# Model definition --------------------------------------------------------

cnn <- model_vgg16(pretrained = TRUE)$features

# we create an nn_module that does the same as the sequential container but
# returns the results of all convolutions. we also replace inplace operations
# for copy on modify ones
features <- nn_module(
  initialize = function(cnn) {
    self$cnn <- cnn
  },
  forward = function(input) {
    conv_outs <- list()
    for (i in seq_along(self$cnn)) {
      layer <- self$cnn[[i]]

      if (inherits(layer, "nn_relu"))
        input <- nnf_relu(input)
      else
        input <- layer(input)

      if (inherits(layer, "nn_conv2d"))
        conv_outs[[length(conv_outs) + 1]] <- input

    }
    conv_outs
  }
)

model <- features(cnn)
model$to(device = device)


# CA model ---------------------------------------------------------------

perchannel_conv <- function(x, filters) {
  c(b, ch, h, w) %<-% x$shape
  y <- x$reshape(c(b*ch, 1, h, w))
  y <- nnf_pad(y, rep(1, 4), mode = "circular")
  y <- nnf_conv2d(y, filters[,newaxis])
  y$reshape(c(b, -1, h, w))
}

perception <- nn_module(
  initialize = function() {
    ident <- torch_tensor(rbind(
      c(0, 0, 0),
      c(0, 1, 0),
      c(0, 0, 0)
    ), device = device)

    sobel_x <- torch_tensor(rbind(
      c(-1, 0, 1),
      c(-2, 0, 2),
      c(-1, 0, 1)
    ), device = device)$div(8)

    lap <- torch_tensor(rbind(
      c(1, 2, 1),
      c(-1, -12, 2),
      c(1, 2, 1)
    ), device = device)$div(16)

    self$filters <- torch_stack(list(
      ident,
      sobel_x,
      sobel_x$t(),
      lap
    ))
  },

  forward = function(x) {
    perchannel_conv(x, self$filters)
  }
)

CA <- nn_module(
  "CA",
  initialize = function(chn = 12, hidden_n = 96) {
    self$chn <- chn
    self$w1 <- nn_conv2d(chn*4, hidden_n, 1)
    self$w2 <- nn_conv2d(hidden_n, chn, 1, bias = FALSE)
    nn_init_zeros_(self$w2$weight)
    self$perception <- perception()
  },
  forward = function(x, update_rate = 0.5) {
    y <- x %>%
      self$perception() %>%
      self$w1() %>%
      torch_relu() %>%
      self$w2()

    update_mask <- torch_rand_like(y[,1,..,drop=FALSE]) < update_rate
    x + y * update_mask
  },
  seed = function(n, size = 128) {
    torch_zeros(n, self$chn, size, size, device = device)
  }
)


# Read image and preprocess -----------------------------------------------

norm_mean <- c(0.485, 0.456, 0.406)
norm_std <- c(0.229, 0.224, 0.225)

normalize <- function(img) {
  transform_normalize(img, norm_mean, norm_std)
}

denormalize <- function(img) {
  transform_normalize(img, -norm_mean/norm_std, 1/norm_std)
}

plot_img <- function(x) {
  im <- denormalize(x)[1,..]$
    permute(c(2, 3, 1))$
    to(device = "cpu")$
    clamp(0,1) %>% # make it [0,1]
    as.array()
  op <- par(mar=rep(0, 4))
  plot(as.raster(im), asp = NA)
  par(op)
}

load_image <- function(path) {
  x <- base_loader(path) %>%
    transform_to_tensor() %>%
    transform_resize(c(128, 128))
  x <- x[newaxis,..]
  x <- normalize(x)
  x$to(device = device)
}

img_path <- tempfile(fileext = ".jpg")
download.file(
  "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Tempera%2C_charcoal_and_gouache_mountain_painting_by_Nicholas_Roerich.jpg/301px-Tempera%2C_charcoal_and_gouache_mountain_painting_by_Nicholas_Roerich.jpg",
  img_path
)

img <- load_image(img_path)
plot_img(img)


# Compute target style ----------------------------------------------------

compute_style <- function(imgs) {
  style_layers <- 1:5 # first five convolutions
  convs <- model(imgs)[style_layers]
  convs %>%
    map(gram_matrix)
}

with_no_grad({
  target_style <- compute_style(img)
})


# Setup training ----------------------------------------------------------

ca <- CA()
ca$to(device = device)
optimizer <- optim_adam(ca$parameters, lr = 1e-3)
scheduler <- lr_step(optimizer, step_size = 2000, gamma = 0.3)

with_no_grad({
  pool <- ca$seed(1024)
})

pb <- progress::progress_bar$new(total = 4000, format = ":current/:total [:bar] :elapsed/:eta")
for (i in 1:4000) {
  with_no_grad({
    batch_idx <- sample.int(pool$size(1), 4)
    x <- pool[batch_idx,..]
  })

  step_n <- sample(32:96, 1)

  for (k in 1:step_n) {
    x <- ca(x)
  }

  styles <- compute_style(x[,1:3,..])
  loss <- purrr::map2(styles, target_style, style_loss) %>%
    purrr::reduce(~.x + .y)

  with_no_grad({

    loss$backward()

    for (p in ca$parameters)
      p$grad$div_(p$grad$norm()+1e-8)

    optimizer$step()
    optimizer$zero_grad()
    scheduler$step()

    pool[batch_idx,..] <- x

  })
  pb$tick()
}

torch_save(ca, "ca.pt")

# NCA video ---------------------------------------------------------------

create_frames <- function() {
  x <- ca$seed(1, 128)
  for (k in 1:300) {
    step_n <- min(2^(k%/%30), 16)
    for (i in 1:step_n) {
      with_no_grad({
        x <- ca(x)
      })
    }
    plot_img(x[1,1:3,..,drop=FALSE])
  }
}

gifski::save_gif(
  create_frames(),
  gif_file = "nca_video.gif",
  width = 128,
  height = 128,
  delay = 0.1
)
