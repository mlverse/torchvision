context("dataset-fgvc")

test_that("FGVC-Aircraft dataset: all splits, levels, and dataloader", {
  t <- tempfile()

  expect_error(
    fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = FALSE),
    class = "runtime_error"
  )

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "variant", download = TRUE)
  expect_equal(length(fgvc), 5333)
  item <- fgvc[1]
  expect_named(item, c("x", "y"))
  expect_true(inherits(item$x, "torch_tensor"))
  expect_type(item$y, "integer")
  expect_equal(as.numeric(item$y), 79)

  resize_collate_fn <- function(batch) {
    xs <- lapply(batch, function(sample) {
      torchvision::transform_resize(sample$x, c(224, 224))
    })
    xs <- torch::torch_stack(xs)
    ys <- torch::torch_tensor(sapply(batch, function(sample) sample$y), dtype = torch::torch_long())
    list(x = xs, y = ys)
  }
  dl <- torch::dataloader(
    dataset = fgvc,
    batch_size = 2,
    shuffle = TRUE,
    collate_fn = resize_collate_fn
  )
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_true(inherits(batch$x, "torch_tensor"))
  expect_equal(dim(batch$x)[1], 2)
  expect_equal(length(batch$y), 2)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "variant")
  expect_equal(length(fgvc), 1334)
  expect_equal(as.numeric(fgvc[1]$y), 65)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "variant")
  expect_equal(length(fgvc), 6667)
  expect_equal(as.numeric(fgvc[1]$y), 1)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "variant")
  expect_equal(length(fgvc), 3333)
  expect_equal(as.numeric(fgvc[1]$y), 1)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "family")
  expect_equal(length(fgvc), 5333)
  expect_equal(as.numeric(fgvc[1]$y), 51)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "family")
  expect_equal(length(fgvc), 1334)
  expect_equal(as.numeric(fgvc[1]$y), 41)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "family")
  expect_equal(length(fgvc), 6667)
  expect_equal(as.numeric(fgvc[1]$y), 13)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "family")
  expect_equal(length(fgvc), 3333)
  expect_equal(as.numeric(fgvc[1]$y), 13)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "train", annotation_level = "manufacturer")
  expect_equal(length(fgvc), 5333)
  expect_equal(as.numeric(fgvc[1]$y), 17)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "val", annotation_level = "manufacturer")
  expect_equal(length(fgvc), 1334)
  expect_equal(as.numeric(fgvc[1]$y), 14)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "trainval", annotation_level = "manufacturer")
  expect_equal(length(fgvc), 6667)
  expect_equal(as.numeric(fgvc[1]$y), 5)

  fgvc <- fgvc_aircraft_dataset(root = t, split = "test", annotation_level = "manufacturer")
  expect_equal(length(fgvc), 3333)
  expect_equal(as.numeric(fgvc[1]$y), 5)
})
