context("models-lw_detr")

test_that("non-pretrained model_lw_detr_tiny works with single image and batch", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_lw_detr_tiny(num_classes = 91)
  input <- base_loader("assets/class/cat/cat.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(256, 256)) %>% torch_unsqueeze(1)
  model$eval()
  torch::with_no_grad({out <- model(input)})
  expect_named(out, "detections")
  expect_is(out$detections, "list")
  expect_length(out$detections, 1)
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_tensor(out$detections[[1]]$boxes)
  expect_tensor(out$detections[[1]]$labels)
  expect_tensor(out$detections[[1]]$scores)
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)

  batch <- torch_stack(
    list(
      base_loader("assets/class/cat/cat.0.jpg") %>% transform_to_tensor() %>% transform_resize(c(256, 256)),
      base_loader("assets/class/cat/cat.1.jpg") %>% transform_to_tensor() %>% transform_resize(c(256, 256))
    ),
    dim = 1
  )
  torch::with_no_grad({out <- model(batch)})
  expect_length(out$detections, 2)
  expect_named(out$detections[[2]], c("boxes", "labels", "scores"))
  expect_equal(out$detections[[2]]$boxes$shape[2], 4L)
})

test_that("model_lw_detr_tiny respects num_classes and num_select", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_lw_detr_tiny(num_classes = 10, num_select = 25)
  input <- base_loader("assets/class/dog/dog.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(256, 256)) %>% torch_unsqueeze(1)
  model$eval()
  torch::with_no_grad({out <- model(input)})
  expect_equal(out$detections[[1]]$boxes$shape[1], 25L)
  expect_equal(out$detections[[1]]$scores$shape[1], 25L)
  labels_vec <- as.integer(out$detections[[1]]$labels$cpu())
  expect_true(all(labels_vec >= 0 & labels_vec < 10))
})

test_that("model_lw_detr_tiny supports pixel_mask", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  model <- model_lw_detr_tiny(num_classes = 91, num_select = 50)
  model$eval()
  input <- base_loader("assets/class/dog/dog.0.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(256, 256)) %>% torch_unsqueeze(1)

  # Mark only the top 160 rows valid, as if the image were letterbox-padded.
  mask <- torch_zeros(c(1, 256, 256), dtype = torch::torch_bool())
  mask[, 1:160, ] <- TRUE
  full <- torch_ones(c(1, 256, 256), dtype = torch::torch_bool())

  torch::with_no_grad({
    out_mask <- model(input, pixel_mask = mask)
    out_full <- model(input, pixel_mask = full)
    out_none <- model(input)
  })

  d <- out_mask$detections[[1]]
  expect_named(d, c("boxes", "labels", "scores"))
  expect_equal(d$boxes$shape[1], 50L)
  expect_equal(d$boxes$shape[2], 4L)
  expect_equal(d$scores$shape[1], 50L)

  expect_equal(as.numeric(out_full$detections[[1]]$scores$cpu()),
               as.numeric(out_none$detections[[1]]$scores$cpu()))

  expect_false(isTRUE(all.equal(
    as.numeric(d$scores$cpu()),
    as.numeric(out_none$detections[[1]]$scores$cpu())
  )))
})

test_that("model_lw_detr pretrained weights require COCO num_classes", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  expect_error(model_lw_detr_tiny(pretrained = TRUE, num_classes = 10), "num_classes = 91")
})

test_that("tests for pretrained model_lw_detr_tiny", {
  skip_if(Sys.getenv("TEST_LARGE_MODELS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_MODELS=1 to enable tests requiring large downloads.")
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  input <- base_loader("assets/class/cat/cat.4.jpg") %>%
    transform_to_tensor() %>% transform_resize(c(640, 640)) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225)) %>%
    torch_unsqueeze(1)
  model <- model_lw_detr_tiny(pretrained = TRUE)
  model$eval()
  torch::with_no_grad({out <- model(input)})
  expect_named(out, "detections")
  expect_named(out$detections[[1]], c("boxes", "labels", "scores"))
  expect_equal(out$detections[[1]]$boxes$shape[2], 4L)
  labels_vec <- as.integer(out$detections[[1]]$labels$cpu())
  scores_vec <- as.numeric(out$detections[[1]]$scores$cpu())
  expect_true(all(labels_vec >= 0 & labels_vec <= 90))

  # Correctness: the top detection on a cat image should be the cat class
  # (COCO id 17) with a confident score.
  top <- which.max(scores_vec)
  expect_equal(labels_vec[top], 17L)
  expect_gt(scores_vec[top], 0.4)
})
