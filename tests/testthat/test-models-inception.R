test_that("inception_v3 pretrained", {
  model <- model_inception_v3(pretrained = TRUE)
  model$eval()
  x <- model(torch_ones(2, 3, 299, 299))
  # the value has been copied from running the same model on pytorch.
  expect_equal(as.numeric(x[1,1]), 0.18005196750164032, tol = 5e-6)
  expect_tensor_shape(x, c(2, 1000))
})

test_that("inception_v3 keeps the auxiliary branch by default", {
  model <- model_inception_v3(pretrained = FALSE, init_weights = FALSE)
  expect_true(model$aux_logits)

  # in training mode both heads are returned, in evaluation mode only the logits
  model$train()
  out <- model(torch_ones(2, 3, 299, 299))
  expect_named(out, c("logits", "aux_logits"))
  expect_tensor_shape(out$logits, c(2, 1000))
  expect_tensor_shape(out$aux_logits, c(2, 1000))

  model$eval()
  expect_tensor_shape(model(torch_ones(2, 3, 299, 299)), c(2, 1000))
})

test_that("inception_v3 respects aux_logits = FALSE", {
  model <- model_inception_v3(pretrained = FALSE, aux_logits = FALSE, init_weights = FALSE)
  expect_false(model$aux_logits)

  # without the auxiliary branch a single tensor is returned, also in training mode
  model$train()
  expect_tensor_shape(model(torch_ones(2, 3, 299, 299)), c(2, 1000))
})

test_that("inception_v3 pretrained respects aux_logits = FALSE", {
  # the auxiliary branch has to be enabled while the state dict is loaded, but must be removed
  # afterwards, including its parameters
  model <- model_inception_v3(pretrained = TRUE, aux_logits = FALSE)
  expect_false(model$aux_logits)

  reference <- model_inception_v3(pretrained = FALSE, aux_logits = FALSE, init_weights = FALSE)
  expect_equal(length(model$parameters), length(reference$parameters))

  # the weights that are kept are still the pretrained ones
  model$eval()
  x <- model(torch_ones(2, 3, 299, 299))
  expect_equal(as.numeric(x[1,1]), 0.18005196750164032, tol = 5e-6)

  # a single tensor is returned, also in training mode. This has to be checked after the value
  # above, because a forward pass in training mode updates the batch norm statistics.
  model$train()
  expect_tensor_shape(model(torch_ones(2, 3, 299, 299)), c(2, 1000))
})
