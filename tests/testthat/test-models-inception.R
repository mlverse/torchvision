test_that("inception_v3 pretrained", {
  model <- model_inception_v3(pretrained = TRUE)
  model$eval()
  x <- model(torch_ones(1, 3, 299, 299))
  expect_equal(as.numeric(x[1,1]), 0.18005196750164032, tol = 5e-6)
})
