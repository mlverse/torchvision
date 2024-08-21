test_that("R-level type_error messages are correctly translated in FR", {
  withr::with_language(lang = "fr",
                       expect_error(
                         transform_adjust_gamma(torch::torch_rand_like(c(4, 4, 3)), gamma = 0.5),
                        regexp = "Le tenseur n'est pas une image torch",
                        fixed = TRUE
                      )
  )
})

test_that("R-level value_error messages are correctly translated in FR", {
  withr::with_language(lang = "fr",
                       expect_warning(
                         transform_adjust_gamma(torch::torch_rand(c(4, 4, 3)), gamma = -0.5),
                        regexp = "`gamma` doit être positif",
                        fixed = TRUE
                      )
  )
})

