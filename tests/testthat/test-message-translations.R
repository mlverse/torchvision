test_that("R-level type_error messages are correctly translated in FR", {
  withr::with_language(lang = "fr",
                       expect_error(
                         transform_adjust_gamma(torch::torch_rand_like(c(3, 5, 5)), gamma = 0.5),
                        regexp = "Le tenseur n'est pas une image torch",
                        fixed = TRUE
                      )
  )
})

test_that("R-level cli_warning messages are correctly translated in FR", {
  withr::with_language(lang = "fr",
                       expect_warning(
                         torchvision:::Inception3(),
                        regexp = "L'initialisation des poids par défaut de inception_v3",
                        fixed = TRUE
                      )
  )
})

test_that("R-level value_error messages are glued and correctly translated in FR", {
  withr::with_language(lang = "fr",
                       expect_error(
                         transform_normalize(torch::torch_rand(c(3,5,5)), 3, 0),
                        regexp = "Après conversion en Float,",
                        fixed = TRUE
                      )
  )
})

