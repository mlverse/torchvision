context("dataset-flickr")

t <- withr::local_tempdir()

test_that("tests for the flickr8k dataset for train split", {
  skip_on_cran()

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")


  flickr8k <- flickr8k_caption_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(flickr8k, 6000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,598500)
  expect_identical(first_item$y[1], "A black dog is running after a white dog in the snow .")
  expect_identical(first_item$y[2], "Black dog chasing brown dog through snow")
  expect_identical(first_item$y[3], "Two dogs chase each other across the snowy ground .")
  expect_identical(first_item$y[4], "Two dogs play together in the snow .")
  expect_identical(first_item$y[5], "Two dogs running through a low lying body of water .")
})

test_that("tests for the flickr8k dataset for test split", {
  skip_on_cran()

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  flickr8k <- flickr8k_caption_dataset(root = t, train = FALSE)
  expect_length(flickr8k, 1000)
  first_item <- flickr8k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,502500)
  expect_identical(first_item$y[1], "The dogs are in the snow in front of a fence .")
  expect_identical(first_item$y[2], "The dogs play on the snow .")
  expect_identical(first_item$y[3], "Two brown dogs playfully fight in the snow .")
  expect_identical(first_item$y[4], "Two brown dogs wrestle in the snow .")
  expect_identical(first_item$y[5], "Two dogs playing in the snow .")
})

test_that("tests for the flickr8k dataset for dataloader", {
  skip_on_cran()

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
  
  flickr8k <- flickr8k_caption_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    },
    target_transform = function(y) glue::glue_collapse(y, sep = " ")
  )
  dl <- dataloader(flickr8k, batch_size = 4)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y,"character")
  expect_length(batch$y,4)
  expect_identical(batch$y[1], "A black dog is running after a white dog in the snow . Black dog chasing brown dog through snow Two dogs chase each other across the snowy ground . Two dogs play together in the snow . Two dogs running through a low lying body of water .")
  expect_identical(batch$y[2], "A little baby plays croquet . A little girl plays croquet next to a truck . The child is playing croquette by the truck . The kid is in front of a car with a put and a ball . The little boy is playing with a croquet hammer and ball beside the car .")
  expect_identical(batch$y[3], "A brown dog in the snow has something hot pink in its mouth . A brown dog in the snow holding a pink hat . A brown dog is holding a pink shirt in the snow . A dog is carrying something pink in its mouth while walking through the snow . A dog with something pink in its mouth is looking forward .")
  expect_identical(batch$y[4], "A brown dog is running along a beach . A brown dog wearing a black collar running across the beach . A dog walks on the sand near the water . Brown dog running on the beach . The large brown dog is running on the beach by the ocean .")
})

test_that("tests for the flickr30k dataset for train split", {
  skip_on_cran()

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")


  flickr30k <- flickr30k_caption_dataset(root = t, train = TRUE, download = TRUE)
  expect_length(flickr30k, 29000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,499500)
  expect_identical(first_item$y[1], "Two young guys with shaggy hair look at their hands while hanging out in the yard.")
  expect_identical(first_item$y[2], "Two young, White males are outside near many bushes.")
  expect_identical(first_item$y[3], "Two men in green shirts are standing in a yard.")
  expect_identical(first_item$y[4], "A man in a blue shirt standing in a garden.")
  expect_identical(first_item$y[5], "Two friends enjoy time spent together.")

})

test_that("tests for the flickr30k dataset for test split", {
  skip_on_cran()

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  flickr30k <- flickr30k_caption_dataset(root = t, train = FALSE)
  expect_length(flickr30k, 1000)
  first_item <- flickr30k[1]
  expect_named(first_item, c("x", "y"))
  expect_type(first_item$x, "double")
  expect_type(first_item$y,"character")
  expect_length(first_item$x,691500)
  expect_identical(first_item$y[1], "The man with pierced ears is wearing glasses and an orange hat.")
  expect_identical(first_item$y[2], "A man with glasses is wearing a beer can crocheted hat.")
  expect_identical(first_item$y[3], "A man with gauges and glasses is wearing a Blitz hat.")
  expect_identical(first_item$y[4], "A man in an orange hat starring at something.")
  expect_identical(first_item$y[5], "A man wears an orange hat and glasses.")
})

test_that("tests for the flickr30k dataset for dataloader", {
  skip_on_cran()

  skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
      "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")

  flickr30k <- flickr30k_caption_dataset(
    root = t,
    transform = function(x) {
      x %>% transform_to_tensor() %>% transform_resize(c(224, 224))
    },
    target_transform = function(y) glue::glue_collapse(y, sep = " ")
  )
  dl <- dataloader(flickr30k, batch_size = 4)
  iter <- dataloader_make_iter(dl)
  batch <- dataloader_next(iter)
  expect_named(batch, c("x", "y"))
  expect_tensor(batch$x)
  expect_length(batch$x,602112)
  expect_tensor_shape(batch$x,c(4,3,224,224))
  expect_tensor_dtype(batch$x,torch_float())
  expect_type(batch$y, "character")
  expect_identical(batch$y[1], "Two young guys with shaggy hair look at their hands while hanging out in the yard. Two young, White males are outside near many bushes. Two men in green shirts are standing in a yard. A man in a blue shirt standing in a garden. Two friends enjoy time spent together.")
  expect_identical(batch$y[2], "Several men in hard hats are operating a giant pulley system. Workers look down from up above on a piece of equipment. Two men working on a machine wearing hard hats. Four men on top of a tall structure. Three men on a large rig.")
  expect_identical(batch$y[3], "A child in a pink dress is climbing up a set of stairs in an entry way. A little girl in a pink dress going into a wooden cabin. A little girl climbing the stairs to her playhouse. A little girl climbing into a wooden playhouse. A girl going into a wooden building.")
  expect_identical(batch$y[4], "Someone in a blue shirt and hat is standing on stair and leaning against a window. A man in a blue shirt is standing on a ladder cleaning a window. A man on a ladder cleans the window of a tall building. Man in blue shirt and jeans on ladder cleaning windows A man on a ladder cleans a window")
})
