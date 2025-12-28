context("dataset-rf100-doc")

test_that("rf100_document_collection handles missing dataset gracefully", {
  expect_error(
    rf100_document_collection(dataset = "paragraph", split = "train", download = FALSE),
    "'arg' should be one of"
  )
})

# small datasets
dataset <- data.frame(name = c("tweeter_post", "tweeter_profile", "document_part",
                                "activity_diagram", "signature", "currency"),
                       nlevels = c(2L, 1L, 2L, 19L, 1L, 10L)
)

for (ds_name in dataset$name) {
  test_that(paste0("rf100_document_collection loads ", ds_name, " correctly"), {
    ds <- rf100_document_collection(dataset = ds_name, split = "train", download = TRUE)

    expect_s3_class(ds, "rf100_document_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_length(unique(ds$classes), dataset[dataset$name == ds_name,]$nlevels)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("image_id","labels","boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}

dataset <- data.frame(name = c("paper_part", "wine_label"),
                      nlevels = c(19L, 12L)
)

for (ds_name in dataset$name) {
  test_that(paste0("rf100_document_collection loads ", ds_name, " correctly"), {
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
          "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
    ds <- rf100_document_collection(dataset = ds_name, split = "test", download = TRUE)

    expect_s3_class(ds, "rf100_document_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_length(unique(ds$classes), dataset[dataset$name == ds_name,]$nlevels)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("image_id","labels","boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}


test_that("rf100_document_collection datasets can be turned into a dataloader wo transform", {
  ds <- rf100_document_collection(dataset = "document_part", split = "test", download = TRUE)

  expect_equal(ds$.length(), 181)
  expect_type(ds$classes, "character")
  expect_length(unique(ds$classes), 2)

  items <- ds[7:9]
  # Check shape, dtype, and values on X
  expect_named(items, c("x", "y"))
  expect_length(items$x, 3)
  expect_equal(dim(items$x[[1]]), c(640, 640, 3))
  expect_lte(max(items$x[[1]]), 1)
  # Check shape, dtype and names on y
  expect_named(items$y,c("image_id","labels","boxes"))
  expect_length(items$y$image_id,6)
  expect_tensor(items$y$boxes)
  expect_tensor_shape(items$y$boxes, c(6,4))


  dl <- dataloader(ds, batch_size = 10, shuffle = TRUE)
  # 181 turns into 19 batches of 10
  expect_length(dl, 19)
  iter <- dataloader_make_iter(dl)
  expect_no_error(
    i <- dataloader_next(iter)
  )
  # Check shape, dtype, and values on X
  expect_named(i, c("x", "y"))
  expect_length(i$x, 10)
  expect_equal(dim(i$x[[1]]), c(640, 640, 3))
  expect_lte(max(i$x[[1]])$item(), 1)
  # Check shape, dtype and names on y
  expect_named(i$y, c("image_id","labels","boxes"))
  expect_tensor(i$y$image_id)
  expect_length(unique(as_array(i$y$image_id)), 10)
  expect_tensor(i$y$labels)
  expect_tensor(i$y$boxes)
  N <- i$y$boxes$shape[1]
  expect_gte(N, 10)
  box_points <- i$y$boxes$shape[2]
  expect_equal(box_points, 4)

})

test_that("rf100_document_collection datasets can be turned into a dataloader with transform", {
  ds <- rf100_document_collection(dataset = "document_part", split = "test",
                                  download = TRUE, transform = transform_to_tensor)

  items <- ds[2:4]
  expect_tensor(items$x)
  expect_tensor_shape(items$x, c(3,3,640, 640))
  expect_tensor_dtype(items$x, torch_float())
  # Check shape, dtype and names on y
  expect_named(items$y,c("image_id","labels","boxes"))
  expect_length(items$y$image_id,6)
  expect_tensor(items$y$boxes)
  expect_tensor_shape(items$y$boxes, c(6,4))


  dl <- dataloader(ds, batch_size = 10, shuffle = TRUE)
  # 181 turns into 19 batches of 10
  expect_length(dl, 19)
  iter <- dataloader_make_iter(dl)
  expect_no_error(
    i <- dataloader_next(iter)
  )
  expect_tensor(i$x)
  expect_tensor_shape(i$x, c(10, 3, 640, 640))
  # Check shape, dtype and names on y
  expect_named(i$y, c("image_id","labels","boxes"))
  expect_tensor(i$y$image_id)
  expect_length(unique(as_array(i$y$image_id)), 10)
  expect_tensor(i$y$labels)
  expect_tensor(i$y$boxes)
  N <- i$y$boxes$shape[1]
  expect_gte(N, 10)
  box_points <- i$y$boxes$shape[2]
  expect_equal(box_points, 4)

})
