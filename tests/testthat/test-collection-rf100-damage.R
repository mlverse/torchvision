context("dataset-rf100-damage")

t <- withr::local_tempdir()

test_that("rf100_damage_collection handles missing files gracefully", {
  expect_error(
    rf100_damage_collection(dataset = "asbestos", split = "train", download = FALSE),
    class = "runtime_error"
  )
})

datasets <- c("liquid_crystals", "solar_panel", "asbestos")

for (ds_name in datasets) {
  test_that(paste0("rf100_damage_collection loads ", ds_name, " correctly"), {
    ds <- rf100_damage_collection(dataset = ds_name, split = "train", download = TRUE)

    expect_s3_class(ds, "rf100_damage_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_gt(length(unique(ds$classes)), 1)

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

