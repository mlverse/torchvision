context("dataset-rf100-medical")

t <- withr::local_tempdir()

test_that("rf100_medical_collection handles missing files gracefully", {
  expect_error(
    rf100_medical_collection(dataset = "abdomen_mri", split = "train", download = FALSE),
    class = "runtime_error"
  )
})

datasets <- c("radio_signal", "rheumatology", "knee",
  "abdomen_mri", "brain_axial_mri", "gynecology_mri",
  "brain_tumor", "fracture"
)

for (ds_name in datasets) {
  test_that(paste0("rf100_medical_collection loads ", ds_name, " correctly"), {
    ds <- rf100_medical_collection(dataset = ds_name, split = "train", download = TRUE)

    expect_s3_class(ds, "rf100_medical_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_gt(length(unique(ds$classes)), 1)

    item <- ds[2] # as 2 datasets have their first item wo bbox

    expect_type(item$y, "list")
    expect_named(item$y, c("image_id","labels","boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}
