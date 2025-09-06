context("dataset-rf100-electromagnetic")

t <- withr::local_tempdir()

test_that("rf100_medical_collection handles missing files gracefully", {
  expect_error(
    rf100_medical_collection(dataset = "thermal_dog_and_people", split = "train", download = FALSE),
    class = "runtime_error"
  )
})

datasets <- c(
  "thermal_dog_and_people", "solar_panel", "radio_signal",
  "thermal_cheetah", "rheumatology", "knee",
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
    expect_named(item$y, c("labels", "boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })
}
test_that(paste0("rf100_medical_collection loads 'ir_object' correctly"), {
    skip_if(Sys.getenv("TEST_LARGE_DATASETS", unset = 0) != 1,
            "Skipping test: set TEST_LARGE_DATASETS=1 to enable tests requiring large downloads.")
    ds <- rf100_medical_collection(dataset ="ir_object", split = "train", download = TRUE)

    expect_s3_class(ds, "rf100_medical_collection")
    expect_gt(ds$.length(), 1)
    expect_type(ds$classes, "character")
    expect_gt(length(unique(ds$classes)), 1)

    item <- ds[1]

    expect_type(item$y, "list")
    expect_named(item$y, c("labels", "boxes"))
    expect_type(item$y$labels, "integer")
    expect_tensor(item$y$boxes)
    expect_equal(item$y$boxes$ndim, 2)
    expect_equal(item$y$boxes$size(2), 4)
    expect_s3_class(item, "image_with_bounding_box")
  })

