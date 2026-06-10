context("sahi-target-transform")

test_that("target_transform_sahi_crop keeps fully contained boxes unchanged", {

  y <- list(
    boxes = torch_tensor(
      matrix(c(10, 10, 30, 30), nrow = 1, byrow = TRUE)
    ),
    labels = torch_tensor(c(1L)),
    area = torch_tensor(c(400)),
    iscrowd = torch_tensor(c(FALSE)),
    image_height = 100L,
    image_width = 100L
  )

  result <- target_transform_sahi_crop(
    y,
    list(
      list(
        top = 0,
        left = 0,
        height = 100,
        width = 100
      )
    )
  )

  expect_length(result, 1)

  expect_equal(
    as.array(result[[1]]$boxes),
    matrix(c(10, 10, 30, 30), nrow = 1)
  )

  expect_equal(as.vector(as.array(result[[1]]$labels)), 1L)
  expect_equal(as.vector(as.array(result[[1]]$area)), 400)
  expect_equal(as.vector(as.array(result[[1]]$iscrowd)), FALSE)

  expect_equal(result[[1]]$image_height, 100L)
  expect_equal(result[[1]]$image_width, 100L)
})

test_that("target_transform_sahi_crop clips partially overlapping boxes", {

  y <- list(
    boxes = torch_tensor(
      matrix(c(10, 10, 30, 30), nrow = 1, byrow = TRUE)
    ),
    labels = torch_tensor(c(1L)),
    area = torch_tensor(c(400))
  )

  result <- target_transform_sahi_crop(
    y,
    list(
      list(
        top = 0,
        left = 0,
        height = 20,
        width = 20
      )
    )
  )

  expect_equal(
    as.array(result[[1]]$boxes),
    matrix(c(10, 10, 20, 20), nrow = 1)
  )

  expect_equal(as.vector(as.array(result[[1]]$labels)), 1L)
  expect_equal(as.vector(as.array(result[[1]]$area)), 100)
})

test_that("target_transform_sahi_crop removes boxes outside crop", {

  y <- list(
    boxes = torch_tensor(
      matrix(c(30, 30, 40, 40), nrow = 1, byrow = TRUE)
    ),
    labels = torch_tensor(c(1L))
  )

  result <- target_transform_sahi_crop(
    y,
    list(
      list(
        top = 0,
        left = 0,
        height = 20,
        width = 20
      )
    )
  )

  expect_equal(as.integer(result[[1]]$boxes$size(1)), 0L)
  expect_equal(as.integer(result[[1]]$labels$size(1)), 0L)
})

test_that("target_transform_sahi_crop supports multiple crop windows", {

  y <- list(
    boxes = torch_tensor(
      matrix(
        c(
          10, 10, 30, 30,
          40, 40, 60, 60
        ),
        nrow = 2,
        byrow = TRUE
      )
    ),
    labels = torch_tensor(c(1L, 2L))
  )

  crops <- list(
    list(
      top = 0,
      left = 0,
      height = 25,
      width = 25
    ),
    list(
      top = 30,
      left = 30,
      height = 40,
      width = 40
    )
  )

  result <- target_transform_sahi_crop(y, crops)

  expect_length(result, 2)

  expect_equal(
    as.array(result[[1]]$boxes),
    matrix(c(10, 10, 25, 25), nrow = 1)
  )

  expect_equal(
    as.vector(as.array(result[[1]]$labels)),
    1L
  )

  expect_equal(
    as.array(result[[2]]$boxes),
    matrix(c(10, 10, 30, 30), nrow = 1)
  )

  expect_equal(
    as.vector(as.array(result[[2]]$labels)),
    2L
  )
})

test_that("target_transform_sahi_crop filters boxes below minimum area", {

  y <- list(
    boxes = torch_tensor(
      matrix(c(10, 10, 15, 15), nrow = 1, byrow = TRUE)
    ),
    labels = torch_tensor(c(1L))
  )

  result <- target_transform_sahi_crop(
    y,
    list(
      list(
        top = 0,
        left = 0,
        height = 20,
        width = 20
      )
    ),
    min_area = 30
  )

  expect_equal(as.integer(result[[1]]$boxes$size(1)), 0L)
  expect_equal(as.integer(result[[1]]$labels$size(1)), 0L)
})

test_that("target_transform_sahi_crop recomputes area after clipping", {

  y <- list(
    boxes = torch_tensor(
      matrix(c(0, 0, 50, 50), nrow = 1)
    ),
    labels = torch_tensor(c(1L)),
    area = torch_tensor(c(2500))
  )

  result <- target_transform_sahi_crop(
    y,
    list(
      list(
        top = 0,
        left = 0,
        height = 20,
        width = 20
      )
    )
  )

  expect_equal(
    as.vector(as.array(result[[1]]$area)),
    400
  )
})

test_that("target_transform_sahi_crop translates coordinates into crop frame", {

  y <- list(
    boxes = torch_tensor(
      matrix(c(40, 40, 60, 60), nrow = 1)
    ),
    labels = torch_tensor(c(1L))
  )

  result <- target_transform_sahi_crop(
    y,
    list(
      list(
        top = 30,
        left = 30,
        height = 40,
        width = 40
      )
    )
  )

  expect_equal(
    as.array(result[[1]]$boxes),
    matrix(c(10, 10, 30, 30), nrow = 1)
  )
})