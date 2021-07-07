x1 = torch::torch_ones(5)
y1 = torch::torch_ones(5)
x2 = x1 + 1
y2 = y1 + 1
xyxy = torch::torch_stack(list(x1,y1,x2,y2))$transpose(2,1)

test_that("box_cxcywh_to_xyxy box_xyxy_to_cxcywh box_xywh_to_xyxy box_xyxy_to_xywh", {
  # from xyxy
  cxcywh <- box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(xyxy))
  xywh <- box_xywh_to_xyxy(box_xyxy_to_xywh(xyxy))

  expect_tensor(cxcywh)
  expect_tensor(xywh)
  expect_equal(torch::as_array(cxcywh), torch::as_array(xyxy))
  expect_equal(torch::as_array(xywh), torch::as_array(xyxy))
})

