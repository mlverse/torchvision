#' A simplified version of torchvision.utils.make_grid
#'
#' Arranges a batch of (image) tensors in a grid, with optional padding between
#'   images. Expects a 4d mini-batch tensor of shape (B x C x H x W).
#'
#' @param tensor tensor to arrange in grid.
#' @param scale whether to normalize (min-max-scale) the input tensor.
#' @param num_rows number of rows making up the grid (default 8).
#' @param padding amount of padding between batch images (default 2).
#' @param pad_value pixel value to use for padding.
#'
#' @export
vision_make_grid <- function(tensor,
           scale = TRUE,
           num_rows = 8,
           padding = 2,
           pad_value = 0) {

    min_max_scale <- function(x) {
      min = x$min()$item()
      max = x$max()$item()
      x$clamp_(min = min, max = max)
      x$add_(-min)$div_(max - min + 1e-5)
      x
    }
    if(scale) tensor <- min_max_scale(tensor)

    nmaps <- tensor$size(1)
    xmaps <- min(num_rows, nmaps)
    ymaps <- ceiling(nmaps / xmaps)
    height <- floor(tensor$size(3) + padding)
    width <- floor(tensor$size(4) + padding)
    num_channels <- tensor$size(2)
    grid <-
      tensor$new_full(c(num_channels, height * ymaps + padding, width * xmaps + padding),
                      pad_value)
    k <- 0

    for (y in 0:(ymaps - 1)) {
      for (x in 0:(xmaps - 1)) {
        if (k >= nmaps)
          break
        grid$narrow(
          dim = 2,
          start =  1 + torch::torch_tensor(y * height + padding, dtype = torch::torch_int64())$sum(dim = 1),
          length = height - padding
        )$narrow(
          dim = 3,
          start = 1 + torch::torch_tensor(x * width + padding, dtype = torch::torch_int64())$sum(dim = 1),
          length = width - padding
        )$copy_(tensor[k + 1, , ,])
        k <- k + 1
      }
    }

    grid
}


#' Draws bounding boxes on image.
#'
#' Draws bounding boxes on top of one image tensor
#'
#' @param image: Tensor of shape (C x H x W) and dtype uint8.
#' @param boxes: Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
#'            the boxes are absolute coordinates with respect to the image. In other words: `0  = xmin < xmax < W` and
#'            `0  = ymin < ymax < H`.
#' @param labels: character vector containing the labels of bounding boxes.
#' @param colors: character vector containing the colors
#'            of the boxes or single color for all boxes. The color can be represented as
#'            strings e.g. "red" or "#FF00FF". By default, viridis colors are generated for boxes.
#' @param fill: If `TRUE` fills the bounding box with specified color.
#' @param width: Width of text shift to the bounding box.
#' @param font: NULL for the current font family, or a character vector of length 2 for Hershey vector fonts.
# ' The first element of the vector selects a typeface and the second element selects a style.
#' @param font_size: The requested font size in points.
#'
#' @return  torch_tensorc(C, H, W)): Image Tensor of dtype uint8 with bounding boxes plotted.
#'
#' @examples
#'   image <- 1 - (torch::torch_randn(c(3, 360, 360)) / 20)
#'   x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
#'   y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
#'   boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
#'   bboxed <- draw_bounding_boxes(image, boxes, colors = "black", fill = TRUE)
#'
#' @export
draw_bounding_boxes <- function(image,
                               boxes,
                               labels = NULL,
                               colors = NULL,
                               fill = FALSE,
                               width = 1,
                               font = c("serif", "plain"),
                               font_size = 10) {
  stopifnot("Image is expected to be a torch_tensor" = inherits(image, "torch_tensor"))
  stopifnot("Image is expected to be of dtype torch_uint8" = image$dtype == torch::torch_uint8())
  stopifnot("Pass individual images, not batches" = image$dim() == 3)
  stopifnot("Only grayscale and RGB images are supported" = image$size(1) %in% c(1, 3))
  stopifnot(
    "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision$ops$box_convert to convert them" = (boxes[, 1] < boxes[, 3])$all() %>% as.logical() &&
      (boxes[, 2] < boxes[, 4])$all() %>% as.logical()
  )

  num_boxes <- boxes$shape[1]
  if (num_boxes == 0) {
    rlang::warn("boxes doesn't contain any box. No box was drawn")
    return(image)
  }
  if (!is.null(labels) && (lenght(labels) != num_boxes)) {
    rlang::abort(
      paste0(
        "Number of boxes ",
        num_boxes,
        " and labels ",
        length(labels),
        " mismatch. Please specify one label for each box."
      )
    )
  }
  if (is.null(colors)) {
    colors <- viridisLite::viridis(num_boxes)
  } else if (is.list(colors)) {
    if (length(colors) < num_boxes) {
      rlang::abort(paste0(
        "Number of colors ",
        length(colors),
        " is less than number of boxes ",
        num_boxes
      ))
    }
  }
  if (!fill) {
    fill_col <- NA
  } else {
    fill_col <- colors
  }

  if (is.null(font)) {
    vfont <- c("serif", "plain")
  } else {
    if (is.null(font_size)) {
      font_size <- 10
    }
  }
  # Handle Grayscale images
  if (image$size(1) == 1) {
    image <- image$tile(c(4, 2, 2))
  }

  img_bb <- boxes$to(torch::torch_int64()) %>% as.array
  img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu", dtype = torch::torch_long()) %>% as.array


  draw <- png::writePNG(img_to_draw / 255) %>%
    magick::image_read() %>%
    magick::image_draw()
  rect(img_bb[, 1],
       img_bb[, 2],
       img_bb[, 3],
       img_bb[, 4],
       col = fill_col,
       border = colors)
  if (!is.null(labels)) {
    text(
      img_bb[, 1] + width,
      img_bb[, 2] + width,
      labels = labels,
      col = colors,
      vfont = font,
      cex = font_size / 10
    )
  }
  dev.off()
  draw_tt <-
    draw %>% magick::image_data(channels = "rgb") %>% as.integer %>% torch::torch_tensor(dtype = torch::torch_uint8())
  return(draw_tt$permute(c(3, 1, 2)))
}

#' draw_segmentation_masks = function(
    #'     image: torch::torch_Tensor,
#'     masks: torch::torch_Tensor,
#'     alpha: float <- 0.8,
#'     colors: Optionalc(Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]) <- NULL,
#' ) -> torch::torch_Tensor:
#'
#'     """
#'     Draws segmentation masks on given RGB image.
#'     The values of the input image should be uint8 between 0 and 255.
#'     Args:
#'         image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
#'         masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
#'         alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
#'             0 means full transparency, 1 means no transparency.
#'         colors (color or list of colors, optional): List containing the colors
#'             of the masks or single color for all masks. The color can be represented as
#'             PIL strings e$g. "red" or "#FF00FF", or as RGB tuples e$g. ``(240, 10, 157)``.
#'             By default, random colors are generated for each mask.
#'     Returns:
#'         img (Tensorc(C, H, W)): Image Tensor, with segmentation masks drawn on top.
#'     """
#'
#'     if not torch::torch_jit$is_scripting() and not torch::torch_jit$is_tracing() {
#'         _log_api_usage_once(draw_segmentation_masks)
#'     if not isinstance(image, torch::torch_Tensor) {
#'         raise TypeError(f"The image must be a tensor, got list(type(image))")
#'     elif (image$dtype  = torch::torch_uint8) {
#'         raise ValueError(f"The image dtype must be uint8, got list(image$dtype)")
#'     elif (image$dim()  = 3) {
#'         raise ValueError("Pass individual images, not batches")
#'     elif (image$size()[0]  = 3) {
#'         raise ValueError("Pass an RGB image. Other Image formats are not supported")
#'     if (masks$ndim == 2) {
#'         masks <- masks[NULL, :, :]
#'     if (masks$ndim  = 3) {
#'         raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
#'     if (masks$dtype  = torch::torch_bool) {
#'         raise ValueError(f"The masks must be of dtype bool. Got list(masks$dtype)")
#'     if (masks$shape[-2:]  = image$shape[-2:]) {
#'         raise ValueError("The image and the masks must have the same height and width")
#'
#'     num_masks <- masks$size()[0]
#'     if !is.null(colors) and num_masks > len(colors) {
#'         raise ValueError(f"There are more masks (list(num_masks}) than colors ({len(colors)))")
#'
#'     if (num_masks == 0) {
#'         warnings$warn("masks doesn't contain any mask. No mask was drawn")
#'         return(image)
#'
#'     if (colors is NULL) {
#'         colors <- _generate_color_palette(num_masks)
#'
#'     if not isinstance(colors, list) {
#'         colors <- [colors]
#'     if not isinstance(colors[0], (tuple, str)) {
#'         raise ValueError("colors must be a tuple or a string, or a list thereof")
#'     if (isinstance(colors[0], tuple) and len(colors[0])  = 3) {
#'         raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")
#'
#'     out_dtype <- torch::torch_uint8
#'
#'     colors_ <- []
#'     for color in colors:
#'         if isinstance(color, str) {
#'             color <- ImageColor$getrgb(color)
#'         colors_.append(torch::torch_tensor(color, dtyp = out_dtype))
#'
#'     img_to_draw <- image$detach()$clone()
#'     # TODO: There might be a way to vectorize this
#'     for mask, color in zip(masks, colors_) {
#'         img_to_draw[:, mask] <- color[:, NULL]
#'
#'     out <- image * (1 - alpha) + img_to_draw * alpha
#'     return(out$to(out_dtype))
#'
#'
#' @torch::torch_no_grad()
#' }
#' draw_keypoints = function(
    #'     image: torch::torch_Tensor,
#'     keypoints: torch::torch_Tensor,
#'     connectivity: Optionalc(List[Tuple[int, int]]) <- NULL,
#'     colors: Optionalc(Union[str, Tuple[int, int, int]]) <- NULL,
#'     radius: int <- 2,
#'     width: int <- 3,
#' ) -> torch::torch_Tensor:
#'
#'     """
#'     Draws Keypoints on given RGB image.
#'     The values of the input image should be uint8 between 0 and 255.
#'     Args:
#'         image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
#'         keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
#'             in the format c(x, y).
#'         connectivity (Listc(Tuple[int, int]])): A List of tuple where,
#'             each tuple contains pair of keypoints to be connected.
#'         colors (str, Tuple): The color can be represented as
#'             PIL strings e$g. "red" or "#FF00FF", or as RGB tuples e$g. ``(240, 10, 157)``.
#'         radius (int): Integer denoting radius of keypoint.
#'         width (int): Integer denoting width of line connecting keypoints.
#'     Returns:
#'         img (Tensorc(C, H, W)): Image Tensor of dtype uint8 with keypoints drawn.
#'     """
#'
#'     if not torch::torch_jit$is_scripting() and not torch::torch_jit$is_tracing() {
#'         _log_api_usage_once(draw_keypoints)
#'     if not isinstance(image, torch::torch_Tensor) {
#'         raise TypeError(f"The image must be a tensor, got list(type(image))")
#'     elif (image$dtype  = torch::torch_uint8) {
#'         raise ValueError(f"The image dtype must be uint8, got list(image$dtype)")
#'     elif (image$dim()  = 3) {
#'         raise ValueError("Pass individual images, not batches")
#'     elif (image$size()[0]  = 3) {
#'         raise ValueError("Pass an RGB image. Other Image formats are not supported")
#'
#'     if (keypoints$ndim  = 3) {
#'         raise ValueError("keypoints must be of shape (num_instances, K, 2)")
#'
#'     ndarr <- image$permute(1, 2, 0)$cpu()$numpy()
#'     img_to_draw <- Image$fromarray(ndarr)
#'     draw <- ImageDraw$Draw(img_to_draw)
#'     img_kpts <- keypoints$to(torch::torch_int64)$tolist()
#'
#'     for kpt_id, kpt_inst in enumerate(img_kpts) {
#'         for inst_id, kpt in enumerate(kpt_inst) {
#'             x1 <- kpt[0] - radius
#'             x2 <- kpt[0] + radius
#'             y1 <- kpt[1] - radius
#'             y2 <- kpt[1] + radius
#'             draw$ellipse([x1, y1, x2, y2], fil = colors, outlin = NULL, widt = 0)
#'
#'         if (connectivity) {
#'             for connection in connectivity:
#'                 start_pt_x <- kpt_inst[connection[0]][0]
#'                 start_pt_y <- kpt_inst[connection[0]][1]
#'
#'                 end_pt_x <- kpt_inst[connection[1]][0]
#'                 end_pt_y <- kpt_inst[connection[1]][1]
#'
#'                 draw$line(
#'                     ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
#'                     widt = width,
#'                 )
#'
#'     return(torch::torch_from_numpy(np$array(img_to_draw))$permute(2, 0, 1)$to(dtyp = torch::torch_uint8))
#'
