#' @importFrom magrittr %>%
NULL

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
#' @param image : Tensor of shape (C x H x W) and dtype uint8.
#' @param boxes : Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
#'            the boxes are absolute coordinates with respect to the image. In other words: `0  = xmin < xmax < W` and
#'            `0  = ymin < ymax < H`.
#' @param labels : character vector containing the labels of bounding boxes.
#' @param colors : character vector containing the colors
#'            of the boxes or single color for all boxes. The color can be represented as
#'            strings e.g. "red" or "#FF00FF". By default, viridis colors are generated for boxes.
#' @param fill : If `TRUE` fills the bounding box with specified color.
#' @param width : Width of text shift to the bounding box.
#' @param font : NULL for the current font family, or a character vector of length 2 for Hershey vector fonts.
# ' The first element of the vector selects a typeface and the second element selects a style.
#' @param font_size : The requested font size in points.
#'
#' @return  torch_tensor of size (C, H, W) of dtype uint8: Image Tensor with bounding boxes plotted.
#'
#' @examples
#' \donttest{
#'   image <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
#'   x <- torch::torch_randint(low = 1, high = 160, size = c(12,1))
#'   y <- torch::torch_randint(low = 1, high = 260, size = c(12,1))
#'   boxes <- torch::torch_cat(c(x, y, x + 20, y +  10), dim = 2)
#'   bboxed <- draw_bounding_boxes(image, boxes, colors = "black", fill = TRUE)
#'   tensor_image_browse(bboxed)
#' }
#' @export
draw_bounding_boxes <- function(image,
                               boxes,
                               labels = NULL,
                               colors = NULL,
                               fill = FALSE,
                               width = 1,
                               font = c("serif", "plain"),
                               font_size = 10) {
  rlang::check_installed("magick")

  stopifnot("Image is expected to be a torch_tensor" = inherits(image, "torch_tensor"))
  stopifnot("Image is expected to be of dtype torch_uint8" = image$dtype == torch::torch_uint8())
  stopifnot("Pass individual images, not batches" = image$ndim == 3)
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
  if (!is.null(labels) && (num_boxes %% length(labels) != 0)) {
    rlang::abort(
      paste0(
        "Number of labels ",
        length(labels),
        "cannot be broadcasted on number of boxes",
        num_boxes
      )
    )
  }
  if (is.null(colors)) {
    colors <- grDevices::hcl.colors(n = num_boxes)
  }
  stopifnot("colors vector cannot be broadcasted on boxes" = num_boxes %% length(colors) == 0)

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
  graphics::rect(img_bb[, 1],
       img_bb[, 2],
       img_bb[, 3],
       img_bb[, 4],
       col = fill_col,
       border = colors)
  if (!is.null(labels)) {
    graphics::text(
      img_bb[, 1] + width,
      img_bb[, 2] + width,
      labels = labels,
      col = colors,
      vfont = font,
      cex = font_size / 10
    )
  }
  grDevices::dev.off()
  draw_tt <-
    draw %>% magick::image_data(channels = "rgb") %>% as.integer %>% torch::torch_tensor(dtype = torch::torch_uint8())
  return(draw_tt$permute(c(3, 1, 2)))
}

#' Draw segmentation masks
#'
#' Draw segmentation masks with their respective colors on top of a given RGB tensor image
#'
#' @param image : torch_tensor of shape (3, H, W) and dtype uint8.
#' @param masks : torch_tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
#' @param alpha : number between 0 and 1 denoting the transparency of the masks.
#   0 means full transparency, 1 means no transparency.
#' @param colors : character vector containing the colors
#'            of the boxes or single color for all boxes. The color can be represented as
#'            strings e.g. "red" or "#FF00FF". By default, viridis colors are generated for masks
#'
#' @return torch_tensor of shape (3, H, W) and dtype uint8 of the image with segmentation masks drawn on top.
#'
#' @examples
#'   image <- torch::torch_randint(170, 250, size = c(3, 360, 360))$to(torch::torch_uint8())
#'   mask <- torch::torch_tril(torch::torch_ones(c(360, 360)))$to(torch::torch_bool())
#'   masked_image <- draw_segmentation_masks(image, mask, alpha = 0.2)
#'   tensor_image_browse(masked_image)
#'
#' @export
draw_segmentation_masks  <-  function(image,
                                      masks,
                                      alpha = 0.8,
                                      colors = NULL) {
  rlang::check_installed("magick")
  stopifnot("`image` is expected to be a torch_tensor" = inherits(image, "torch_tensor"))
  stopifnot("`image` is expected to be of dtype torch_uint8" = image$dtype == torch::torch_uint8())
  stopifnot("Pass individual images, not batches" = image$ndim == 3)
  stopifnot("Only grayscale and RGB images are supported" = image$size(1) %in% c(1, 3))
  if (masks$ndim == 2) {
    masks <- masks$unsqueeze(1)
  }
  stopifnot("`masks` must be of shape (H, W) or (num_masks, H, W)" = masks$ndim == 3)
  stopifnot("`masks` is expected to be of dtype torch_bool" = masks$dtype == torch::torch_bool())
  stopifnot("`masks` and `image` must have the same height and width" = masks$shape[2:3] == image$shape[2:3])
  num_masks <- masks$size(1)
  if (num_masks == 0) {
    rlang::warn("masks doesn't contain any mask. No mask was drawn")
    return(image)
  }
  if (is.null(colors)) {
    colors <- grDevices::hcl.colors(n = num_masks)
  }
  stopifnot("colors vector cannot be broadcasted on masks" = num_masks %% length(colors) == 0)

  out_dtype <- torch::torch_uint8()

  color_tt <-
    colors %>% grDevices::col2rgb() %>% t() %>% torch::torch_tensor(dtype = out_dtype)

  img_to_draw <- image$detach()$clone()

  colored_mask_stack <- torch::torch_stack(lapply(
     seq(masks$size(1)),
     function(x) color_tt[x, ]$unsqueeze(2)$unsqueeze(2)$mul(masks[x:x, , ])
     ),
    dim = 1
  )
  out <- img_to_draw * (1 - alpha) + torch::torch_sum(colored_mask_stack, dim = 1) * alpha
  return(out$to(out_dtype))
}


#' Draws Keypoints
#'
#' Draws Keypoints, an object describing a body part (like rightArm or leftShoulder), on given RGB tensor image.
#' @param image : Tensor of shape (3, H, W) and dtype uint8
#' @param keypoints : Tensor of shape (N, K, 2) the K keypoints location for each of the N detected poses instance,
#         in the format c(x, y).
#' @param connectivity : Vector of pair of keypoints to be connected (currently unavailable)
#' @param colors : character vector containing the colors
#'            of the boxes or single color for all boxes. The color can be represented as
#'            strings e.g. "red" or "#FF00FF". By default, viridis colors are generated for keypoints
#' @param radius : radius of the plotted keypoint.
#' @param width : width of line connecting keypoints.
#'
#' @return Image Tensor of dtype uint8 with keypoints drawn.
#'
#' @examples
#'    image <- torch::torch_randint(190, 255, size = c(3, 360, 360))$to(torch::torch_uint8())
#'    keypoints <- torch::torch_randint(low = 60, high = 300, size = c(4, 5, 2))
#'    keypoint_image <- draw_keypoints(image, keypoints)
#'    tensor_image_browse(keypoint_image)
#'
#' @export
draw_keypoints <- function(image,
    keypoints,
    connectivity = NULL,
    colors = NULL,
    radius = 2,
    width = 3) {

  rlang::check_installed("magick")
  stopifnot("`image` is expected to be a torch_tensor" = inherits(image, "torch_tensor"))
  stopifnot("`image` is expected to be of dtype torch_uint8" = image$dtype == torch::torch_uint8())
  stopifnot("Pass individual images, not batches" = image$ndim == 3)
  stopifnot("Only RGB images are supported" = image$size(1) == 3)
  stopifnot("keypoints must be of shape (num_instances, K, 2)" = keypoints$ndim == 3)

  img_kpts <- keypoints$to(torch::torch_int64()) %>% as.array
  img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu", dtype = torch::torch_long()) %>% as.array
  draw <- png::writePNG(img_to_draw / 255) %>%
    magick::image_read() %>%
    magick::image_draw()

  for (pose in dim(img_kpts)[[1]]) {
    graphics::points(img_kpts[pose,,1],img_kpts[pose,,2], pch = ".", col = colors, cex = radius)

  }
  # TODO need R-isation and vectorisation
    # for (kpt_id, kpt_inst in enumerate(img_kpts)) {
    #     if (connectivity) {
    #         for (connection in connectivity) {
    #             start_pt_x <- kpt_inst[connection[0]][0]
    #             start_pt_y <- kpt_inst[connection[0]][1]
    #
    #             end_pt_x <- kpt_inst[connection[1]][0]
    #             end_pt_y <- kpt_inst[connection[1]][1]
    #
    #             draw$line(
    #                 ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
    #                 widt = width,
    #             )
    #         }
    #     }
    # }
  grDevices::dev.off()
  draw_tt <-
    draw %>% magick::image_data(channels = "rgb") %>% as.integer %>% torch::torch_tensor(dtype = torch::torch_uint8())

    return(draw_tt$permute(c(3, 1, 2)))
}


#' Display image tensor
#'
#' Display image tensor onto the X11 device
#' @param image `torch_tensor()` of shape (1, W, H) for grayscale image or (3, W, H) for color image,
#'   of type `torch_uint8()` to display
#' @param animate support animations in the X11 display
#'
#' @export
tensor_image_display <- function(image, animate = TRUE) {
  stopifnot("`image` is expected to be of dtype torch_uint8" = image$dtype == torch::torch_uint8())
  stopifnot("Pass individual images, not batches" = image$ndim == 3)
  stopifnot("Only grayscale and RGB images are supported" = image$size(1) %in% c(1, 3))

  img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu", dtype = torch::torch_long()) %>% as.array

  png::writePNG(img_to_draw / 255) %>% magick::image_read() %>% magick::image_display(animate = animate)

  invisible(NULL)
}


#' Display image tensor
#'
#' Display image tensor into browser
#' @param image `torch_tensor()` of shape (1, W, H) for grayscale image or (3, W, H) for color image,
#'   of type `torch_uint8()` to display
#' @param browser argument passed to [browseURL]
#'
#' @export
tensor_image_browse <- function(image, browser = getOption("browser")) {
  stopifnot("`image` is expected to be of dtype torch_uint8" = image$dtype == torch::torch_uint8())
  stopifnot("Pass individual images, not batches" = image$ndim == 3)
  stopifnot("Only grayscale and RGB images are supported" = image$size(1) %in% c(1, 3))

  img_to_draw <- image$permute(c(2, 3, 1))$to(device = "cpu", dtype = torch::torch_long()) %>% as.array

  png::writePNG(img_to_draw / 255) %>% magick::image_read() %>% magick::image_browse(browser = browser)

  invisible(NULL)
}

#' Imagenet classes
#'
#' A list of imagenet classes that can be used to decode predictions from
#' imagenet pre-trained models.
#'
#' The list has been adapted from a [Gist](https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt)
#' For examples, see inception_v3 tests.
#'
#' @export
imagenet_classes <- function() {
  c('tench, Tinca tinca',
    'goldfish, Carassius auratus',
    'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    'tiger shark, Galeocerdo cuvieri',
    'hammerhead, hammerhead shark',
    'electric ray, crampfish, numbfish, torpedo',
    'stingray',
    'cock',
    'hen',
    'ostrich, Struthio camelus',
    'brambling, Fringilla montifringilla',
    'goldfinch, Carduelis carduelis',
    'house finch, linnet, Carpodacus mexicanus',
    'junco, snowbird',
    'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
    'robin, American robin, Turdus migratorius',
    'bulbul',
    'jay',
    'magpie',
    'chickadee',
    'water ouzel, dipper',
    'kite',
    'bald eagle, American eagle, Haliaeetus leucocephalus',
    'vulture',
    'great grey owl, great gray owl, Strix nebulosa',
    'European fire salamander, Salamandra salamandra',
    'common newt, Triturus vulgaris',
    'eft',
    'spotted salamander, Ambystoma maculatum',
    'axolotl, mud puppy, Ambystoma mexicanum',
    'bullfrog, Rana catesbeiana',
    'tree frog, tree-frog',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    'loggerhead, loggerhead turtle, Caretta caretta',
    'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
    'mud turtle',
    'terrapin',
    'box turtle, box tortoise',
    'banded gecko',
    'common iguana, iguana, Iguana iguana',
    'American chameleon, anole, Anolis carolinensis',
    'whiptail, whiptail lizard',
    'agama',
    'frilled lizard, Chlamydosaurus kingi',
    'alligator lizard',
    'Gila monster, Heloderma suspectum',
    'green lizard, Lacerta viridis',
    'African chameleon, Chamaeleo chamaeleon',
    'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
    'African crocodile, Nile crocodile, Crocodylus niloticus',
    'American alligator, Alligator mississipiensis',
    'triceratops',
    'thunder snake, worm snake, Carphophis amoenus',
    'ringneck snake, ring-necked snake, ring snake',
    'hognose snake, puff adder, sand viper',
    'green snake, grass snake',
    'king snake, kingsnake',
    'garter snake, grass snake',
    'water snake',
    'vine snake',
    'night snake, Hypsiglena torquata',
    'boa constrictor, Constrictor constrictor',
    'rock python, rock snake, Python sebae',
    'Indian cobra, Naja naja',
    'green mamba',
    'sea snake',
    'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
    'diamondback, diamondback rattlesnake, Crotalus adamanteus',
    'sidewinder, horned rattlesnake, Crotalus cerastes',
    'trilobite',
    'harvestman, daddy longlegs, Phalangium opilio',
    'scorpion',
    'black and gold garden spider, Argiope aurantia',
    'barn spider, Araneus cavaticus',
    'garden spider, Aranea diademata',
    'black widow, Latrodectus mactans',
    'tarantula',
    'wolf spider, hunting spider',
    'tick',
    'centipede',
    'black grouse',
    'ptarmigan',
    'ruffed grouse, partridge, Bonasa umbellus',
    'prairie chicken, prairie grouse, prairie fowl',
    'peacock',
    'quail',
    'partridge',
    'African grey, African gray, Psittacus erithacus',
    'macaw',
    'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
    'lorikeet',
    'coucal',
    'bee eater',
    'hornbill',
    'hummingbird',
    'jacamar',
    'toucan',
    'drake',
    'red-breasted merganser, Mergus serrator',
    'goose',
    'black swan, Cygnus atratus',
    'tusker',
    'echidna, spiny anteater, anteater',
    'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
    'wallaby, brush kangaroo',
    'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
    'wombat',
    'jellyfish',
    'sea anemone, anemone',
    'brain coral',
    'flatworm, platyhelminth',
    'nematode, nematode worm, roundworm',
    'conch',
    'snail',
    'slug',
    'sea slug, nudibranch',
    'chiton, coat-of-mail shell, sea cradle, polyplacophore',
    'chambered nautilus, pearly nautilus, nautilus',
    'Dungeness crab, Cancer magister',
    'rock crab, Cancer irroratus',
    'fiddler crab',
    'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
    'American lobster, Northern lobster, Maine lobster, Homarus americanus',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
    'crayfish, crawfish, crawdad, crawdaddy',
    'hermit crab',
    'isopod',
    'white stork, Ciconia ciconia',
    'black stork, Ciconia nigra',
    'spoonbill',
    'flamingo',
    'little blue heron, Egretta caerulea',
    'American egret, great white heron, Egretta albus',
    'bittern',
    'crane',
    'limpkin, Aramus pictus',
    'European gallinule, Porphyrio porphyrio',
    'American coot, marsh hen, mud hen, water hen, Fulica americana',
    'bustard',
    'ruddy turnstone, Arenaria interpres',
    'red-backed sandpiper, dunlin, Erolia alpina',
    'redshank, Tringa totanus',
    'dowitcher',
    'oystercatcher, oyster catcher',
    'pelican',
    'king penguin, Aptenodytes patagonica',
    'albatross, mollymawk',
    'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
    'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
    'dugong, Dugong dugon',
    'sea lion',
    'Chihuahua',
    'Japanese spaniel',
    'Maltese dog, Maltese terrier, Maltese',
    'Pekinese, Pekingese, Peke',
    'Shih-Tzu',
    'Blenheim spaniel',
    'papillon',
    'toy terrier',
    'Rhodesian ridgeback',
    'Afghan hound, Afghan',
    'basset, basset hound',
    'beagle',
    'bloodhound, sleuthhound',
    'bluetick',
    'black-and-tan coonhound',
    'Walker hound, Walker foxhound',
    'English foxhound',
    'redbone',
    'borzoi, Russian wolfhound',
    'Irish wolfhound',
    'Italian greyhound',
    'whippet',
    'Ibizan hound, Ibizan Podenco',
    'Norwegian elkhound, elkhound',
    'otterhound, otter hound',
    'Saluki, gazelle hound',
    'Scottish deerhound, deerhound',
    'Weimaraner',
    'Staffordshire bullterrier, Staffordshire bull terrier',
    'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
    'Bedlington terrier',
    'Border terrier',
    'Kerry blue terrier',
    'Irish terrier',
    'Norfolk terrier',
    'Norwich terrier',
    'Yorkshire terrier',
    'wire-haired fox terrier',
    'Lakeland terrier',
    'Sealyham terrier, Sealyham',
    'Airedale, Airedale terrier',
    'cairn, cairn terrier',
    'Australian terrier',
    'Dandie Dinmont, Dandie Dinmont terrier',
    'Boston bull, Boston terrier',
    'miniature schnauzer',
    'giant schnauzer',
    'standard schnauzer',
    'Scotch terrier, Scottish terrier, Scottie',
    'Tibetan terrier, chrysanthemum dog',
    'silky terrier, Sydney silky',
    'soft-coated wheaten terrier',
    'West Highland white terrier',
    'Lhasa, Lhasa apso',
    'flat-coated retriever',
    'curly-coated retriever',
    'golden retriever',
    'Labrador retriever',
    'Chesapeake Bay retriever',
    'German short-haired pointer',
    'vizsla, Hungarian pointer',
    'English setter',
    'Irish setter, red setter',
    'Gordon setter',
    'Brittany spaniel',
    'clumber, clumber spaniel',
    'English springer, English springer spaniel',
    'Welsh springer spaniel',
    'cocker spaniel, English cocker spaniel, cocker',
    'Sussex spaniel',
    'Irish water spaniel',
    'kuvasz',
    'schipperke',
    'groenendael',
    'malinois',
    'briard',
    'kelpie',
    'komondor',
    'Old English sheepdog, bobtail',
    'Shetland sheepdog, Shetland sheep dog, Shetland',
    'collie',
    'Border collie',
    'Bouvier des Flandres, Bouviers des Flandres',
    'Rottweiler',
    'German shepherd, German shepherd dog, German police dog, alsatian',
    'Doberman, Doberman pinscher',
    'miniature pinscher',
    'Greater Swiss Mountain dog',
    'Bernese mountain dog',
    'Appenzeller',
    'EntleBucher',
    'boxer',
    'bull mastiff',
    'Tibetan mastiff',
    'French bulldog',
    'Great Dane',
    'Saint Bernard, St Bernard',
    'Eskimo dog, husky',
    'malamute, malemute, Alaskan malamute',
    'Siberian husky',
    'dalmatian, coach dog, carriage dog',
    'affenpinscher, monkey pinscher, monkey dog',
    'basenji',
    'pug, pug-dog',
    'Leonberg',
    'Newfoundland, Newfoundland dog',
    'Great Pyrenees',
    'Samoyed, Samoyede',
    'Pomeranian',
    'chow, chow chow',
    'keeshond',
    'Brabancon griffon',
    'Pembroke, Pembroke Welsh corgi',
    'Cardigan, Cardigan Welsh corgi',
    'toy poodle',
    'miniature poodle',
    'standard poodle',
    'Mexican hairless',
    'timber wolf, grey wolf, gray wolf, Canis lupus',
    'white wolf, Arctic wolf, Canis lupus tundrarum',
    'red wolf, maned wolf, Canis rufus, Canis niger',
    'coyote, prairie wolf, brush wolf, Canis latrans',
    'dingo, warrigal, warragal, Canis dingo',
    'dhole, Cuon alpinus',
    'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
    'hyena, hyaena',
    'red fox, Vulpes vulpes',
    'kit fox, Vulpes macrotis',
    'Arctic fox, white fox, Alopex lagopus',
    'grey fox, gray fox, Urocyon cinereoargenteus',
    'tabby, tabby cat',
    'tiger cat',
    'Persian cat',
    'Siamese cat, Siamese',
    'Egyptian cat',
    'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
    'lynx, catamount',
    'leopard, Panthera pardus',
    'snow leopard, ounce, Panthera uncia',
    'jaguar, panther, Panthera onca, Felis onca',
    'lion, king of beasts, Panthera leo',
    'tiger, Panthera tigris',
    'cheetah, chetah, Acinonyx jubatus',
    'brown bear, bruin, Ursus arctos',
    'American black bear, black bear, Ursus americanus, Euarctos americanus',
    'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
    'sloth bear, Melursus ursinus, Ursus ursinus',
    'mongoose',
    'meerkat, mierkat',
    'tiger beetle',
    'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
    'ground beetle, carabid beetle',
    'long-horned beetle, longicorn, longicorn beetle',
    'leaf beetle, chrysomelid',
    'dung beetle',
    'rhinoceros beetle',
    'weevil',
    'fly',
    'bee',
    'ant, emmet, pismire',
    'grasshopper, hopper',
    'cricket',
    'walking stick, walkingstick, stick insect',
    'cockroach, roach',
    'mantis, mantid',
    'cicada, cicala',
    'leafhopper',
    'lacewing, lacewing fly',
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    'damselfly',
    'admiral',
    'ringlet, ringlet butterfly',
    'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
    'cabbage butterfly',
    'sulphur butterfly, sulfur butterfly',
    'lycaenid, lycaenid butterfly',
    'starfish, sea star',
    'sea urchin',
    'sea cucumber, holothurian',
    'wood rabbit, cottontail, cottontail rabbit',
    'hare',
    'Angora, Angora rabbit',
    'hamster',
    'porcupine, hedgehog',
    'fox squirrel, eastern fox squirrel, Sciurus niger',
    'marmot',
    'beaver',
    'guinea pig, Cavia cobaya',
    'sorrel',
    'zebra',
    'hog, pig, grunter, squealer, Sus scrofa',
    'wild boar, boar, Sus scrofa',
    'warthog',
    'hippopotamus, hippo, river horse, Hippopotamus amphibius',
    'ox',
    'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
    'bison',
    'ram, tup',
    'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
    'ibex, Capra ibex',
    'hartebeest',
    'impala, Aepyceros melampus',
    'gazelle',
    'Arabian camel, dromedary, Camelus dromedarius',
    'llama',
    'weasel',
    'mink',
    'polecat, fitch, foulmart, foumart, Mustela putorius',
    'black-footed ferret, ferret, Mustela nigripes',
    'otter',
    'skunk, polecat, wood pussy',
    'badger',
    'armadillo',
    'three-toed sloth, ai, Bradypus tridactylus',
    'orangutan, orang, orangutang, Pongo pygmaeus',
    'gorilla, Gorilla gorilla',
    'chimpanzee, chimp, Pan troglodytes',
    'gibbon, Hylobates lar',
    'siamang, Hylobates syndactylus, Symphalangus syndactylus',
    'guenon, guenon monkey',
    'patas, hussar monkey, Erythrocebus patas',
    'baboon',
    'macaque',
    'langur',
    'colobus, colobus monkey',
    'proboscis monkey, Nasalis larvatus',
    'marmoset',
    'capuchin, ringtail, Cebus capucinus',
    'howler monkey, howler',
    'titi, titi monkey',
    'spider monkey, Ateles geoffroyi',
    'squirrel monkey, Saimiri sciureus',
    'Madagascar cat, ring-tailed lemur, Lemur catta',
    'indri, indris, Indri indri, Indri brevicaudatus',
    'Indian elephant, Elephas maximus',
    'African elephant, Loxodonta africana',
    'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
    'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    'barracouta, snoek',
    'eel',
    'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
    'rock beauty, Holocanthus tricolor',
    'anemone fish',
    'sturgeon',
    'gar, garfish, garpike, billfish, Lepisosteus osseus',
    'lionfish',
    'puffer, pufferfish, blowfish, globefish',
    'abacus',
    'abaya',
    "academic gown, academic robe, judge's robe",
    'accordion, piano accordion, squeeze box',
    'acoustic guitar',
    'aircraft carrier, carrier, flattop, attack aircraft carrier',
    'airliner',
    'airship, dirigible',
    'altar',
    'ambulance',
    'amphibian, amphibious vehicle',
    'analog clock',
    'apiary, bee house',
    'apron',
    'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
    'assault rifle, assault gun',
    'backpack, back pack, knapsack, packsack, rucksack, haversack',
    'bakery, bakeshop, bakehouse',
    'balance beam, beam',
    'balloon',
    'ballpoint, ballpoint pen, ballpen, Biro',
    'Band Aid',
    'banjo',
    'bannister, banister, balustrade, balusters, handrail',
    'barbell',
    'barber chair',
    'barbershop',
    'barn',
    'barometer',
    'barrel, cask',
    'barrow, garden cart, lawn cart, wheelbarrow',
    'baseball',
    'basketball',
    'bassinet',
    'bassoon',
    'bathing cap, swimming cap',
    'bath towel',
    'bathtub, bathing tub, bath, tub',
    'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
    'beacon, lighthouse, beacon light, pharos',
    'beaker',
    'bearskin, busby, shako',
    'beer bottle',
    'beer glass',
    'bell cote, bell cot',
    'bib',
    'bicycle-built-for-two, tandem bicycle, tandem',
    'bikini, two-piece',
    'binder, ring-binder',
    'binoculars, field glasses, opera glasses',
    'birdhouse',
    'boathouse',
    'bobsled, bobsleigh, bob',
    'bolo tie, bolo, bola tie, bola',
    'bonnet, poke bonnet',
    'bookcase',
    'bookshop, bookstore, bookstall',
    'bottlecap',
    'bow',
    'bow tie, bow-tie, bowtie',
    'brass, memorial tablet, plaque',
    'brassiere, bra, bandeau',
    'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
    'breastplate, aegis, egis',
    'broom',
    'bucket, pail',
    'buckle',
    'bulletproof vest',
    'bullet train, bullet',
    'butcher shop, meat market',
    'cab, hack, taxi, taxicab',
    'caldron, cauldron',
    'candle, taper, wax light',
    'cannon',
    'canoe',
    'can opener, tin opener',
    'cardigan',
    'car mirror',
    'carousel, carrousel, merry-go-round, roundabout, whirligig',
    "carpenter's kit, tool kit",
    'carton',
    'car wheel',
    'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
    'cassette',
    'cassette player',
    'castle',
    'catamaran',
    'CD player',
    'cello, violoncello',
    'cellular telephone, cellular phone, cellphone, cell, mobile phone',
    'chain',
    'chainlink fence',
    'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',
    'chain saw, chainsaw',
    'chest',
    'chiffonier, commode',
    'chime, bell, gong',
    'china cabinet, china closet',
    'Christmas stocking',
    'church, church building',
    'cinema, movie theater, movie theatre, movie house, picture palace',
    'cleaver, meat cleaver, chopper',
    'cliff dwelling',
    'cloak',
    'clog, geta, patten, sabot',
    'cocktail shaker',
    'coffee mug',
    'coffeepot',
    'coil, spiral, volute, whorl, helix',
    'combination lock',
    'computer keyboard, keypad',
    'confectionery, confectionary, candy store',
    'container ship, containership, container vessel',
    'convertible',
    'corkscrew, bottle screw',
    'cornet, horn, trumpet, trump',
    'cowboy boot',
    'cowboy hat, ten-gallon hat',
    'cradle',
    'crane',
    'crash helmet',
    'crate',
    'crib, cot',
    'Crock Pot',
    'croquet ball',
    'crutch',
    'cuirass',
    'dam, dike, dyke',
    'desk',
    'desktop computer',
    'dial telephone, dial phone',
    'diaper, nappy, napkin',
    'digital clock',
    'digital watch',
    'dining table, board',
    'dishrag, dishcloth',
    'dishwasher, dish washer, dishwashing machine',
    'disk brake, disc brake',
    'dock, dockage, docking facility',
    'dogsled, dog sled, dog sleigh',
    'dome',
    'doormat, welcome mat',
    'drilling platform, offshore rig',
    'drum, membranophone, tympan',
    'drumstick',
    'dumbbell',
    'Dutch oven',
    'electric fan, blower',
    'electric guitar',
    'electric locomotive',
    'entertainment center',
    'envelope',
    'espresso maker',
    'face powder',
    'feather boa, boa',
    'file, file cabinet, filing cabinet',
    'fireboat',
    'fire engine, fire truck',
    'fire screen, fireguard',
    'flagpole, flagstaff',
    'flute, transverse flute',
    'folding chair',
    'football helmet',
    'forklift',
    'fountain',
    'fountain pen',
    'four-poster',
    'freight car',
    'French horn, horn',
    'frying pan, frypan, skillet',
    'fur coat',
    'garbage truck, dustcart',
    'gasmask, respirator, gas helmet',
    'gas pump, gasoline pump, petrol pump, island dispenser',
    'goblet',
    'go-kart',
    'golf ball',
    'golfcart, golf cart',
    'gondola',
    'gong, tam-tam',
    'gown',
    'grand piano, grand',
    'greenhouse, nursery, glasshouse',
    'grille, radiator grille',
    'grocery store, grocery, food market, market',
    'guillotine',
    'hair slide',
    'hair spray',
    'half track',
    'hammer',
    'hamper',
    'hand blower, blow dryer, blow drier, hair dryer, hair drier',
    'hand-held computer, hand-held microcomputer',
    'handkerchief, hankie, hanky, hankey',
    'hard disc, hard disk, fixed disk',
    'harmonica, mouth organ, harp, mouth harp',
    'harp',
    'harvester, reaper',
    'hatchet',
    'holster',
    'home theater, home theatre',
    'honeycomb',
    'hook, claw',
    'hoopskirt, crinoline',
    'horizontal bar, high bar',
    'horse cart, horse-cart',
    'hourglass',
    'iPod',
    'iron, smoothing iron',
    "jack-o'-lantern",
    'jean, blue jean, denim',
    'jeep, landrover',
    'jersey, T-shirt, tee shirt',
    'jigsaw puzzle',
    'jinrikisha, ricksha, rickshaw',
    'joystick',
    'kimono',
    'knee pad',
    'knot',
    'lab coat, laboratory coat',
    'ladle',
    'lampshade, lamp shade',
    'laptop, laptop computer',
    'lawn mower, mower',
    'lens cap, lens cover',
    'letter opener, paper knife, paperknife',
    'library',
    'lifeboat',
    'lighter, light, igniter, ignitor',
    'limousine, limo',
    'liner, ocean liner',
    'lipstick, lip rouge',
    'Loafer',
    'lotion',
    'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
    "loupe, jeweler's loupe",
    'lumbermill, sawmill',
    'magnetic compass',
    'mailbag, postbag',
    'mailbox, letter box',
    'maillot',
    'maillot, tank suit',
    'manhole cover',
    'maraca',
    'marimba, xylophone',
    'mask',
    'matchstick',
    'maypole',
    'maze, labyrinth',
    'measuring cup',
    'medicine chest, medicine cabinet',
    'megalith, megalithic structure',
    'microphone, mike',
    'microwave, microwave oven',
    'military uniform',
    'milk can',
    'minibus',
    'miniskirt, mini',
    'minivan',
    'missile',
    'mitten',
    'mixing bowl',
    'mobile home, manufactured home',
    'Model T',
    'modem',
    'monastery',
    'monitor',
    'moped',
    'mortar',
    'mortarboard',
    'mosque',
    'mosquito net',
    'motor scooter, scooter',
    'mountain bike, all-terrain bike, off-roader',
    'mountain tent',
    'mouse, computer mouse',
    'mousetrap',
    'moving van',
    'muzzle',
    'nail',
    'neck brace',
    'necklace',
    'nipple',
    'notebook, notebook computer',
    'obelisk',
    'oboe, hautboy, hautbois',
    'ocarina, sweet potato',
    'odometer, hodometer, mileometer, milometer',
    'oil filter',
    'organ, pipe organ',
    'oscilloscope, scope, cathode-ray oscilloscope, CRO',
    'overskirt',
    'oxcart',
    'oxygen mask',
    'packet',
    'paddle, boat paddle',
    'paddlewheel, paddle wheel',
    'padlock',
    'paintbrush',
    "pajama, pyjama, pj's, jammies",
    'palace',
    'panpipe, pandean pipe, syrinx',
    'paper towel',
    'parachute, chute',
    'parallel bars, bars',
    'park bench',
    'parking meter',
    'passenger car, coach, carriage',
    'patio, terrace',
    'pay-phone, pay-station',
    'pedestal, plinth, footstall',
    'pencil box, pencil case',
    'pencil sharpener',
    'perfume, essence',
    'Petri dish',
    'photocopier',
    'pick, plectrum, plectron',
    'pickelhaube',
    'picket fence, paling',
    'pickup, pickup truck',
    'pier',
    'piggy bank, penny bank',
    'pill bottle',
    'pillow',
    'ping-pong ball',
    'pinwheel',
    'pirate, pirate ship',
    'pitcher, ewer',
    "plane, carpenter's plane, woodworking plane",
    'planetarium',
    'plastic bag',
    'plate rack',
    'plow, plough',
    "plunger, plumber's helper",
    'Polaroid camera, Polaroid Land camera',
    'pole',
    'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
    'poncho',
    'pool table, billiard table, snooker table',
    'pop bottle, soda bottle',
    'pot, flowerpot',
    "potter's wheel",
    'power drill',
    'prayer rug, prayer mat',
    'printer',
    'prison, prison house',
    'projectile, missile',
    'projector',
    'puck, hockey puck',
    'punching bag, punch bag, punching ball, punchball',
    'purse',
    'quill, quill pen',
    'quilt, comforter, comfort, puff',
    'racer, race car, racing car',
    'racket, racquet',
    'radiator',
    'radio, wireless',
    'radio telescope, radio reflector',
    'rain barrel',
    'recreational vehicle, RV, R.V.',
    'reel',
    'reflex camera',
    'refrigerator, icebox',
    'remote control, remote',
    'restaurant, eating house, eating place, eatery',
    'revolver, six-gun, six-shooter',
    'rifle',
    'rocking chair, rocker',
    'rotisserie',
    'rubber eraser, rubber, pencil eraser',
    'rugby ball',
    'rule, ruler',
    'running shoe',
    'safe',
    'safety pin',
    'saltshaker, salt shaker',
    'sandal',
    'sarong',
    'sax, saxophone',
    'scabbard',
    'scale, weighing machine',
    'school bus',
    'schooner',
    'scoreboard',
    'screen, CRT screen',
    'screw',
    'screwdriver',
    'seat belt, seatbelt',
    'sewing machine',
    'shield, buckler',
    'shoe shop, shoe-shop, shoe store',
    'shoji',
    'shopping basket',
    'shopping cart',
    'shovel',
    'shower cap',
    'shower curtain',
    'ski',
    'ski mask',
    'sleeping bag',
    'slide rule, slipstick',
    'sliding door',
    'slot, one-armed bandit',
    'snorkel',
    'snowmobile',
    'snowplow, snowplough',
    'soap dispenser',
    'soccer ball',
    'sock',
    'solar dish, solar collector, solar furnace',
    'sombrero',
    'soup bowl',
    'space bar',
    'space heater',
    'space shuttle',
    'spatula',
    'speedboat',
    "spider web, spider's web",
    'spindle',
    'sports car, sport car',
    'spotlight, spot',
    'stage',
    'steam locomotive',
    'steel arch bridge',
    'steel drum',
    'stethoscope',
    'stole',
    'stone wall',
    'stopwatch, stop watch',
    'stove',
    'strainer',
    'streetcar, tram, tramcar, trolley, trolley car',
    'stretcher',
    'studio couch, day bed',
    'stupa, tope',
    'submarine, pigboat, sub, U-boat',
    'suit, suit of clothes',
    'sundial',
    'sunglass',
    'sunglasses, dark glasses, shades',
    'sunscreen, sunblock, sun blocker',
    'suspension bridge',
    'swab, swob, mop',
    'sweatshirt',
    'swimming trunks, bathing trunks',
    'swing',
    'switch, electric switch, electrical switch',
    'syringe',
    'table lamp',
    'tank, army tank, armored combat vehicle, armoured combat vehicle',
    'tape player',
    'teapot',
    'teddy, teddy bear',
    'television, television system',
    'tennis ball',
    'thatch, thatched roof',
    'theater curtain, theatre curtain',
    'thimble',
    'thresher, thrasher, threshing machine',
    'throne',
    'tile roof',
    'toaster',
    'tobacco shop, tobacconist shop, tobacconist',
    'toilet seat',
    'torch',
    'totem pole',
    'tow truck, tow car, wrecker',
    'toyshop',
    'tractor',
    'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',
    'tray',
    'trench coat',
    'tricycle, trike, velocipede',
    'trimaran',
    'tripod',
    'triumphal arch',
    'trolleybus, trolley coach, trackless trolley',
    'trombone',
    'tub, vat',
    'turnstile',
    'typewriter keyboard',
    'umbrella',
    'unicycle, monocycle',
    'upright, upright piano',
    'vacuum, vacuum cleaner',
    'vase',
    'vault',
    'velvet',
    'vending machine',
    'vestment',
    'viaduct',
    'violin, fiddle',
    'volleyball',
    'waffle iron',
    'wall clock',
    'wallet, billfold, notecase, pocketbook',
    'wardrobe, closet, press',
    'warplane, military plane',
    'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
    'washer, automatic washer, washing machine',
    'water bottle',
    'water jug',
    'water tower',
    'whiskey jug',
    'whistle',
    'wig',
    'window screen',
    'window shade',
    'Windsor tie',
    'wine bottle',
    'wing',
    'wok',
    'wooden spoon',
    'wool, woolen, woollen',
    'worm fence, snake fence, snake-rail fence, Virginia fence',
    'wreck',
    'yawl',
    'yurt',
    'web site, website, internet site, site',
    'comic book',
    'crossword puzzle, crossword',
    'street sign',
    'traffic light, traffic signal, stoplight',
    'book jacket, dust cover, dust jacket, dust wrapper',
    'menu',
    'plate',
    'guacamole',
    'consomme',
    'hot pot, hotpot',
    'trifle',
    'ice cream, icecream',
    'ice lolly, lolly, lollipop, popsicle',
    'French loaf',
    'bagel, beigel',
    'pretzel',
    'cheeseburger',
    'hotdog, hot dog, red hot',
    'mashed potato',
    'head cabbage',
    'broccoli',
    'cauliflower',
    'zucchini, courgette',
    'spaghetti squash',
    'acorn squash',
    'butternut squash',
    'cucumber, cuke',
    'artichoke, globe artichoke',
    'bell pepper',
    'cardoon',
    'mushroom',
    'Granny Smith',
    'strawberry',
    'orange',
    'lemon',
    'fig',
    'pineapple, ananas',
    'banana',
    'jackfruit, jak, jack',
    'custard apple',
    'pomegranate',
    'hay',
    'carbonara',
    'chocolate sauce, chocolate syrup',
    'dough',
    'meat loaf, meatloaf',
    'pizza, pizza pie',
    'potpie',
    'burrito',
    'red wine',
    'espresso',
    'cup',
    'eggnog',
    'alp',
    'bubble',
    'cliff, drop, drop-off',
    'coral reef',
    'geyser',
    'lakeside, lakeshore',
    'promontory, headland, head, foreland',
    'sandbar, sand bar',
    'seashore, coast, seacoast, sea-coast',
    'valley, vale',
    'volcano',
    'ballplayer, baseball player',
    'groom, bridegroom',
    'scuba diver',
    'rapeseed',
    'daisy',
    "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
    'corn',
    'acorn',
    'hip, rose hip, rosehip',
    'buckeye, horse chestnut, conker',
    'coral fungus',
    'agaric',
    'gyromitra',
    'stinkhorn, carrion fungus',
    'earthstar',
    'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
    'bolete',
    'ear, spike, capitulum',
    'toilet tissue, toilet paper, bathroom tissue')
}
