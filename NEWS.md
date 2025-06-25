# torchvision (development version)

- fix `transform_rotation` wrongly using w x w for image size (@114, cregouby)
- `tensor_image_display` and `tensor_image_browse` now accept all tensor_image dtypes. (#115, @cregouby) 
- fix `transform_affine` help to remove confusion with `transforme_random_affine` help (#116, @cregouby)
- add message translation in french (#112, @cregouby)
- Added the Fashion-MNIST dataset. (#148, @koshtiakanksha)
- Added 3 EuroSAT datasets (#126 @cregouby)
- Added `qmnist_dataset()` – a dataset loader for the QMNIST dataset (#153, @DerrickUnleashed)
- Added `emnist_dataset()` – a dataset loader for the EMNIST dataset. (#152, @DerrickUnleashed)
- Added `fgvc_aircraft_dataset()` – a dataset loader for the FGCV-Aircraft dataset. (#156, @DerrickUnleashed)
- Added support for the MS COCO dataset. (#161, @koshtiakanksha)
- add tiff image support to `folder_dataset()` (#169, @cregouby)
- Added support for `annotation_level = "all"` in `fgvc_aircraft_dataset()` (#168, @DerrickUnleashed)
- Add `nms()`and `batched_nms()` Non-Maximum Suppression, and bounding-box manipulation via `box_convert()` (#40, @Athospd)
- Added `caltech101_dataset()` and `caltech256_dataset()` – for the Caltech 101 and 256 datasets. (#158, @DerrickUnleashed)
- Added `fer_dataset()` – a dataset loader for the FER-2013 dataset. (#154, @DerrickUnleashed)
- Added `flowers102_dataset()` – a dataset loader for the Flowers102 dataset. (#157, @DerrickUnleashed)

# torchvision 0.6.0

- Remove again dependency on `zip::unzip` added in version 0.4.0. (#89)
- Improve performance on `tinyimagenet-alexnet` example (#90, @statist-bhfz)
- Updated URL of downloaded resources to use the new torch CDN. (#109)

# torchvision 0.5.1

- Remove usage of `torch_lstsq` that was removed in torch v0.10.0

# torchvision 0.5.0

-   Bugs fixed in `transform_adjust_hue()` and `transform_linear_transformation()` (#72, #73, @sebffischer)
-   add `draw_bounding_boxes()` , `draw_segmentation_masks()` and `draw_keypoints()` on top of image tensors, and add a convenience `tensor_image_browse()` and `tensor_image_display()` functions to visualize image tensors respectively in browser or in X11 device (#80, @cregouby)
-   Added the InceptionV3 model. (#82)

# torchvision 0.4.1

-   Implemented MobileNetV2 (#60)
-   Improved vignettes so they use `nnf_cross_entropy` for numerical stability. (#61)
-   Implement the full list of ResNet model family (#66, @cregouby)
-   Improved how datasets and models are downloaded by using a large timeout by default and downloading to temporary file to avoid hard to debug errors when the files are corrupt. (#67)

# torchvision 0.4.0

-   Added a dependency on `zip` to `zip::unzip` the tinyimagenet dataset.
-   Removed all usages of `torch::enumerate()` from docs and tests in favor of `coro::loop()` (#57)
-   Fixed non-namespaced calls to `torch`. (#58)

# torchvision 0.3.0

-   Use a self hosted version of the MNIST dataset to avoid frequent download failures. (#48)
-   Fix `torch_arange` calls after breaking change in `torch`. (#47)
-   Fix bug in `transform_resize` when passing `size` with length 1. (#49)

# torchvision 0.2.0

-   Fixed bugs in `transform_rotate`. (#31)
-   Fixed bugs in `transform_random_affine` and `transform_affine` (#32)
-   Added VGG model (#35)

# torchvision 0.1.0

-   Added a `NEWS.md` file to track changes to the package.
