# torchvision (development version)

## New features

* Added RF100 dataset catalog with `search_rf100()`, `get_rf100_catalog()`, and `list_rf100_datasets()` functions for discovering and exploring RoboFlow 100 datasets (#271, @ANAMASGARD).

## New models

* Added `model_convnext_*_detection()` for object detection, with * within tiny/small/base (#262, @ANAMASGARD).
* Added `model_convnext_*_fcn()` and `model_convnext_*_upernet()` for semantic segmentation, with * within tiny/small/base (#265, @ANAMASGARD).

## Bug fixes and improvements

* fix rf100 collection bounding-box now consider the correct native COCO format being 'xywh' (#272)
* Remove `.getbatch` method from MNIST as it is providing inconsistent tensor dimensions with `.getitem` due 
to non-vectorized `transform_` operations (#264)

# torchvision 0.8.0

## New datasets

* Added `lfw_people_dataset()` and `lfw_pairs_dataset()` for loading Labelled Faces in the Wild (LFW) datasets (@DerrickUnleashed, #203).
* Added `places365_dataset()`for loading the Places365 dataset (@koshtiakanksha, #196).
* Added `pascal_segmentation_dataset()`, and `pascal_detection_dataset()` for loading the Pascal Visual Object Classes datasets (@DerrickUnleashed, #209).
* Added `whoi_plankton_dataset()`, `whoi_small_plankton_dataset()`, and  `whoi_small_coral_dataset()` (@cregouby, #236).
* Added `rf100_document_collection()`, `rf100_medical_collection()`, `rf100_biology_collection()`, `rf100_damage_collection()`, `rf100_infrared_collection()`, 
  and `rf100_underwater_collection()` . Those are collection of datasets from RoboFlow 100 under the same 
  thematic, for a total of 35 datasets (@koshtiakanksha, @cregouby, #239).
* Added `rf100_peixos_segmentation_dataset()`.  (@koshtiakanksha, @cregouby, #250).

## New models

* Added `model_maxvit()` for MaxViT: Multi-Axis Vision Transformer (#229, @koshtiakanksha).
* Added `model_facenet_pnet()`, `model_facenet_rnet()`, and `model_facenet_onet()` for Facenet MTCNN face detection models. (@DerrickUnleashed, #227)
* Added `model_mtcnn()` and `model_inception_resnet_v1()` models for face detection and recognition. (@DerrickUnleashed, #217)
* Added `model_mobilenet_v3_large()` and `model_mobilenet_v3_small()` models for efficient image classification. (@DerrickUnleashed, #237)
* Added 8 of the `model_convnext_()` family models for image classification, thanks to @horlar1 contribution. (@cregouby, #251)
* Added 2 `model_fasterrcnn_resnet50_()` models and 2 `model_fasterrcnn_mobilenet_v3_large_()` for object detection. (@koshtiakanksha, #251)


## New features

* Added `imagenet_label()` and `imagenet_classes()` for ImageNet classes resolution (#229, @koshtiakanksha).
* `base_loader()` now accept URLs (@cregouby, #246).
* `draw_segmentation_masks()` now accepts semantic segmentation models torch_float() output. (@cregouby #247) 
* MNIST datasets and Roboflow 100 collections now have a `.getbatch` attached method (@cregouby #255)

## Bug fixes and improvements

* Switch pre 0.5.0 models to their `/v2/` URL in torch-cdn.mlverse.org. (#215)
* Models are now separated in the documentation by tasks between classification, object detection, and semantic segmentation models (@cregouby, #247)
* Breaking Change : Refactoring of `coco_*` dataset family now provides each `item$x` being an image array (for consistency with other datasets). 
You can use `transform = transform_to_tensor` to restore the previous x output to be a `torch_tensor()`.
* `transform_` are now documented into 3 different categories: unitary transformations, random transformations and combining transformations. (@cregouby, #250)
* Deprecation : `emnist_dataset` is deprecated in favor of `emnist_collection()` (@cregouby, #260).

# torchvision 0.7.0

## New datasets

* Added `fashion_mnist_dataset()` for loading the Fashion-MNIST dataset (@koshtiakanksha, #148).
* Added `eurosat_dataset()`, `eurosat_all_bands_dataset()`, and `eurosat100_dataset()` for loading RGB, all-band, and small-subset variants of the EuroSAT dataset (@cregouby, #126).
* Added `qmnist_dataset()` for loading the QMNIST dataset (@DerrickUnleashed, #153).
* Added `emnist_dataset()` for loading the EMNIST dataset (@DerrickUnleashed, #152).
* Added `fgvc_aircraft_dataset()` for loading the FGVC-Aircraft dataset (@DerrickUnleashed, #156).
* Added `coco_detection_dataset()` and `coco_caption_dataset()` for loading the MS COCO detection and captions datasets (@koshtiakanksha, #161, #172).
* Added `caltech101_dataset()` and `caltech256_dataset()` for loading the Caltech 101 and 256 datasets (@DerrickUnleashed, #158).
* Added `fer_dataset()` for loading the FER-2013 dataset (@DerrickUnleashed, #154).
* Added `flowers102_dataset()` for loading the Flowers102 dataset (@DerrickUnleashed, #157).
* Added `flickr8k_dataset()` and `flickr30k_dataset()` for loading the Flickr8k and Flickr30k datasets (@DerrickUnleashed, #159).
* Added `oxfordiiitpet_dataset()`, `oxfordiiitpet_binary_dataset()`, and `oxfordiiitpet_segmentation_dataset()` for loading the Oxford-IIIT Pet datasets (@DerrickUnleashed, #162).
* Added `rf100_document_collection()`, `rf100_underwater_collection()`, `rf100_medical_collection()`, `rf100_biology_collection()`, and `rf100_peixos_segmentation_dataset()` for loading Roboflow 100 datasets (@koshtiakanksha, #239).

## New models

* Added EfficientNet model family (B0–B7) – scalable CNNs for image classification. (#166, @koshtiakanksha)
* Added EfficientNetV2 model family (V2-S/M/L) – improved EfficientNet models for faster training. (#166, @koshtiakanksha)
* Added `model_vit_b_16()`, `model_vit_b_32()`, `model_vit_l_16()`, `model_vit_l_32()`, and `model_vit_h_14()` for loading Vision Transformer models (@DerrickUnleashed, #202).

## New features

* `tensor_image_display()` and `tensor_image_browse()` now accept all `tensor_image` dtypes (@cregouby, #115).
* `draw_bounding_boxes()` and `draw_segmentation_masks()` now accept `image_with_bounding_box` and `image_with_segmentation_mask` inputs which are 
  the default items class for respectively detection datasets and segmentation datasets (@koshtiakanksha, #175).
* `fgvc_aircraft_dataset()` gains support for `annotation_level = "all"` (@DerrickUnleashed, #168).
* `folder_dataset()` now supports TIFF image formats (@cregouby, #169).
* New `nms()` and `batched_nms()` functions provide Non-Maximum Suppression utilities. Added `box_convert()` to convert between bounding box formats (@Athospd, #40).

## Minor bug fixes and improvements

* `transform_rotation()` now correctly uses width × height for image size instead of width × width (@cregouby, #114).
* Clarified documentation for `transform_affine()` to reduce confusion with `transform_random_affine()` (@cregouby, #116).
* Added French translations for message outputs (@cregouby, #112).

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
