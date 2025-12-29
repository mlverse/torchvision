## code to prepare `rf100_catalog` dataset goes here


# This creates a comprehensive catalog of all 39 RF100 datasets


# Biology Collection
# Image counts and dimensions from RoboFlow dataset metadata
biology <- data.frame(
  collection = "biology",
  dataset = c("stomata_cell", "blood_cell", "parasite", "cell",
              "bacteria", "cotton_desease", "mitosis", "phage", "liver_desease", "moth"),
  description = c(
    "Stomata cells for plant biology research",
    "Blood cell detection (RBC, WBC, platelets)",
    "Parasite detection in microscopy images",
    "General cell detection in microscopy",
    "Bacteria detection in microscopy images",
    "Cotton plant disease detection",
    "Mitosis phase detection in cell images",
    "Bacteriophage detection in microscopy",
    "Liver disease pathology detection",
    "Moths for agriculture, entomology, and crop pest management, particularly in Asia"
  ),
  task = "object_detection",
  num_classes = c(2L, 3L, 8L, 1L, 1L, 1L, 1L, 2L, 4L, 28L),
  num_images = c(1046, 364, 2084, 52, 580, 1742, 515, 1361, 4756, 711),
  image_width = c(640, 640, 640, 640, 640, 640, 640, 640, 640, 640),
  image_height = c(640, 480, 640, 640, 640, 640, 640, 640, 640, 640),
  train_size_mb = c(81, 6.4, 65.1, 0.3, 1.4, 62, 19, 69, 192, 12),
  test_size_mb = c(24, 1.8, 17.9, 0.1, 2.5, 16.8, 5.3, 9.0, 55.6, 43),
  valid_size_mb = c(12, 0.9, 9, 0.05, 0.8, 9, 2.7, 5.7, 28, 4.3),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  roboflow_url = c("https://universe.roboflow.com/object-detection/stomata-cells",
                   "https://universe.roboflow.com/object-detection/bccd-ouzjz",
                   "https://universe.roboflow.com/object-detection/parasites-1s07h",
                   "https://universe.roboflow.com/object-detection/cells-uyemf",
                   "https://universe.roboflow.com/object-detection/bacteria-ptywi",
                   "https://universe.roboflow.com/object-detection/cotton-plant-disease",
                   "https://universe.roboflow.com/object-detection/mitosis-gjs3g",
                   "https://universe.roboflow.com/object-detection/phages",
                   "https://universe.roboflow.com/object-detection/liver-disease",
                   "https://universe.roboflow.com/roboflow-100/pests-2xlvx"),
  stringsAsFactors = FALSE
)


# Medical Collection
medical <- data.frame(
  collection = "medical",
  dataset = c("radio_signal", "rheumatology", "knee", "abdomen_mri",
              "brain_axial_mri", "gynecology_mri", "brain_tumor", "fracture"),
  description = c(
    "Radio signal detection in medical imaging",
    "Rheumatology X-ray abnormality detection",
    "ACL and knee X-ray analysis",
    "Abdomen MRI organ and structure detection",
    "Brain axial MRI structure detection",
    "Gynecology MRI structure detection",
    "Brain tumor detection in MRI scans",
    "Bone fracture detection in X-rays"
  ),
  task = "object_detection",
  num_classes = c(2L, 12L, 1L, 1L, 2L, 3L, 3L, 4L),
  num_images = c(1320, 120, 1695, 1410, 148, 2620, 3064, 628),
  image_width = c(640, 640, 640, 640, 640, 640, 640, 640),
  image_height = c(640, 640, 640, 640, 640, 640, 640, 640),
  train_size_mb = c(51, 2.5, 46, 62, 3.4, 50, 142, 9),
  test_size_mb = c(15, 0.6, 13, 15, 0.8, 13, 40, 2),
  valid_size_mb = c(7, 0.3, 6.5, 9, 0.5, 7, 20, 1),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  roboflow_url = c("https://universe.roboflow.com/object-detection/radio-signal",
                   "https://universe.roboflow.com/object-detection/x-ray-rheumatology",
                   "https://universe.roboflow.com/object-detection/acl-x-ray",
                   "https://universe.roboflow.com/object-detection/abdomen-mri",
                   "https://universe.roboflow.com/object-detection/axial-mri",
                   "https://universe.roboflow.com/object-detection/gynecology-mri",
                   "https://universe.roboflow.com/object-detection/brain-tumor-m2pbp",
                   "https://universe.roboflow.com/object-detection/bone-fracture-7fylg"),
  stringsAsFactors = FALSE
)


# Infrared Collection (4 datasets)
infrared <- data.frame(
  collection = "infrared",
  dataset = c("thermal_dog_and_people", "solar_panel", "thermal_cheetah", "ir_object"),
  description = c(
    "Thermal imaging of dogs and people",
    "Solar panel detection in infrared/thermal imagery",
    "Thermal imaging of cheetahs for wildlife monitoring",
    "FLIR camera object detection in infrared"
  ),
  task = "object_detection",
  num_classes = c(2L, 5L, 2L, 4L),
  num_images = c(203, 384, 96, 8862),
  image_width = c(640, 640, 640, 640),
  image_height = c(512, 640, 480, 512),
  train_size_mb = c(6, 8.5, 2.7, 411),
  test_size_mb = c(1.7, 2.3, 0.7, 148),
  valid_size_mb = c(0.8, 1.5, 0.4, 74),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  roboflow_url = c("https://universe.roboflow.com/object-detection/thermal-dogs-and-people-x6ejw",
                   "https://universe.roboflow.com/object-detection/solar-panels-taxvb",
                   "https://universe.roboflow.com/object-detection/thermal-cheetah-my4dp",
                   "https://universe.roboflow.com/object-detection/flir-camera-objects"),
  stringsAsFactors = FALSE
)


# Damage Collection (3 datasets)
damage <- data.frame(
  collection = "damage",
  dataset = c("liquid_crystals", "solar_panel", "asbestos"),
  description = c(
    "4-fold defect detection in liquid crystal displays",
    "Solar panel defect and damage detection",
    "Asbestos detection for safety inspection"
  ),
  task = "object_detection",
  num_classes = c(1L, 5L, 4L),
  num_images = c(614, 384, 748),
  image_width = c(640, 640, 640),
  image_height = c(640, 640, 640),
  train_size_mb = c(21.5, 8.5, 28),
  test_size_mb = c(5.7, 2.3, 7.8),
  valid_size_mb = c(1.4, 1.5, 4),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  roboflow_url = c("https://universe.roboflow.com/object-detection/4-fold-defect",
                   "https://universe.roboflow.com/object-detection/solar-panels-taxvb",
                   "https://universe.roboflow.com/object-detection/asbestos"),
  stringsAsFactors = FALSE
)


# Underwater Collection (4 datasets)
underwater <- data.frame(
  collection = "underwater",
  dataset = c("pipes", "aquarium", "objects", "coral"),
  description = c(
    "Underwater pipe detection for infrastructure inspection",
    "Aquarium fish and species detection",
    "Underwater object detection",
    "Coral reef detection and monitoring"
  ),
  task = "object_detection",
  num_classes = c(1L, 7L, 5L, 14L),
  num_images = c(5618, 638, 8174, 716),
  image_width = c(640, 640, 640, 640),
  image_height = c(480, 640, 480, 480),
  train_size_mb = c(223, 26, 290, 33),
  test_size_mb = c(60, 7.5, 85, 7),
  valid_size_mb = c(30, 3.5, 42, 5),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  roboflow_url = c("https://universe.roboflow.com/object-detection/underwater-pipes-4ng4t",
                   "https://universe.roboflow.com/object-detection/aquarium-qlnqy",
                   "https://universe.roboflow.com/object-detection/underwater-objects-5v7p8",
                   "https://universe.roboflow.com/object-detection/coral-lwptl"),
  stringsAsFactors = FALSE
)


# Document Collection
document <- data.frame(
  collection = "document",
  dataset = c("tweeter_post", "tweeter_profile", "document_part",
              "activity_diagram", "signature", "paper_part", "currency", "wine_label"),
  description = c(
    "Twitter post element detection and parsing",
    "Twitter profile element detection",
    "Document structure and part detection",
    "Activity diagram element detection",
    "Signature detection in documents",
    "Academic paper structure and part detection",
    "Combination of Dollar Bill Detection project from Alex Hyams and Coin Counter project from Dawson Mcgee.",
    "Wine Label elements detection"
  ),
  task = "object_detection",
  num_classes = c(2L, 1L, 2L, 19L, 1L, 19L, 10L, 12L),
  num_images = c(575, 468, 2402, 2604, 1894, 2202, 789, 4642),
  image_width = c(640, 640, 640, 640, 640, 640, 640, 640),
  image_height = c(640, 640, 640, 640, 640, 640, 640, 640),
  train_size_mb = c(13, 11, 75, 130, 82, 64, 32, 103),
  test_size_mb = c(3.3, 2.9, 21, 39, 23, 18, 9, 32),
  valid_size_mb = c(2.0, 1.8, 11, 19, 12, 9.5, 5, 22),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  roboflow_url = c("https://universe.roboflow.com/object-detection/tweeter-posts",
                   "https://universe.roboflow.com/object-detection/tweeter-profile",
                   "https://universe.roboflow.com/object-detection/document-parts",
                   "https://universe.roboflow.com/object-detection/activity-diagrams-qdobr",
                   "https://universe.roboflow.com/object-detection/signatures-xc8up",
                   "https://universe.roboflow.com/object-detection/paper-parts",
                   "https://universe.roboflow.com/object-detection/currency-v4f8j",
                   "https://universe.roboflow.com/object-detection/wine-labels"),
  stringsAsFactors = FALSE
)


# Combine all catalogs
rf100_catalog <- rbind(
  biology,
  medical,
  infrared,
  damage,
  underwater,
  document
)


# Add additional metadata
rf100_catalog$total_size_mb <- rf100_catalog$train_size_mb +
  rf100_catalog$test_size_mb +
  rf100_catalog$valid_size_mb


rf100_catalog$function_name <- paste0("rf100_", rf100_catalog$collection, "_collection")


rf100_catalog$roboflow_url <- ifelse(is.na(rf100_catalog$roboflow_url),
                                     paste0("https://universe.roboflow.com/browse/", rf100_catalog$collection),
                                     rf100_catalog$roboflow_url)


# Reorder columns for better readability
rf100_catalog <- rf100_catalog[, c(
  "collection", "dataset", "description", "task", "num_classes",
  "num_images", "image_width", "image_height",
  "train_size_mb", "test_size_mb", "valid_size_mb", "total_size_mb",
  "has_train", "has_test", "has_valid",
  "function_name", "roboflow_url"
)]


# Sort by collection and dataset
rf100_catalog <- rf100_catalog[order(rf100_catalog$collection, rf100_catalog$dataset), ]
rownames(rf100_catalog) <- NULL


# Save as R data
usethis::use_data(rf100_catalog, overwrite = TRUE)


# Also save as CSV for easy viewing
write.csv(rf100_catalog, "inst/extdata/rf100_catalog.csv", row.names = FALSE)


# Summarize it
cli::cli_h1("RF100 Dataset Catalog Summary")
cli::cli_text("Total datasets: {nrow(rf100_catalog)}")
cli::cli_text("Collections: {.val {paste(unique(rf100_catalog$collection), collapse = ', ')}}")
cli::cli_h2("Datasets by collection")
cli::cli_verbatim(capture.output(print(table(rf100_catalog$collection))))
cli::cli_text("Total size (all datasets): {.strong {prettyunits::pretty_bytes(sum(rf100_catalog$total_size_mb) * 1024^2)}}")
cli::cli_h2("Catalog saved to:")
cli::cli_ul()
cli::cli_li("{.path data/rf100_catalog.rda}")
cli::cli_li("{.path inst/extdata/rf100_catalog.csv}")
