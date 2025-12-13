# Script to create RF100 dataset catalog
# This creates a comprehensive catalog of all 39 RF100 datasets

# Biology Collection (9 datasets)
biology <- data.frame(
  collection = "biology",
  dataset = c("stomata_cell", "blood_cell", "parasite", "cell",
              "bacteria", "cotton_desease", "mitosis", "phage", "liver_desease"),
  description = c(
    "Stomata cells for plant biology research",
    "Blood cell detection (RBC, WBC, platelets)",
    "Parasite detection in microscopy images",
    "General cell detection in microscopy",
    "Bacteria detection in microscopy images",
    "Cotton plant disease detection",
    "Mitosis phase detection in cell images",
    "Bacteriophage detection in microscopy",
    "Liver disease pathology detection"
  ),
  task = "object_detection",
  train_size_mb = c(81, 6.4, 65.1, 0.3, 1.4, 62, 19, 69, 192),
  test_size_mb = c(24, 1.8, 17.9, 0.1, 2.5, 16.8, 5.3, 9.0, 55.6),
  valid_size_mb = c(12, 0.9, 9, 0.05, 0.8, 9, 2.7, 5.7, 28),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  stringsAsFactors = FALSE
)

# Medical Collection (8 datasets)
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
  train_size_mb = c(51, 2.5, 46, 62, 3.4, 50, 142, 9),
  test_size_mb = c(15, 0.6, 13, 15, 0.8, 13, 40, 2),
  valid_size_mb = c(7, 0.3, 6.5, 9, 0.5, 7, 20, 1),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
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
  train_size_mb = c(6, 8.5, 2.7, 411),
  test_size_mb = c(1.7, 2.3, 0.7, 148),
  valid_size_mb = c(0.8, 1.5, 0.4, 74),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
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
  train_size_mb = c(21.5, 8.5, 28),
  test_size_mb = c(5.7, 2.3, 7.8),
  valid_size_mb = c(1.4, 1.5, 4),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
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
  train_size_mb = c(223, 26, 290, 33),
  test_size_mb = c(60, 7.5, 85, 7),
  valid_size_mb = c(30, 3.5, 42, 5),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
  stringsAsFactors = FALSE
)

# Document Collection (6 datasets)
document <- data.frame(
  collection = "document",
  dataset = c("tweeter_post", "tweeter_profile", "document_part",
              "activity_diagram", "signature", "paper_part"),
  description = c(
    "Twitter post element detection and parsing",
    "Twitter profile element detection",
    "Document structure and part detection",
    "Activity diagram element detection",
    "Signature detection in documents",
    "Academic paper structure and part detection"
  ),
  task = "object_detection",
  train_size_mb = rep(50, 6),  # Placeholder values from source
  test_size_mb = rep(50, 6),
  valid_size_mb = rep(50, 6),
  has_train = TRUE,
  has_test = TRUE,
  has_valid = TRUE,
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

rf100_catalog$roboflow_url <- paste0("https://universe.roboflow.com/browse/", 
                                     rf100_catalog$collection)

# Estimate number of images based on size
rf100_catalog$estimated_images <- ifelse(
  rf100_catalog$total_size_mb < 10, "< 1,000",
  ifelse(rf100_catalog$total_size_mb < 50, "1,000-5,000",
  ifelse(rf100_catalog$total_size_mb < 100, "5,000-10,000",
  ifelse(rf100_catalog$total_size_mb < 200, "10,000-20,000", "> 20,000")))
)

# Reorder columns for better readability
rf100_catalog <- rf100_catalog[, c(
  "collection", "dataset", "description", "task",
  "train_size_mb", "test_size_mb", "valid_size_mb", "total_size_mb",
  "has_train", "has_test", "has_valid",
  "estimated_images", "function_name", "roboflow_url"
)]

# Sort by collection and dataset
rf100_catalog <- rf100_catalog[order(rf100_catalog$collection, rf100_catalog$dataset), ]
rownames(rf100_catalog) <- NULL

# Save as R data
usethis::use_data(rf100_catalog, overwrite = TRUE)

# Also save as CSV for easy viewing
write.csv(rf100_catalog, "inst/extdata/rf100_catalog.csv", row.names = FALSE)

# Print summary
cat("\n=== RF100 Dataset Catalog Summary ===\n")
cat("Total datasets:", nrow(rf100_catalog), "\n")
cat("Collections:", paste(unique(rf100_catalog$collection), collapse = ", "), "\n")
cat("\nDatasets by collection:\n")
print(table(rf100_catalog$collection))
cat("\nTotal size (all datasets):", round(sum(rf100_catalog$total_size_mb) / 1024, 2), "GB\n")
cat("\nCatalog saved to:\n")
cat("  - data/rf100_catalog.rda\n")
cat("  - inst/extdata/rf100_catalog.csv\n")

