
# torchvision <img src='https://torch.mlverse.org/css/images/hex/torchvision.png' align="right" style="width: 15%"/>

<!-- badges: start -->
[![R build status](https://github.com/mlverse/torchvision/workflows/R-CMD-check/badge.svg)](https://github.com/mlverse/torchvision/actions)
[![CRAN status](https://www.r-pkg.org/badges/version/torchvision)](https://CRAN.R-project.org/package=torchvision)
[![](https://cranlogs.r-pkg.org/badges/torchvision)](https://cran.r-project.org/package=torchvision)
<!-- badges: end -->

torchvision is an extension for [torch](https://github.com/mlverse/torch) providing image loading, transformations, common architectures for computer vision, pre-trained weights and access to commonly used datasets. 

## Installation

The CRAN release can be installed with:

```r
install.packages("torchvision")
```

You can install the development version from GitHub with:

``` r
remotes::install_github("mlverse/torchvision@main")
```

## RF100 Dataset Catalog

torchvision includes 34 datasets from the RoboFlow 100 benchmark, organized into 6 collections. Use the catalog to easily discover and search for datasets:

```r
library(torchvision)

# Search for datasets by keyword
search_rf100("solar")       # Find solar/photovoltaic datasets
search_rf100("cell")        # Find cell-related datasets
search_rf100("medical")     # Find medical imaging datasets

# Browse by collection
search_rf100(collection = "biology")    # All biology datasets
search_rf100(collection = "medical")    # All medical datasets

# View complete catalog
catalog <- get_rf100_catalog()
View(catalog)
```

### Available Collections

- **Biology** (9 datasets): Microscopy, cells, bacteria, parasites, plant diseases
- **Medical** (8 datasets): X-rays, MRI, pathology, tumor detection
- **Infrared** (4 datasets): Thermal imaging, FLIR cameras, solar panels
- **Damage** (3 datasets): Infrastructure damage, defect detection
- **Underwater** (4 datasets): Marine life, coral reefs, underwater objects
- **Document** (6 datasets): OCR, document parsing, diagrams

See `vignette("rf100-datasets")` for the complete catalog and detailed information.

### Example Usage

```r
# Search for a dataset
search_rf100("blood")

# Load the dataset
ds <- rf100_biology_collection(
  dataset = "blood_cell",
  split = "train",
  download = TRUE
)

# Visualize a sample
item <- ds[1]
boxed <- draw_bounding_boxes(item)
tensor_image_browse(boxed)
```

