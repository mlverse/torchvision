% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dataset-places365.R
\name{places365_dataset}
\alias{places365_dataset}
\alias{places365_dataset_large}
\title{Places365 Dataset}
\usage{
places365_dataset(
  root = tempdir(),
  split = c("train", "val", "test"),
  transform = NULL,
  target_transform = NULL,
  download = FALSE,
  loader = magick_loader
)

places365_dataset_large(
  root = tempdir(),
  split = c("train", "val", "test"),
  transform = NULL,
  target_transform = NULL,
  download = FALSE,
  loader = magick_loader
)
}
\arguments{
\item{root}{Root directory for dataset storage. The dataset will be stored under \verb{root/<dataset-name>}. Defaults to \code{tempdir()}.}

\item{split}{One of \code{"train"}, \code{"val"}, or \code{"test"}.}

\item{transform}{Optional. A function that takes an image and returns a transformed version (e.g., normalization, cropping).}

\item{target_transform}{Optional. A function that transforms the label.}

\item{download}{Logical. If TRUE, downloads the dataset to \verb{root/}. If the dataset is already present, download is skipped.}

\item{loader}{A function to load an image given its path. Defaults to
\code{\link[=magick_loader]{magick_loader()}}, which uses the \code{{magick}} package.}
}
\value{
A torch dataset of class \code{places365_dataset}. Each element is a named
list with:
\itemize{
\item \code{x}: the image as loaded (or transformed if \code{transform} is set).
\item \code{y}: the integer class label. For the \code{test} split, no labels are available
and \code{y} will always be \code{NA}.
}
}
\description{
Loads the MIT Places365 dataset for scene classification.
}
\details{
The dataset provides three splits: \code{"train"}, \code{"val"}, and \code{"test"}.
Folder structure and image layout on disk are handled internally by the loader.

This function downloads and prepares the smaller 256x256 image version (~30 GB).
For the high-resolution variant (~160 GB), use \code{\link[=places365_dataset_large]{places365_dataset_large()}}.
Note that images in the large version come in varying sizes, so resizing may be
needed before batching.

The \code{test} split corresponds to the \emph{private} evaluation set used in the
Places365 challenge. Annotation files are not publicly released, so only the
images are provided.
}
\section{Functions}{
\itemize{
\item \code{places365_dataset_large()}: High resolution variant (~160 GB).

}}
\examples{
\dontrun{
ds <- places365_dataset(
  split = "val",
  download = TRUE,
  transform = transform_to_tensor
)
item <- ds[1]
tensor_image_browse(item$x)

# Show class index and label
label_idx <- item$y
label_name <- ds$classes[label_idx]
cat("Label index:", label_idx, "Class name:", label_name, "\n")

dl <- dataloader(ds, batch_size = 2)
batch <- dataloader_next(dataloader_make_iter(dl))
batch$x

ds_large <- places365_dataset_large(
  split = "val",
  download = TRUE,
  transform = . \%>\% transform_to_tensor() \%>\% transform_resize(c(256, 256))
)
dl <- torch::dataloader(dataset = ds_large, batch_size = 2)
batch <- dataloader_next(dataloader_make_iter(dl))
batch$x
}

}
\seealso{
Other classification_dataset: 
\code{\link{caltech_dataset}},
\code{\link{cifar10_dataset}()},
\code{\link{eurosat_dataset}()},
\code{\link{fer_dataset}()},
\code{\link{fgvc_aircraft_dataset}()},
\code{\link{flowers102_dataset}()},
\code{\link{mnist_dataset}()},
\code{\link{oxfordiiitpet_dataset}()},
\code{\link{tiny_imagenet_dataset}()}
}
\concept{classification_dataset}
