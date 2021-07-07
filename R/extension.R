has_ops <- function() {
  not_implemented_error("has_ops() Not implemented yet. https://github.com/pytorch/vision/blob/b266c2f1a5c10f5caf22f5aea7418acc392a5075/torchvision/extension.py#L60")
}

assert_has_ops <- function() {
  if(!has_ops()) {
    runtime_error(
      "Couldn't load custom C++ ops. This can happen if your torch and torchvision
       versions are incompatible, or if you had errors while compiling torchvision
       from source. Please reinstall torchvision so that it matches your torch install."
    )
  }
}
