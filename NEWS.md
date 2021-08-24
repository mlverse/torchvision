# torchvision (development version)

- Implemented MobileNetV2 (#60)
- Improved vignettes so they use `nnf_cross_entropy` for numerical stability. (#61)

# torchvision 0.4.0

- Added a dependency on `zip` to `zip::unzip` the tinyimagenet dataset.
- Removed all usages of `torch::enumerate()` from docs and tests in favor of `coro::loop()` (#57)
- Fixed non-namespaced calls to `torch`. (#58)

# torchvision 0.3.0

- Use a self hosted version of the MNIST dataset to avoid frequent download failures. (#48)
- Fix `torch_arange` calls after breaking change in `torch`. (#47)
- Fix bug in `transform_resize` when passing `size` with length 1. (#49)

# torchvision 0.2.0

* Fixed bugs in `transform_rotate`. (#31)
* Fixed bugs in `transform_random_affine` and `transform_affine` (#32)
* Added VGG model (#35)

# torchvision 0.1.0

* Added a `NEWS.md` file to track changes to the package.
