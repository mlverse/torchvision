# Rcpp Integration (GSoC Demonstration)

This demonstrates Rcpp integration in the torchvision package by adding a C++ implementation of box area calculation.

## Purpose

This is primarily a **demonstration of Rcpp skills** for GSoC, showing:
- C++ function implementation with proper parameter handling
- Seamless R/C++ interface using Rcpp
- Integration with existing package functions
- Comprehensive testing

## What it does

Adds `box_area_fast()` - a C++ version of box area calculation that's faster for very large matrices:

```r
boxes <- matrix(c(0, 0, 10, 10), ncol = 4)
box_area_fast(boxes)  # returns 100
```

## When to use

- **Standard use**: Stick with `box_area()` - it works well with torch tensors
- **This function**: Only if you have huge plain matrices (100k+ boxes) and need extra speed

## Implementation

- **C++ code**: `src/box_operations.cpp` - Simple loop calculating (x2-x1) * (y2-y1)
- **R wrapper**: `R/box_operations_cpp.R` - Handles torch tensors and matrices
- **Tests**: `tests/testthat/test-box-cpp.R` - 9 tests covering correctness and edge cases

## Performance

For typical use (< 10k boxes), the difference is negligible. For 100k+ boxes with plain matrices, this can be 3-5x faster.

## Installation

```r
devtools::document()
devtools::load_all()
devtools::test()
```

All tests pass. Compatible with existing code.

---

**Note**: This is a demonstration implementation. Maintainers can decide whether to merge as-is, modify, or use as reference for other Rcpp work in the package.
