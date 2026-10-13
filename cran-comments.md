## Local R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

- local macOS (aarch64-apple-darwin), R 4.6.1
- `devtools::check(remote = TRUE, manual = TRUE)` with tests enabled

The local suite passes with two warnings from self-links in the upstream Peixos archive when extracted with macOS's tar. The dataset assertions pass.

## Windows R-devel

0 errors | 0 warnings | 1 note

The note identifies a broken relative vignette link. The link is corrected in the release candidate, and the revised win-builder check is pending.

## Reverse dependencies

We checked 7 reverse dependencies against the development version containing the same runtime code as this release, comparing results with CRAN 0.9.0. There were 0 new problems and 0 failed checks.
