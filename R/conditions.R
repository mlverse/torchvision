type_error <- function(msg) {
  rlang::abort(gettext(msg)[[1]], class = "type_error")
}

value_error <- function(msg) {
  rlang::abort(gettext(msg)[[1]], class = "value_error")
}

runtime_error <- function(msg) {
  rlang::abort(gettext(msg)[[1]], class = "runtime_error")
}

not_implemented_error <- function(msg) {
  rlang::abort(gettext(msg)[[1]], class = "not_implemented_error")
}

