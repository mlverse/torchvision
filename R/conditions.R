type_error <- function(msg, env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(msg), .envir = env), class = "type_error")
}

value_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torchvision")[[1]], .envir = env), class = "value_error")
}

runtime_error <- function(msg, env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(msg), .envir = env), class = "runtime_error")
}

not_implemented_error <- function(msg, env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(msg), .envir = env), class = "not_implemented_error")
}

