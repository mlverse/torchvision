type_error <- function(msg, env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(msg), .envir = env), class = "type_error")
}

value_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torchvision"), .envir = env), class = "value_error")
}

runtime_error <- function(msg, env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(msg), .envir = env), class = "runtime_error")
}

not_implemented_error <- function(msg, env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(msg), .envir = env), class = "not_implemented_error")
}

warn <- function(..., env = rlang::caller_env()) {
  rlang::warn(glue::glue(gettext(..., domain = "R-torchvision")[[1]], .envir = env), class = "warning")
}

cli_abort <- function(..., env = rlang::caller_env()) {
  cli::cli_abort(gettext(...)[[1]], .envir = env)
}

cli_inform <- function(..., env = rlang::caller_env()) {
  cli::cli_inform(gettext(..., domain = "R-torchvision")[[1]], .envir = env)
}

stop_iteration_error <- function(..., env = rlang::caller_env()) {
  rlang::abort(glue::glue(gettext(..., domain = "R-torchvision")[[1]], .envir = env), class = "stop_iteration_error")
}

inform <- rlang::inform

deprecated <- function(..., env = rlang::caller_env()) {
  rlang::warn(gettext(..., domain = "R-torchvision")[[1]], class = "deprecated")
}
