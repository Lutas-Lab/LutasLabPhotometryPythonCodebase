args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Expected input CSV, time CSV, output directory, formula, and core count")
}

input_path <- args[[1]]
time_path <- args[[2]]
output_dir <- args[[3]]
formula_text <- args[[4]]
cores <- max(1L, as.integer(args[[5]]))

if (!requireNamespace("fastFMM", quietly = TRUE)) {
  stop("The R package fastFMM is not installed")
}

data <- read.csv(input_path, check.names = FALSE, stringsAsFactors = FALSE)
time <- read.csv(time_path, check.names = FALSE)$time
for (name in intersect(c("mouse", "session", "group", "condition"), names(data))) {
  data[[name]] <- factor(data[[name]])
}

n_timepoints <- sum(startsWith(names(data), "outcome."))
if (n_timepoints != length(time)) {
  stop("Outcome column count does not match the saved time grid")
}
nknots <- max(3L, min(as.integer(round(n_timepoints / 2)), 35L))

fit <- fastFMM::fui(
  formula = stats::as.formula(formula_text),
  data = data,
  family = "gaussian",
  var = TRUE,
  analytic = TRUE,
  parallel = cores > 1L,
  silent = FALSE,
  nknots_min = nknots,
  nknots_min_cov = min(nknots, 35L),
  num_cores = cores,
  REs = TRUE
)

saveRDS(fit, file.path(output_dir, "flmm_fit.rds"))

estimates <- fit$betaHat
terms <- rownames(estimates)
if (is.null(terms)) terms <- paste0("term_", seq_len(nrow(estimates)))
rows <- vector("list", nrow(estimates))
for (index in seq_len(nrow(estimates))) {
  standard_error <- sqrt(diag(fit$betaHat.var[, , index]))
  critical <- if (length(fit$qn) == 1) fit$qn else fit$qn[[index]]
  rows[[index]] <- data.frame(
    term = terms[[index]],
    time = time,
    estimate = estimates[index, ],
    pointwise_lower = estimates[index, ] - 2 * standard_error,
    pointwise_upper = estimates[index, ] + 2 * standard_error,
    joint_lower = estimates[index, ] - critical * standard_error,
    joint_upper = estimates[index, ] + critical * standard_error,
    stringsAsFactors = FALSE
  )
}
write.csv(
  do.call(rbind, rows),
  file.path(output_dir, "flmm_coefficients.csv"),
  row.names = FALSE
)

if (!is.null(fit$randEff)) {
  random_effects <- fit$randEff
  dimensions <- dim(random_effects)
  if (length(dimensions) == 3L) {
    if (dimensions[[2]] != 1L) {
      stop("Expected one mouse random-intercept column")
    }
    random_effects <- random_effects[, 1, , drop = FALSE]
    dim(random_effects) <- c(dimensions[[1]], dimensions[[3]])
  }
  if (length(dim(random_effects)) != 2L) {
    stop("Unexpected fastFMM random-effect dimensions")
  }
  mouse_names <- dimnames(fit$randEff)[[1]]
  if (is.null(mouse_names) || length(mouse_names) != nrow(random_effects)) {
    mouse_names <- levels(data$mouse)
  }
  if (length(mouse_names) != nrow(random_effects)) {
    stop("Could not match random effects to mouse levels")
  }
  random_rows <- vector("list", nrow(random_effects))
  for (mouse_index in seq_len(nrow(random_effects))) {
    random_rows[[mouse_index]] <- data.frame(
      mouse = mouse_names[[mouse_index]],
      time = time,
      deviation = as.numeric(random_effects[mouse_index, ]),
      stringsAsFactors = FALSE
    )
  }
  write.csv(
    do.call(rbind, random_rows),
    file.path(output_dir, "flmm_mouse_random_effects.csv"),
    row.names = FALSE
  )
}
write.csv(
  data.frame(
    fastFMM_version = as.character(utils::packageVersion("fastFMM")),
    R_version = R.version.string,
    formula = formula_text,
    cores = cores,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "flmm_backend_info.csv"),
  row.names = FALSE
)
