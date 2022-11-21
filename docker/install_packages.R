library(reticulate)

print(" ---------- PYTHON PATH IN RSESSION:")
print(Sys.which("python"))
print(reticulate::py_config())

reticulate::virtualenv_create("r-reticulate")
reticulate::use_virtualenv("r-reticulate", required = TRUE)
reticulate::py_install("nevergrad", pip = TRUE)
reticulate::py_install("boto3", pip = TRUE)

