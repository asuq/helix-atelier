#!/usr/bin/env bash
# filter_representatives.sh
# Requires: gawk (for FPAT CSV parsing)
# Usage:
#   ./filter_representatives.sh input.csv            # writes to stdout
#   ./filter_representatives.sh input.csv output.csv # writes to file

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 input.csv [output.csv]" >&2
  exit 1
fi

if ! command -v gawk >/dev/null 2>&1; then
  echo "Error: gawk is required (needs FPAT). Install gawk 4+ and retry." >&2
  exit 3
fi

in="$1"
out="${2:-/dev/stdout}"

gawk -v OFS="," '
BEGIN {
  # FPAT: treat fields as either unquoted text with no commas OR a quoted string with CSV-style escaped quotes
  FPAT = "([^,]*)|(\"([^\"]|\"\")*\")"
}

function trim(s)     { sub(/^[[:space:]]+/, "", s); sub(/[[:space:]]+$/, "", s); return s }
function unquote(s,  t) {
  gsub(/\r$/, "", s)                 # strip trailing CR (Windows line endings)
  if (s ~ /^".*"$/) {
    t = substr(s, 2, length(s)-2)    # drop surrounding quotes
    gsub(/""/, "\"", t)              # unescape double quotes
    return t
  }
  return s
}
function csv_escape(s, t) {
  t = s
  gsub(/"/, "\"\"", t)
  if (t ~ /[",\r\n]/) t = "\"" t "\""
  return t
}

NR==1 {
  # Remove potential UTF-8 BOM
  sub(/^\xEF\xBB\xBF/, "", $0)

  # Map header names -> column indices (robust to column order)
  for (i=1; i<=NF; i++) {
    name = trim(unquote($i))
    col[name] = i
  }

  required[1] = "Cluster_ID"
  required[2] = "Accession"
  required[3] = "Organism_Name"
  required[4] = "Is_Representative"
  required[5] = "Path"

  for (i=1; i<=5; i++) {
    if (!(required[i] in col)) {
      printf "Error: required column \"%s\" not found in header.\n", required[i] > "/dev/stderr"
      exit 2
    }
  }

  print "Sample","Path"
  next
}

{
  ci = col["Cluster_ID"]
  ai = col["Accession"]
  oi = col["Organism_Name"]
  ri = col["Is_Representative"]
  pi = col["Path"]

  isrep = tolower(trim(unquote($ri)))
  if (isrep == "yes") {
    cluster   = trim(unquote($ci))
    accession = trim(unquote($ai))
    org       = trim(unquote($oi))

    sample = cluster "_" accession "_" org
    gsub(/[[:space:]]+/, "_", sample)  # replace any whitespace with underscore

    path = trim(unquote($pi))

    print csv_escape(sample), csv_escape(path)
  }
}
' "$in" > "$out"
