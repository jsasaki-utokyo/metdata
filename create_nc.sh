#!/bin/bash

# Default values
DEFAULT_START="2010-01-01 00:00:00"
DEFAULT_END="2024-01-01 00:00:00"
DEFAULT_MAX_GAP=24
DEFAULT_MISSING_LIMIT=20

# Initialize variables
station=""
start_date="$DEFAULT_START"
end_date="$DEFAULT_END"
output_file=""
overwrite_flag=""
max_gap=$DEFAULT_MAX_GAP
missing_limit=$DEFAULT_MISSING_LIMIT
skip_check=false
skip_interpolate=false

# Function to normalize datetime input
normalize_datetime() {
    local input="$1"
    # If input doesn't contain time (no colon), append default time
    if [[ ! "$input" =~ : ]]; then
        echo "${input} 00:00:00"
    else
        echo "$input"
    fi
}

# Help function
show_help() {
    cat << EOF
Usage: $0 --station <name> [OPTIONS]

Required:
  --station <name>           Station name (e.g., Tokyo, Chiba)

Optional:
  --output <file>            Output NetCDF filename (default: {station}_{start_year}-{end_year}.nc)
  --start <datetime>         Start date (default: $DEFAULT_START)
                             Format: YYYY-MM-DD or "YYYY-MM-DD HH:MM:SS"
  --end <datetime>           End date (default: $DEFAULT_END)
                             Format: YYYY-MM-DD or "YYYY-MM-DD HH:MM:SS"
  --max-gap <hours>          Maximum gap for interpolation (default: $DEFAULT_MAX_GAP)
  --missing-limit <n>        Number of missing values to report (default: $DEFAULT_MISSING_LIMIT)
  --overwrite                Overwrite existing output file
  --skip-check               Skip missing value check step
  --skip-interpolate         Skip interpolation step
  -h, --help                 Show this help message

Examples:
  # Basic usage with defaults (2010-2023)
  $0 --station Tokyo

  # Custom date range with auto-generated filename
  $0 --station Tokyo --start 2015-01-01 --end 2020-01-01

  # Specify custom output filename
  $0 --station Tokyo --output my_data.nc

  # Full customization
  $0 --station Chiba --start 2019-01-01 --end 2023-01-01 \\
     --output chiba_custom.nc --max-gap 12 --missing-limit 50 --overwrite

  # Skip optional steps
  $0 --station Tokyo --skip-check --skip-interpolate
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --station)
            station="$2"
            shift 2
            ;;
        --output)
            output_file="$2"
            shift 2
            ;;
        --start)
            start_date=$(normalize_datetime "$2")
            shift 2
            ;;
        --end)
            end_date=$(normalize_datetime "$2")
            shift 2
            ;;
        --max-gap)
            max_gap="$2"
            shift 2
            ;;
        --missing-limit)
            missing_limit="$2"
            shift 2
            ;;
        --overwrite)
            overwrite_flag="--overwrite"
            shift
            ;;
        --skip-check)
            skip_check=true
            shift
            ;;
        --skip-interpolate)
            skip_interpolate=true
            shift
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$station" ]; then
    echo "Error: --station is required"
    echo ""
    show_help
    exit 1
fi

# Auto-generate output filename if not specified
if [ -z "$output_file" ]; then
    # Extract years from dates
    start_year=$(echo "$start_date" | cut -d'-' -f1)
    end_year=$(echo "$end_date" | cut -d'-' -f1)
    # Handle case where end date is Jan 1 (use previous year)
    if echo "$end_date" | grep -q "^${end_year}-01-01"; then
        end_year=$((end_year - 1))
    fi
    output_file="${station}_${start_year}-${end_year}.nc"
fi

# Check if output file exists and --overwrite not specified
if [ -f "$output_file" ] && [ -z "$overwrite_flag" ]; then
    echo "Error: $output_file already exists. Use --overwrite to replace."
    exit 1
fi

# Display configuration
echo "========================================"
echo "  metdata NetCDF Creation Pipeline"
echo "========================================"
echo "Station:        $station"
echo "Start date:     $start_date"
echo "End date:       $end_date"
echo "Output file:    $output_file"
echo "Max gap:        ${max_gap} hours"
echo "Missing limit:  ${missing_limit} samples"
echo "Skip check:     $skip_check"
echo "Skip interp:    $skip_interpolate"
echo "========================================"
echo ""

# Step 1: Create NetCDF file
echo "[1/3] Creating CF-compliant NetCDF file..."
python scripts/gwo_to_cf_netcdf.py \
  --station "$station" \
  --start "$start_date" \
  --end "$end_date" \
  --output "$output_file" \
  $overwrite_flag

if [ $? -ne 0 ]; then
    echo "Error: Failed to create NetCDF file"
    exit 1
fi
echo "✓ NetCDF file created: $output_file"
echo ""

# Step 2: Check for missing values (optional)
if [ "$skip_check" = false ]; then
    echo "[2/3] Checking for missing values..."
    python scripts/check_netcdf_missing.py "$output_file" --limit "$missing_limit"
    if [ $? -ne 0 ]; then
        echo "Warning: Missing value check encountered issues"
    fi
    echo ""
else
    echo "[2/3] Skipping missing value check (--skip-check specified)"
    echo ""
fi

# Step 3: Interpolate missing values (optional)
if [ "$skip_interpolate" = false ]; then
    echo "[3/3] Interpolating missing values..."
    python scripts/interpolate_netcdf.py "$output_file" --max-gap "$max_gap"
    if [ $? -ne 0 ]; then
        echo "Error: Interpolation failed"
        exit 1
    fi
    echo "✓ Interpolation complete"
    echo ""
else
    echo "[3/3] Skipping interpolation (--skip-interpolate specified)"
    echo ""
fi

echo "========================================"
echo "✓ Pipeline complete!"
echo "Output: $output_file"
echo "========================================"
