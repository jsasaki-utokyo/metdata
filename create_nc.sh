#!/bin/bash

# Check if station argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <station>"
    echo "Example: $0 Tokyo"
    exit 1
fi

station=$1

python scripts/gwo_to_cf_netcdf.py \
  --station $station \
  --start "2010-01-01 00:00:00" \
  --end   "2023-01-01 00:00:00" \
  --output ${station}_2010-2022.nc

python scripts/check_netcdf_missing.py ${station}_2010-2022.nc --limit 20

python scripts/interpolate_netcdf.py ${station}_2010-2022.nc --max-gap 5


