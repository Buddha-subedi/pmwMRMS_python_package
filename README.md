# pmwMRMS

Orbit-wise collocation of passive microwave L1C observations from NASA PPS with NOAA MRMS precipitation rate and type. Python 3.10+.

## Install

```bash
git clone https://github.com/Buddha-subedi/pmwMRMS_python_package.git
cd pmwMRMS_python_package
python -m pip install -e '.[maps]'
```

Restart Jupyter after installation, or add the repository's `src` directory to `sys.path`.

## Run

```python
from pmwMRMS import PMWtoMRMS, MRMStoPMW
options = dict(sensor='ATMS', satellite='NPP', start_date='2023-01-01',
               end_date='2023-01-01', output_dir='./data',
               buffer_minutes=2, delta_t=2, max_cases=1, plot=True, show=True)
PMWtoMRMS(**options, max_distance_km=25)
MRMStoPMW(**options, max_distance_km=25)
```

Omit `max_distance_km` to be prompted in km. The example radius is not a calibrated sensor footprint. Set `max_cases=None` to process all eligible orbits. PPS requests a registered email unless supplied via `email` or `PPS_USERNAME`. The PPS password defaults to that email; override via `password` or `PPS_PASSWORD`. NOAA public AWS requires no credentials. `version` optionally selects the PPS version. The bundled frequency metadata for ATMS, GMI and SSMIS must match the actual product; override with `frequencies={'S1':[...], ...}` for other products or versions. Channel counts are checked, but frequency ordering must be verified from product documentation.

## Algorithm

1. Discover PPS daily orbit metadata and download candidate granules. Inspect all swath groups for pixel centers inside the rectangular MRMS CONUS product domain (20–55 N, 130–60 W; not a political boundary). Skip nonintersecting orbits, with no MRMS downloads for them.
2. Find exact minimum and maximum in-domain observation times across the groups. List archive metadata for touched UTC days, including midnight crossings. Download only common rate/flag timestamps within the buffered overpass interval (default ±2 minutes). Reuse cached files. Each archive object is a complete CONUS grid.
3. Independently per native swath group, use spherical nearest-center distances to assign every MRMS grid cell to its nearest valid PMW center within the user radius. Choose the nearest paired MRMS timestamp to that center's observation time, and reject time differences exceeding `delta_t` minutes. Scan times are broadcast where beam times are not available. Read one MRMS pair at a time. Missing coverage, hail and unknown phases are excluded; dry valid cells contribute zero to both phases. Rain and snow use mm/hr liquid-equivalent rates. See https://www.nssl.noaa.gov/projects/mrms/operational/tables.php.
4. PMWtoMRMS preserves native MRMS rain/snow and copies the assigned center's TBs. MRMStoPMW sums and counts all valid MRMS cells assigned to each center, then divides by count. Equal cell weighting is used, not antenna weighting or physical footprint integration. Zero counts yield NaN.
5. Plot each completed orbit sequentially. For MRMStoPMW, map the aggregated values back using the same ownership grid, not a new interpolation. A rain/snow group with the largest valid cell count is chosen; low/high TB panels use the first available channel below 40/above 80 GHz. Groups may have different coverages and times.

## Output

`output_dir/PMW` and `output_dir/MRMS` are shared download caches. `matches/PMWtoMRMS` and `matches/MRMStoPMW` contain per-orbit NPZ and PNG. NPZ stores one-dimensional `mrms_latitude`, `mrms_longitude`, `mrms_times` (UTC Unix seconds), `groups`, `mode`, and per-group arrays prefixed by the swath name, e.g. `S1_data`, `S1_rain`, `S1_snow`, `S1_frequency_ghz`, `S1_owner` (flat PMW index, -1 unmatched). PMWtoMRMS data are on the MRMS grid and `S1_time_index` indexes `mrms_times`. MRMStoPMW data, `S1_lat`, `S1_lon`, `S1_time`, `S1_count` are on the original group grid; original TBs outside the matching region are retained, with rain/snow NaN and zero count. The compressed owner grid is retained for reproducible plotting. Use `plot_orbit(path)` to redraw. Older `automatic`, `process`, `plot_result` interfaces remain available for legacy single-snapshot cases.

Full CONUS arrays can require several GB of RAM, particularly for multiple groups and channels. This implementation streams time snapshots but retains spatial ownership arrays and output fields per group. It does not download a full day of MRMS observations unless the requested overpasses collectively require them. No physical microwave super-resolution is implied.

## Testing

`PYTHONPATH=src pytest -q` runs synthetic tests. Real PPS authentication, archive availability and GRIB decoding depend on the external services and local environment; live end-to-end validation is not included.
