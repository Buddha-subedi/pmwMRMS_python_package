"""
Orbit-wise PMW-MRMS spatiotemporal coincidence matching.

PMWtoMRMS:
    Map microwave brightness temperatures onto the MRMS grid.

MRMStoPMW:
    Assign MRMS pixels to their nearest PMW footprint center
    and average rainfall and snowfall.

MRMS is downloaded only during meaningful CONUS overpasses.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from functools import lru_cache

import numpy as np
import requests

from scipy.spatial import cKDTree

from cartopy.io import shapereader
from shapely import contains_xy
from shapely.geometry import box
from shapely.ops import unary_union

from .pps import (
    BASE,
    discover,
    interval,
    orbit,
    read_all,
    valid_geo
)

from .matching import _xyz, EARTH_KM
from .automatic import LAYOUTS
from .archive import AWS, list_product, phase_rates
from .download import download_urls
from .mrms import read_grid


# ============================================================
# 1. CONUS GEOGRAPHIC BOUNDARIES
# ============================================================

B = (24.0, 50.0, -125.0, -66.0)


@lru_cache(maxsize=1)
def conus_geometry():
    """
    Load the actual geographic boundary of the contiguous
    United States.

    Natural Earth geometry is downloaded once and cached.
    """

    path = shapereader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_0_countries"
    )

    geometries = [
        record.geometry
        for record in shapereader.Reader(path).records()
        if record.attributes.get("ADMIN")
        == "United States of America"
    ]

    if not geometries:
        raise RuntimeError(
            "Unable to identify United States geometry."
        )

    geometry = unary_union(geometries)

    # Exclude Alaska, Hawaii and overseas territories
    conus = geometry.intersection(
        box(-125.0, 24.0, -66.0, 50.0)
    )

    if conus.is_empty:
        raise RuntimeError("CONUS geometry is empty.")

    return conus


# ============================================================
# 2. ACTUAL CONUS FOOTPRINT MASK
# ============================================================

def domain_mask(a):
    """
    Identify valid satellite footprint centers located
    inside the actual CONUS geographic boundary.
    """

    x = (a["lon"] + 180.0) % 360.0 - 180.0
    y = a["lat"]

    valid = (
        valid_geo(a)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(a["time"])
        & np.any(np.isfinite(a["data"]), axis=-1)
        & (y >= B[0])
        & (y <= B[1])
        & (x >= B[2])
        & (x <= B[3])
    )

    mask = np.zeros(y.shape, dtype=bool)

    if np.any(valid):

        mask[valid] = contains_xy(
            conus_geometry(),
            x[valid],
            y[valid]
        )

    return mask


# ============================================================
# 3. TEMPORAL NEAREST NEIGHBOR
# ============================================================

def nearest_time(t, stamps):
    """
    Return the index of the nearest MRMS observation time.
    """

    j = np.searchsorted(stamps, t)

    a = np.clip(j - 1, 0, len(stamps) - 1)
    b = np.clip(j, 0, len(stamps) - 1)

    return np.where(
        np.abs(t - stamps[a]) <= np.abs(t - stamps[b]),
        a,
        b
    )


# ============================================================
# 4. SPATIOTEMPORAL NEAREST-CENTER ASSIGNMENT
# ============================================================

def assignments(
    src,
    lat,
    lon,
    radius,
    stamps,
    delta,
    crop=False
):
    """
    Assign each MRMS pixel to its nearest PMW footprint.

    Parameters
    ----------
    radius : float
        Maximum center-to-center spatial distance in km.

    delta : float
        Maximum temporal difference in minutes.

    Yields
    ------
    i : int
        Starting MRMS row index.

    owner : ndarray
        Flat PMW footprint index for each MRMS pixel.
        Unmatched cells contain -1.

    ti : ndarray
        Nearest MRMS temporal index.
    """

    ok = (
        valid_geo(src)
        & np.isfinite(src["time"])
        & np.any(
            np.isfinite(src["data"]),
            axis=-1
        )
    )

    # Restrict candidate observations to the available
    # MRMS observation window.

    ok &= (
        (src["time"] >= stamps[0] - delta * 60.0)
        & (src["time"] <= stamps[-1] + delta * 60.0)
    )

    # Remove source observations outside the actual
    # nearest-time tolerance before building the tree.

    source_times = src["time"].ravel()
    candidate_ids = np.flatnonzero(ok)

    if candidate_ids.size == 0:
        return

    candidate_times = source_times[candidate_ids]

    nearest = nearest_time(
        candidate_times,
        stamps
    )

    temporal_valid = (
        np.abs(
            candidate_times - stamps[nearest]
        ) <= delta * 60.0
    )

    ids = candidate_ids[temporal_valid]

    if ids.size == 0:
        return

    source_lat = src["lat"].ravel()[ids]
    source_lon = src["lon"].ravel()[ids]

    source_lon = (
        source_lon + 180.0
    ) % 360.0 - 180.0

    tree = cKDTree(
        _xyz(source_lat, source_lon)
    )

    chord = 2.0 * np.sin(
        radius / (2.0 * EARTH_KM)
    )

    print(
        f"[SPATIAL MATCHING] "
        f"{len(ids):,} eligible PMW footprints",
        flush=True
    )

    # Crop only the query domain; retain original output grid indices.
    row0, row1 = 0, len(lat)
    col0, col1 = 0, len(lon)
    if crop:
        # Bounding box of eligible footprint support, plus 1 km.
        # Use conservative spherical longitude padding at the highest
        # absolute latitude; this also handles descending grid axes.
        angle = (radius + 1.0) / EARTH_KM
        pad_lat = np.rad2deg(angle)
        south = max(-90.0, source_lat.min() - pad_lat)
        north = min(90.0, source_lat.max() + pad_lat)
        rows = np.flatnonzero((lat >= south) & (lat <= north))
        if not rows.size:
            return
        row0, row1 = int(rows[0]), int(rows[-1]) + 1
        max_abs_lat = max(abs(south), abs(north))
        # Fall back to all columns near poles or the antimeridian.
        if max_abs_lat < 90.0 and angle < np.pi / 2:
            pad_lon = np.rad2deg(np.arcsin(min(1.0,
                np.sin(angle) / np.cos(np.deg2rad(max_abs_lat)))))
            west = source_lon.min() - pad_lon
            east = source_lon.max() + pad_lon
            if west >= -180 and east <= 180:
                grid_lon = (lon + 180.0) % 360.0 - 180.0
                cols = np.flatnonzero((grid_lon >= west) & (grid_lon <= east))
                if not cols.size:
                    return
                col0, col1 = int(cols[0]), int(cols[-1]) + 1
        cells = (row1 - row0) * (col1 - col0)
        total = len(lat) * len(lon)
        print(f"[MRMS QUERY CROP] {row1-row0:,} x {col1-col0:,} = "
              f"{cells:,}/{total:,} cells ({100*cells/total:.2f}%)", flush=True)

    for i in range(row0, row1, 64):
        yy, xx = np.meshgrid(
            lat[i:min(i + 64, row1)],
            lon[col0:col1],
            indexing="ij"
        )

        d, j = tree.query(
            _xyz(
                yy.ravel(),
                xx.ravel()
            ),
            distance_upper_bound=chord,
            workers=-1
        )

        good = np.isfinite(d)

        owner = np.full(
            d.shape,
            -1,
            dtype=np.int32
        )

        owner[good] = ids[j[good]]

        ti = np.full(
            d.shape,
            -1,
            dtype=np.int32
        )

        k = np.flatnonzero(good)

        if k.size:

            q = nearest_time(
                source_times[owner[k]],
                stamps
            )

            take = (
                np.abs(
                    source_times[owner[k]]
                    - stamps[q]
                ) <= delta * 60.0
            )

            ti[k[take]] = q[take]
            owner[k[~take]] = -1

        # Preserve the existing full-width block interface.
        if col0 != 0 or col1 != len(lon):
            full_owner = np.full((yy.shape[0], len(lon)), -1, dtype=np.int32)
            full_ti = np.full_like(full_owner, -1)
            full_owner[:, col0:col1] = owner.reshape(yy.shape)
            full_ti[:, col0:col1] = ti.reshape(yy.shape)
            yield i, full_owner, full_ti
        else:
            yield i, owner.reshape(yy.shape), ti.reshape(yy.shape)



# ============================================================
# 5. MRMS PRECIPITATION AGGREGATION
# ============================================================

def aggregate(owner, rain, snow, n):
    """
    Aggregate valid MRMS rainfall and snowfall assigned
    to each native PMW footprint.

    Valid zero precipitation is included.
    Missing precipitation is excluded.
    """

    ok = (
        (owner >= 0)
        & np.isfinite(rain)
        & np.isfinite(snow)
        & (rain >= 0)
        & (snow >= 0)
    )

    ids = owner[ok]

    count = np.bincount(
        ids,
        minlength=n
    )

    rain_sum = np.bincount(
        ids,
        weights=rain[ok],
        minlength=n
    )

    snow_sum = np.bincount(
        ids,
        weights=snow[ok],
        minlength=n
    )

    return count, rain_sum, snow_sum


# ============================================================
# 6. SHARED AUTOMATIC ORBIT PROCESSING
# ============================================================

def fill_empty_footprints(src, state, lat, lon, rain, snow, stamps,
                          index, delta, radius, domain):
    """Fill empty CONUS footprints from the same-time MRMS grid.

    Prefer valid cells within 1 km; otherwise average up to 100 nearest
    valid cells within the configured spatial tolerance. No ownership changes.
    """
    times = src["time"].ravel()
    ids = np.flatnonzero(domain.ravel() & (state["count"] == 0))
    if not ids.size:
        return
    q = nearest_time(times[ids], stamps)
    ids = ids[(q == index) & (np.abs(times[ids] - stamps[index]) <= delta * 60)]
    angle = radius / EARTH_KM
    pad = np.rad2deg(angle)
    glon = (lon + 180) % 360 - 180
    for j in ids:
        y = src["lat"].ravel()[j]
        x = (src["lon"].ravel()[j] + 180) % 360 - 180
        rows = np.flatnonzero(np.abs(lat - y) <= pad)
        if not rows.size:
            continue
        high = abs(y) + pad
        lonpad = 180 if high >= 90 or angle >= np.pi/2 else np.rad2deg(
            np.arcsin(min(1., np.sin(angle)/np.cos(np.deg2rad(high)))))
        cols = np.flatnonzero(np.abs((glon-x+180)%360-180) <= lonpad)
        if not cols.size:
            continue
        rr = rain[np.ix_(rows, cols)]
        ss = snow[np.ix_(rows, cols)]
        a, b = np.nonzero(np.isfinite(rr) & np.isfinite(ss) & (rr >= 0) & (ss >= 0))
        if not a.size:
            continue
        xyz = _xyz(lat[rows[a]], lon[cols[b]])
        d = np.linalg.norm(xyz - _xyz(np.array([y]), np.array([x]))[0], axis=1)
        valid = np.flatnonzero(d <= 2*np.sin(angle/2))
        if not valid.size:
            continue
        near = valid[d[valid] <= 2*np.sin(1./(2*EARTH_KM))]
        if not near.size:
            k = min(100, valid.size)
            near = valid[np.argpartition(d[valid], k-1)[:k]]
        state["fill_count"][j] = near.size
        state["fill_rain"][j] = np.mean(rr[a[near], b[near]], dtype=np.float64)
        state["fill_snow"][j] = np.mean(ss[a[near], b[near]], dtype=np.float64)


def _run(
    mode,
    sensor,
    satellite,
    start_date,
    end_date,
    output_dir="./pmwMRMS_data",
    version=None,
    email=None,
    password=None,
    buffer_minutes=2,
    delta_t=2,
    max_distance_km=None,
    max_cases=None,
    frequencies=None,
    plot=True,
    show=True,
    base=BASE,
    min_conus_footprints=500,
    min_matched_pixels=100000
):

    # --------------------------------------------------------
    # 6.1 Validate user configuration
    # --------------------------------------------------------

    if max_distance_km is None:

        max_distance_km = float(
            input(
                "Maximum PMW center to MRMS pixel "
                "distance (km): "
            )
        )

    if (
        not 0 < max_distance_km < 20000
        or not np.isfinite(delta_t)
        or delta_t < 0
        or not np.isfinite(buffer_minutes)
        or buffer_minutes < 0
    ):
        raise ValueError(
            "Invalid distance or time tolerance."
        )

    if max_cases is not None and max_cases < 1:
        raise ValueError(
            "max_cases must be positive."
        )

    if min_conus_footprints < 1:
        raise ValueError(
            "min_conus_footprints must be positive."
        )

    if min_matched_pixels < 1:
        raise ValueError(
            "min_matched_pixels must be positive."
        )

    start = datetime.fromisoformat(
        start_date
    ).replace(tzinfo=timezone.utc)

    end = datetime.fromisoformat(
        end_date
    ).replace(
        tzinfo=timezone.utc
    ) + timedelta(days=1)

    if end <= start:
        raise ValueError("Invalid date range.")

    # --------------------------------------------------------
    # 6.2 NASA PPS authentication
    # --------------------------------------------------------

    email = (
        email
        or os.getenv("PPS_USERNAME")
        or input("PPS registered email: ").strip()
    )

    password = (
        password
        or os.getenv("PPS_PASSWORD")
        or email
    )

    root = Path(output_dir)

    root.mkdir(
        parents=True,
        exist_ok=True
    )

    layout = (
        frequencies
        or LAYOUTS.get(sensor.upper())
    )

    if layout is None:
        raise ValueError(
            "Provide frequencies mapping for this sensor."
        )

    cache = {}
    results = []

    print("\n" + "=" * 70)
    print("PMW-MRMS ORBIT MATCHING")
    print("=" * 70)
    print("Mapping direction:", mode)
    print("Satellite:", satellite)
    print("Sensor:", sensor)
    print("Start date:", start_date)
    print("End date:", end_date)
    print("Maximum distance:", max_distance_km, "km")
    print("Temporal tolerance:", delta_t, "minutes")
    print("MRMS time buffer:", buffer_minutes, "minutes")
    print("Minimum CONUS footprints:", min_conus_footprints)
    print("Minimum matched MRMS pixels:", min_matched_pixels)
    print("=" * 70)

    # Load and cache CONUS geometry before processing.

    conus_geometry()

    # --------------------------------------------------------
    # 6.3 Discover PMW orbit URLs
    # --------------------------------------------------------

    def pred(n):

        tok = n.upper().split(".")

        return (
            tok[0] in ("1C", "1C-R")
            and sensor.upper() in tok
            and satellite.upper() in tok
            and (
                not version
                or version.upper() in tok
            )
        )

    with requests.Session() as session:

        session.auth = (email, password)

        urls = discover(
            session,
            start - timedelta(days=1),
            end,
            "1C",
            pred,
            base
        )

    urls = sorted(
        set(urls),
        key=lambda u: interval(u)[0]
    )

    print(
        f"\n[PPS] {len(urls)} candidate orbit files",
        flush=True
    )

    # ========================================================
    # 7. PROCESS PMW ORBITS ONE BY ONE
    # ========================================================

    for url in urls:

        a, b = interval(url)

        if b < start or a >= end:
            continue

        orbit_number = orbit(url)

        print("\n" + "=" * 70)
        print(f"[CHECKING ORBIT] {orbit_number}")
        print("[START]", a)
        print("[END]", b)
        print("=" * 70)

        # ----------------------------------------------------
        # 7.1 Download only the current PMW orbit
        # ----------------------------------------------------

        file = download_urls(
            [url],
            root / "PMW",
            email,
            password
        )[0]

        groups = read_all(file)

        if not groups:

            print(
                "[SKIP] No valid PMW swath groups.",
                flush=True
            )

            continue

        # ----------------------------------------------------
        # 7.2 Detect actual CONUS intersection
        # ----------------------------------------------------

        masks = {
            g: domain_mask(v)
            for g, v in groups.items()
        }

        footprint_counts = {
            g: int(np.count_nonzero(mask))
            for g, mask in masks.items()
        }

        print(
            "[CONUS FOOTPRINT COUNTS]",
            footprint_counts,
            flush=True
        )

        if max(
            footprint_counts.values(),
            default=0
        ) < min_conus_footprints:

            print(
                "[SKIP] Insufficient actual CONUS overlap.",
                flush=True
            )

            print(
                "[SKIP] No MRMS data downloaded.",
                flush=True
            )

            continue

        # ----------------------------------------------------
        # 7.3 Determine CONUS entry and exit time
        # ----------------------------------------------------

        chunks = [
            groups[g]["time"][mask]
            for g, mask in masks.items()
            if np.any(mask)
        ]

        if not chunks:

            print(
                "[SKIP] No valid CONUS observation times.",
                flush=True
            )

            continue

        lo = min(x.min() for x in chunks)
        hi = max(x.max() for x in chunks)

        if (
            hi < start.timestamp()
            or lo >= end.timestamp()
        ):
            continue

        entry = datetime.fromtimestamp(
            lo,
            timezone.utc
        )

        exit_time = datetime.fromtimestamp(
            hi,
            timezone.utc
        )

        begin = entry - timedelta(
            minutes=buffer_minutes
        )

        finish = exit_time + timedelta(
            minutes=buffer_minutes
        )

        duration = (
            hi - lo
        ) / 60.0

        print(
            "[CONUS ENTRY]", entry,
            flush=True
        )

        print(
            "[CONUS EXIT]", exit_time,
            flush=True
        )

        print(
            f"[CONUS DURATION] {duration:.2f} minutes",
            flush=True
        )

        print(
            f"[MRMS WINDOW] {begin} to {finish}",
            flush=True
        )

        # ----------------------------------------------------
        # 7.4 Find relevant MRMS timestamps
        # ----------------------------------------------------

        rate = {}
        flag = {}

        day = begin

        while day.date() <= finish.date():

            if day.date() not in cache:

                cache[day.date()] = (
                    list_product("PrecipRate", day),
                    list_product("PrecipFlag", day)
                )

            r, f = cache[day.date()]

            rate.update(r)
            flag.update(f)

            day += timedelta(days=1)

        keys = sorted(
            t
            for t in rate.keys() & flag.keys()
            if begin <= t <= finish
        )

        if not keys:

            print(
                "[SKIP] No paired MRMS timestamps.",
                flush=True
            )

            continue

        stamps = np.array(
            [t.timestamp() for t in keys],
            dtype=np.float64
        )

        print(
            f"[MRMS] {len(keys)} snapshot pairs required",
            flush=True
        )

        # ----------------------------------------------------
        # 7.5 Initialize orbit state
        # ----------------------------------------------------

        states = {}
        reference_index = int(np.argmin(np.abs(stamps - (lo + hi) / 2)))
        reference_rain = reference_snow = None

        lat = None
        lon = None

        # ----------------------------------------------------
        # 7.6 Download and process MRMS sequentially
        # ----------------------------------------------------

        for index, t in enumerate(keys):

            print(
                f"\n[MRMS SNAPSHOT "
                f"{index + 1}/{len(keys)}] {t}",
                flush=True
            )

            paths = download_urls(
                [
                    AWS + "/" + quote(rate[t]),
                    AWS + "/" + quote(flag[t])
                ],
                root / "MRMS"
            )

            y, x, r, rt, _ = read_grid(paths[0])

            fy, fx, f, ft, _ = read_grid(paths[1])

            if (
                rt != ft
                or abs(rt - stamps[index]) > 1
                or not np.array_equal(y, fy)
                or not np.array_equal(x, fx)
            ):

                raise ValueError(
                    "MRMS rainfall and precipitation "
                    "flag grid/time mismatch."
                )

            rain, snow = phase_rates(r, f)
            if index == reference_index:
                reference_rain, reference_snow = rain.copy(), snow.copy()

            del r, f

            # ------------------------------------------------
            # Initialize assignments once per orbit
            # ------------------------------------------------

            if lat is None:

                lat = y
                lon = x

                print(
                    "[MRMS GRID]",
                    (len(lat), len(lon)),
                    flush=True
                )

                for g, src in groups.items():

                    if (
                        g not in layout
                        or len(layout[g])
                        != src["data"].shape[-1]
                    ):

                        raise ValueError(
                            f"Frequency metadata mismatch: {g}. "
                            "Supply correct frequencies."
                        )

                    owner = np.full(
                        (len(y), len(x)),
                        -1,
                        dtype=np.int32
                    )

                    ti = owner.copy()

                    for i, o, j in assignments(
                        src,
                        y,
                        x,
                        max_distance_km,
                        stamps,
                        delta_t,
                        crop=(mode == "MRMStoPMW")
                    ):

                        owner[i:i + len(o)] = o
                        ti[i:i + len(o)] = j

                    spatial_count = int(
                        np.count_nonzero(owner >= 0)
                    )

                    print(
                        f"[{g}] Spatiotemporal pixels: "
                        f"{spatial_count:,}",
                        flush=True
                    )

                    n = src["lat"].size

                    states[g] = dict(
                        owner=owner,
                        ti=ti,
                        count=np.zeros(
                            n,
                            dtype=np.int64
                        ),
                        rs=np.zeros(n),
                        ss=np.zeros(n)
                    )

                    if mode == "MRMStoPMW":
                        states[g].update(fill_count=np.zeros(n, dtype=np.int32),
                                         fill_rain=np.full(n, np.nan),
                                         fill_snow=np.full(n, np.nan))

                    if mode == "PMWtoMRMS":

                        states[g].update(
                            rain=np.full(
                                owner.shape,
                                np.nan,
                                dtype=np.float32
                            ),
                            snow=np.full(
                                owner.shape,
                                np.nan,
                                dtype=np.float32
                            )
                        )

                spatial_counts = {
                    g: int(
                        np.count_nonzero(s["owner"] >= 0)
                    )
                    for g, s in states.items()
                }

                if max(
                    spatial_counts.values(),
                    default=0
                ) < min_matched_pixels:

                    print(
                        "[SKIP] Insufficient spatiotemporal "
                        "coverage. Stopping MRMS downloads.",
                        flush=True
                    )

                    states.clear()

                    del rain, snow

                    break

            elif (
                not np.array_equal(lat, y)
                or not np.array_equal(lon, x)
            ):

                raise ValueError(
                    "MRMS grid changed within orbit."
                )

            # ------------------------------------------------
            # Assign precipitation at the nearest time
            # ------------------------------------------------

            for g, state in states.items():

                select = state["ti"] == index

                ids = state["owner"][select]

                rr = rain[select]
                ss = snow[select]

                valid = (
                    np.isfinite(rr)
                    & np.isfinite(ss)
                    & (rr >= 0)
                    & (ss >= 0)
                    & (ids >= 0)
                )

                n = len(state["count"])

                c, r_sum, s_sum = aggregate(
                    ids,
                    rr,
                    ss,
                    n
                )

                state["count"] += c
                state["rs"] += r_sum
                state["ss"] += s_sum

                if mode == "PMWtoMRMS":

                    state["rain"][select] = np.where(
                        valid,
                        rr,
                        np.nan
                    )

                    state["snow"][select] = np.where(
                        valid,
                        ss,
                        np.nan
                    )

                if mode == "MRMStoPMW":
                    fill_empty_footprints(groups[g], state, lat, lon, rain, snow,
                                          stamps, index, delta_t, max_distance_km, masks[g])

                # Reject pixels without valid precipitation
                # references from the paired dataset.

                state["owner"][select] = np.where(
                    valid,
                    ids,
                    -1
                )

                print(
                    f"[{g}] Valid matched pixels: "
                    f"{np.count_nonzero(valid):,}",
                    flush=True
                )

            del rain, snow

        # ====================================================
        # 8. CHECK FINAL MATCHED COVERAGE
        # ====================================================

        if not states:
            continue

        matched_counts = {
            g: int(np.sum(s["count"]))
            for g, s in states.items()
        }

        total_pixels = len(lat) * len(lon)

        print("\n" + "-" * 70)
        print("[FINAL MATCHING STATISTICS]")
        print("-" * 70)

        for g, count in matched_counts.items():

            coverage = (
                100.0 * count / total_pixels
            )

            native_footprints = int(
                np.count_nonzero(
                    states[g]["count"] > 0
                )
            )

            print(
                f"[{g}] "
                f"MRMS pixels: {count:,} | "
                f"Coverage: {coverage:.4f}% | "
                f"PMW footprints: {native_footprints:,}",
                flush=True
            )

        if max(
            matched_counts.values(),
            default=0
        ) < min_matched_pixels:

            print(
                "[SKIP] Insufficient valid matched coverage.",
                flush=True
            )

            print(
                "[SKIP] No NPZ or blank figure saved.",
                flush=True
            )

            continue

        # ====================================================
        # 9. PREPARE OUTPUT FILE
        # ====================================================

        target = (
            root
            / "matches"
            / mode
            / (
                f"{satellite}_{sensor}_"
                f"{keys[0]:%Y%m%d_%H%M%S}_"
                f"{orbit_number:06d}.npz"
            )
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        out = dict(
            mode=mode,
            reference_rain=reference_rain,
            reference_snow=reference_snow,
            reference_time=stamps[reference_index],
            groups=np.array(list(states)),
            mrms_latitude=lat,
            mrms_longitude=lon,
            mrms_times=stamps,
            max_distance_km=max_distance_km,
            pmw_file=str(file),
            conus_entry_time=lo,
            conus_exit_time=hi,
            temporal_tolerance_minutes=delta_t,
            buffer_minutes=buffer_minutes,
            orbit_number=orbit_number
        )

        # ====================================================
        # 10. SAVE ACCORDING TO MAPPING DIRECTION
        # ====================================================

        for g, s in states.items():

            src = groups[g]

            owner = s["owner"]

            good = owner >= 0

            count = s["count"]

            shape = src["lat"].shape

            out[g + "_frequency_ghz"] = np.array(
                layout[g]
            )

            out[g + "_owner"] = owner

            # ------------------------------------------------
            # CASE 1: MRMS onto native PMW grid
            # ------------------------------------------------

            if mode == "MRMStoPMW":

                for k in (
                    "lat",
                    "lon",
                    "time",
                    "data"
                ):

                    out[g + "_" + k] = src[k]

                out[g + "_count"] = count.reshape(
                    shape
                )

                filled = (count == 0) & (s["fill_count"] > 0)
                out[g + "_fallback_count"] = np.where(filled, s["fill_count"], 0).reshape(shape)
                out[g + "_fallback_mask"] = filled.reshape(shape)
                print(f"[{g}] Fallback-filled footprints: {filled.sum():,}", flush=True)

                for k, v in [
                    ("rain", s["rs"]),
                    ("snow", s["ss"])
                ]:

                    averaged = np.divide(
                        v,
                        count,
                        out=np.full(
                            v.shape,
                            np.nan
                        ),
                        where=count > 0
                    )

                    averaged[filled] = s["fill_" + k][filled]

                    out[g + "_" + k] = (
                        averaged
                        .reshape(shape)
                        .astype(np.float32)
                    )

            # ------------------------------------------------
            # CASE 2: PMW onto MRMS grid
            # ------------------------------------------------

            else:

                out[g + "_rain"] = s["rain"]
                out[g + "_snow"] = s["snow"]

                n_channels = src["data"].shape[-1]

                tb = np.full(
                    (*owner.shape, n_channels),
                    np.nan,
                    dtype=np.float32
                )

                tb[good] = (
                    src["data"]
                    .reshape(-1, n_channels)
                    [owner[good]]
                )

                out[g + "_data"] = tb

                ti = s["ti"].copy()

                ti[~good] = -1

                out[g + "_time_index"] = ti

        # ====================================================
        # 11. SAVE MATCHED ORBIT
        # ====================================================

        print(
            "\n[SAVING]", target,
            flush=True
        )

        np.savez_compressed(
            target,
            **out
        )

        # ====================================================
        # 12. VISUALIZE CURRENT ORBIT
        # ====================================================

        if plot:

            from .orbit_plot import plot_orbit

            plot_orbit(
                target,
                show=show,
                states=states
            )

        results.append(str(target))

        print(
            "[SAVED]", target,
            flush=True
        )

        print(
            f"[COMPLETED ORBITS] {len(results)}",
            flush=True
        )

        # Release large orbit-specific arrays.

        del out, states, groups, masks

        if (
            max_cases is not None
            and len(results) >= max_cases
        ):

            print(
                "[STOP] Requested number of "
                "successful orbits reached.",
                flush=True
            )

            break

    # ========================================================
    # 13. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETED")
    print("=" * 70)
    print("Mapping direction:", mode)
    print("Successful orbits:", len(results))

    if not results:

        print(
            "No orbit satisfied the geographic and "
            "matching quality-control thresholds."
        )

    return results


# ============================================================
# 14. PUBLIC FUNCTION: PMW TO MRMS
# ============================================================

def PMWtoMRMS(*args, **kwargs):
    """
    Map PMW brightness temperatures onto the native MRMS grid.

    Preserve the original fine-resolution matched MRMS
    rainfall and snowfall variability.
    """

    return _run(
        "PMWtoMRMS",
        *args,
        **kwargs
    )


# ============================================================
# 15. PUBLIC FUNCTION: MRMS TO PMW
# ============================================================

def MRMStoPMW(*args, **kwargs):
    """
    Assign each MRMS pixel to its nearest PMW footprint
    and average all valid rainfall and snowfall values.

    Save precipitation on the native PMW swath grid.
    """

    return _run(
        "MRMStoPMW",
        *args,
        **kwargs
    )
