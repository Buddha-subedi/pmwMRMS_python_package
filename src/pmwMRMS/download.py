"""HTTPS satellite granule downloader with live progress."""

from pathlib import Path
from urllib.parse import urlparse
import os
import time
import requests
from requests.auth import HTTPBasicAuth


def download_urls(
    urls,
    output_dir,
    username=None,
    password=None,
    overwrite=False
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    auth = (
        HTTPBasicAuth(username, password)
        if username and password else None
    )

    paths = []

    with requests.Session() as session:

        for i, url in enumerate(urls, start=1):

            if not url.startswith("https://"):
                raise ValueError("Only HTTPS URLs accepted")

            name = Path(urlparse(url).path).name

            if not name:
                raise ValueError("URL must point to a file")

            target = out / name
            part = out / (name + ".part")

            if target.exists() and not overwrite:

                print(f"[CACHED] {name}", flush=True)

                paths.append(str(target))
                continue

            offset = (
                part.stat().st_size
                if part.exists() and not overwrite else 0
            )

            headers = (
                {"Range": f"bytes={offset}-"}
                if offset else {}
            )

            print(
                f"\n[DOWNLOAD {i}/{len(urls)}] {name}",
                flush=True
            )

            with session.get(
                url,
                auth=auth,
                headers=headers,
                stream=True,
                timeout=(30, 120)
            ) as response:

                response.raise_for_status()

                if offset and response.status_code != 206:
                    offset = 0

                remaining = int(
                    response.headers.get("Content-Length", 0)
                )

                total = remaining + offset if remaining else 0

                downloaded = offset
                start_time = time.monotonic()
                last_print = 0

                mode = "ab" if offset else "wb"

                with part.open(mode) as f:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if not chunk:
                            continue

                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.monotonic()
                        elapsed = max(now - start_time, 0.001)

                        speed = (
                            (downloaded - offset)
                            / elapsed / 1024**2
                        )

                        if now - last_print >= 2:

                            if total > 0:

                                percent = 100 * downloaded / total

                                print(
                                    f"\r[PROGRESS] "
                                    f"{percent:6.2f}% | "
                                    f"{downloaded / 1024**2:.1f}/"
                                    f"{total / 1024**2:.1f} MB | "
                                    f"{speed:.2f} MB/s",
                                    end="",
                                    flush=True
                                )

                            else:

                                print(
                                    f"\r[PROGRESS] "
                                    f"{downloaded / 1024**2:.1f} MB | "
                                    f"{speed:.2f} MB/s",
                                    end="",
                                    flush=True
                                )

                            last_print = now

                if total and downloaded != total:
                    raise IOError(
                        f"Incomplete download: {name}"
                    )

            part.replace(target)

            print(
                f"\n[COMPLETED] {name} | "
                f"{target.stat().st_size / 1024**2:.2f} MB",
                flush=True
            )

            paths.append(str(target))

    return paths


def download_pps(
    urls,
    output_dir,
    username=None,
    password=None
):
    user = username or os.getenv("PPS_USERNAME")
    pw = password or os.getenv("PPS_PASSWORD")

    if not user:
        user = input("PPS registered username/email: ")

    if not pw:
        import getpass
        pw = getpass.getpass("PPS password: ")

    return download_urls(
        urls, output_dir, user, pw
    )
