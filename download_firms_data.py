#!/usr/bin/env python3
"""
NASA FIRMS Country Data Downloader
Downloads VIIRS SNPP and MODIS country data ZIP files from 2012-2025
"""

import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from urllib.parse import urljoin
from tqdm import tqdm
from typing import List, Tuple

# Configuration
OUTPUT_DIR = Path("./firms_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/zips/"

# Define download ranges
DOWNLOAD_RANGES = [
    # (file_prefix, start_year, end_year, description)
    ("viirs-snpp", 2012, 2025, "VIIRS SNPP"),
    ("modis", 2000, 2025, "MODIS"),
]

# Timeout configuration (in seconds)
TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Parallel download workers (like uv's concurrent installs)
MAX_WORKERS = 4


def test_url(url: str) -> bool:
    """
    Test if a download URL exists by sending a GET request with stream=True.
    HEAD is not used because the server may not support it for these files.
    """
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as response:
            return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def download_file(url: str, filepath: Path, position: int | None = None) -> bool:
    """
    Download a single file with retries, showing a dedicated progress bar.
    """
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(filepath, "wb") as f, tqdm(
                    desc=filepath.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    position=position,
                    leave=False,
                    mininterval=0.1,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

                return True

        except requests.exceptions.RequestException as e:
            tqdm.write(
                f"  Error downloading {filepath.name} "
                f"(attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            if filepath.exists():
                filepath.unlink()
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                tqdm.write(f"  Failed to download {filepath.name} after {MAX_RETRIES} attempts")
                return False

    return False


def main():
    print("=" * 70)
    print("NASA FIRMS Country Data Downloader")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Base URL: {BASE_URL}")
    print()

    # Prepare download list (check existence + URL availability)
    files_to_download: List[Tuple[str, Path, str]] = []
    existing_count = 0
    missing_urls: List[str] = []

    print("Preparing download list:")
    for file_prefix, start_year, end_year, description in DOWNLOAD_RANGES:
        for year in range(start_year, end_year + 1):
            url = urljoin(BASE_URL, f"{file_prefix}_{year}_all_countries.zip")
            filepath = OUTPUT_DIR / f"{file_prefix}_{year}_all_countries.zip"
            label = f"{description} {year}"

            if filepath.exists():
                existing_count += 1
                print(f"  - {label}: already exists, skipping")
                continue

            # Test the actual download URL (GET, not HEAD)
            if test_url(url):
                files_to_download.append((url, filepath, label))
                print(f"  - {label}: will download")
            else:
                missing_urls.append(url)
                print(f"  - {label}: URL not found")

    print()
    if existing_count:
        print(f"Already downloaded: {existing_count} files")
    if missing_urls:
        print(f"URLs not found: {len(missing_urls)}")
        for url in missing_urls:
            print(f"    {url}")
    print(f"Files to download: {len(files_to_download)}")

    if not files_to_download:
        print("\nNothing to download.")
        return

    # Confirm before downloading
    print("\n" + "=" * 70)
    response = input("Proceed with downloading? (y/N): ").strip().lower()
    if response not in ("y", "yes"):
        print("Cancelled.")
        return

    print("=" * 70)
    print(f"Starting downloads ({MAX_WORKERS} workers)...")
    print("=" * 70)

    start_time = time.time()
    success_count = 0
    failure_count = 0
    failed_labels: List[str] = []
    # Pool of terminal rows — one per concurrent worker
    slot_pool: Queue[int] = Queue()
    for i in range(MAX_WORKERS):
        slot_pool.put(i)

    def download_task(url: str, filepath: Path, label: str) -> Tuple[bool, str]:
        position = slot_pool.get()
        try:
            ok = download_file(url, filepath, position=position)
        finally:
            slot_pool.put(position)
        return ok, label

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_task, url, filepath, label): label
            for url, filepath, label in files_to_download
        }
        overall = tqdm(
            total=len(files_to_download),
            desc="Overall",
            position=MAX_WORKERS,
            leave=True,
            mininterval=0.1,
        )
        try:
            for future in as_completed(futures):
                ok, label = future.result()
                if ok:
                    success_count += 1
                else:
                    failure_count += 1
                    failed_labels.append(label)
                overall.set_postfix(ok=success_count, fail=failure_count)
                overall.update(1)
        finally:
            overall.close()

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    if failed_labels:
        print("Failed files:")
        for label in failed_labels:
            print(f"  - {label}")

    # List downloaded files
    zips = sorted(OUTPUT_DIR.glob("*.zip"))
    if zips:
        total_size = sum(f.stat().st_size for f in zips)
        print(f"\nFiles in {OUTPUT_DIR}: {len(zips)} ({total_size / (1024**3):.2f} GB)")
        for filepath in zips:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  - {filepath.name} ({size_mb:.1f} MB)")

    print("=" * 70)


if __name__ == "__main__":
    main()
