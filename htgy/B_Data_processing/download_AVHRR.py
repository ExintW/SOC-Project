# -*- coding: utf-8 -*-
"""
ERA5 download
"""

import cdsapi
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from globals import *
import os
import requests
from bs4 import BeautifulSoup

# Base URL for AVHRR‐Land LAI CDR daily files
BASE_URL = "https://www.ncei.noaa.gov/data/land-leaf-area-index-and-fapar/access"

# Local directory to store downloads
OUT_ROOT = DATA_DIR / "AVHRR"

# Years you want to download
START_YEAR, END_YEAR = 1982, 2025

session = requests.Session()

for year in range(START_YEAR, END_YEAR + 1):
    year_url = f"{BASE_URL}/{year}/"
    print(f"Listing files in {year_url}")

    # 1. 获取该年目录的 HTML
    resp = session.get(year_url)
    resp.raise_for_status()

    # 2. 解析出所有以 .nc 结尾的链接
    soup = BeautifulSoup(resp.text, "html.parser")
    nc_files = [
        a["href"] for a in soup.find_all("a", href=True)
        if a["href"].endswith(".nc")
    ]

    # 3. 下载每一个 .nc 文件
    local_year_dir = os.path.join(OUT_ROOT, str(year))
    os.makedirs(local_year_dir, exist_ok=True)

    for fname in nc_files:
        file_url = year_url + fname
        local_path = os.path.join(local_year_dir, fname)
        # 已存在则跳过
        if os.path.exists(local_path):
            continue

        print(f"Downloading {fname} ...")
        with session.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)

    print(f"Finished year {year}\n")

print("All downloads complete!")
