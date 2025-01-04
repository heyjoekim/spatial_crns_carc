#!/usr/bin/env bash

# - SNODAS Download ----------------------------
echo "Downloading SNODAS Data"
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210115.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210121.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210122.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210129.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/02_Feb/SNODAS_20210217.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/02_Feb/SNODAS_20210218.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/02_Feb/SNODAS_20210224.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/03_Mar/SNODAS_20210304.tar

echo "Finishing Downloading SNODAS DATA"
# ----------------------------------------------------
#
#
#
