#!/usr/bin/env bash
while getopts u:p: flag
do
    case "${flag}" in
        u) uid=${OPTARG};;
        p) pwd=${OPTARG};;
    esac
done

# - SNODAS Download ----------------------------
#echo "Downloading SNODAS Data"

# create snodas directory and save to it
mkdir snodas
cd snodas

wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210115.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210121.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210122.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/01_Jan/SNODAS_20210129.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/02_Feb/SNODAS_20210217.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/02_Feb/SNODAS_20210218.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/02_Feb/SNODAS_20210224.tar
wget -nd --no-check-certificate --reject "index.html*" -np -e robots=off https://noaadata.apps.nsidc.org/NOAA/G02158/masked/2021/03_Mar/SNODAS_20210304.tar

#echo "Finishing Downloading SNODAS DATA"
cd ..
# ----------------------------------------------------



# download from NASA DAAC
# - UA 4km --------------------------------------------
mkdir ua_4km_swe
cd ua_4km_swe 
wget --http-user=$uid --http-password=$pwd --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r --reject "index.html*" -np -e robots=off -nd https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/4km_SWE_Depth_WY2021_v01.nc
cd ..
# ---------------------------------------------------

# WUS UCLA Snow Reanalysis ---------------------------------
mkdir wus_ucla_sr
cd wus_ucla_sr
wget --http-user=$uid --http-password=$pwd --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r --reject "index.html*" -np -e robots=off -nd https://n5eil01u.ecs.nsidc.org/SNOWEX/WUS_UCLA_SR.001/2020.10.01/WUS_UCLA_SR_v01_N47_0W110_0_agg_16_WY2020_21_SD_POST.nc
wget --http-user=$uid --http-password=$pwd --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r --reject "index.html*" -np -e robots=off -nd https://n5eil01u.ecs.nsidc.org/SNOWEX/WUS_UCLA_SR.001/2020.10.01/WUS_UCLA_SR_v01_N47_0W110_0_agg_16_WY2020_21_SCA_POST.nc
cd ..
# -------------------------------------------------
