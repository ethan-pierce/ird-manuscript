echo "Downloading BedMachineGreenland-v5.nc"
touch .netrc
echo "machine urs.earthdata.nasa.gov login $EARTHDATA_USERNAME password $EARTHDATA_PASSWORD" >> .netrc
chmod 600 .netrc
wget -P data/ https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ICEBRIDGE/IDBMG4/5/1993/01/01/BedMachineGreenland-v5.nc

echo "Downloading GBaTSv2-GBaTSv2.zip"
wget -P data/ https://zenodo.org/records/5714527/files/joemacgregor/GBaTSv2-GBaTSv2.zip
mkdir data/GBaTSv2
unzip data/GBaTSv2-GBaTSv2.zip -d data/GBaTSv2/
mv data/GBaTSv2/joemacgregor-GBaTSv2-879adcb/GBaTSv2.nc data/GBaTSv2.nc
rm -rf data/GBaTSv2-GBaTSv2.zip
rm -rf data/GBaTSv2

echo "Downloading MEaSUREs_120m.nc"
wget -P data/ https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/static/ITS_LIVE_velocity_120m_RGI05A_0000_v02.nc
mv data/ITS_LIVE_velocity_120m_RGI05A_0000_v02.nc data/MEaSUREs_120m.nc
