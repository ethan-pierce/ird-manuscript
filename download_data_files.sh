echo "Downloading BedMachineGreenland-v5.nc"
touch .netrc
echo "machine urs.earthdata.nasa.gov login $EARTHDATA_USERNAME password $EARTHDATA_PASSWORD" >> .netrc
chmod 600 .netrc
wget -P data/ https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ICEBRIDGE/IDBMG4/5/1993/01/01/BedMachineGreenland-v5.nc

echo "Downloading basal melt rates from Karlsson et al. (2021)"
wget -P data/ https://dataverse.geus.dk/api/access/datafile/19458?gbrecs=true
mv data/19458?gbrecs=true data/basalmelt.nc

echo "Downloading MEaSUREs_120m.nc"
wget -P data/ https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/static/ITS_LIVE_velocity_120m_RGI05A_0000_v02.nc
mv data/ITS_LIVE_velocity_120m_RGI05A_0000_v02.nc data/MEaSUREs_120m.nc
