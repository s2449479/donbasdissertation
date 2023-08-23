# donbasdissertation
The python scripts added to this repository act as the second step in the process of my dissertation project. 
The first step was carried out in google earth engine, using sentinel-2 and landsat 8 data to detect water bodies contaminated with acid mine drainage 
in the occupied Donbas, and monitor how they change over time. 

The Acid Mine Water Index (a normalised difference of bands 2 and 4), is tested for its use detecting AMD in a case where no ground-truth data is available.

The scripts below take Sentinel-2 imagery, and build a nested collection of 'contaminated' polygons feature collections, 
derived from  images in an image collection. These are then flattened into one collection, and using an id tag created, are counted across time.

Unfortunately, due to the prevelance of false positives and ambiguity with the index, counting contaminated bodies over time was not successful.  The method was successful at identifying which bodies were repeatedly and heavily contaminated.

STEP 1 https://code.earthengine.google.com/c8bda1a3d01b12dd5d0f07a651def118
STEP 2 https://code.earthengine.google.com/0050e3b6304464be17e96165713437aa

The second part of the process took those identified water bodies, and sought to chart the full change in AMWI across the timeseries. Here Landsat 8 as well as 
Sentinel-2 proved useful. Landsat provided a longer time series, to understand changes since occupation in 2014, however its coarse spatial resolution was a limit.

Sentinel-2's higher spatial resolution, and greater return time led to higher quality insights. However that atmospherically corrected imagery was only available
from 2019 did limit insights into long terms processes of contamination.

LANDSAT 8 https://code.earthengine.google.com/99dbad5ad72016cda32c453fc9ec2465
SENTINEL-2 https://code.earthengine.google.com/581d895467adc372028dfeccecc6388e

Outputs from this process were then used alongside other data sources in the python script within this repository.
