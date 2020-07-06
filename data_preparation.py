import os
import gdal
import rasterio
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon
import earthpy.spatial as es
import shutil
import pandas as pd

# Add part for deleting folder content of test and train
# Set data direction
datadir = os.getcwd()

# Delete all files within train, test and validation folders
if os.path.exists('data\\derived_data') and os.path.isdir('data\\derived_data'):
    shutil.rmtree('data\\derived_data')
if os.path.exists('data\\derived_data\\tiles') and os.path.isdir('data\\derived_data\\tiles'):
    shutil.rmtree('data\\derived_data\\tiles')
if os.path.exists('data\\derived_data\\confirmed_sites') and os.path.isdir('data\\derived_data\\confirmed_sites'):
    shutil.rmtree('data\\derived_data\\confirmed_sites')
if os.path.exists('data\\derived_data\\negative_examples') and os.path.isdir('data\\derived_data\\negative_examples'):
    shutil.rmtree('data\\derived_data\\negative_examples')

# Create folder structure:
# Get data
tell_sites = gpd.read_file("data\\raw_data\\tell_sites.geojson").to_crs('epsg:32637')


# Clip raster by centered tell point
def clip_by_point(POI, ras):
    for points, row in POI.iterrows():
        point = gpd.GeoDataFrame(gpd.GeoSeries(row.geometry)).rename(columns={0: 'geometry'})
        point['center'] = point.geometry
        point = point.set_geometry('center')
        point['center'] = shapely.geometry.box(*point['center'].buffer(1000).total_bounds)

        pol = [Point(np.asarray(point.total_bounds[0]), np.asarray(point.total_bounds)[1]),
               Point(np.asarray(point.total_bounds)[2], np.asarray(point.total_bounds)[1]),
               Point(np.asarray(point.total_bounds)[2], np.asarray(point.total_bounds)[3]),
               Point(np.asarray(point.total_bounds)[0], np.asarray(point.total_bounds)[3])]

        bb = gpd.GeoSeries(Polygon(sum(map(list, (p.coords for p in pol)), [])), crs='EPSG:32637')
        bb = gpd.GeoDataFrame(bb, geometry=bb)

        with rasterio.open(ras) as rast:
            rast_crop, rast_crop_meta = es.crop_image(rast, bb)

        rast_crop_affine = rast_crop_meta["transform"]

        rast_crop_meta.update({'transform': rast_crop_affine,
                               'height': rast_crop.shape[1],
                               'width': rast_crop.shape[2]})

        with rasterio.open('data\\derived_data\\confirmed_sites\\''confirmed_' + str(points) + '.tif',
                           'w', **rast_crop_meta) as ff:
            ff.write(rast_crop)


clip_by_point(POI=tell_sites, ras="data\\raw_data\\study_area_hillshade_32637_GT_saturated.tif")

# Tiling the data
# Preparation for tiling
data_folder = datadir + '\\data'
input_filename = '\\raw_data\\study_area_hillshade_32637_GT_saturated.tif'

# Define tiling size
tile_size_x = 73
tile_size_y = 73

ds = gdal.Open(data_folder + input_filename)
band = ds.GetRasterBand(1)
n = 1

# For loop for tiling using GDAL
for i in range(0, band.XSize, tile_size_x):
    for j in range(0, band.YSize, tile_size_y):
        com_string = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(
            tile_size_x) + ", " + str(tile_size_y) + " " + str(data_folder) + str(input_filename) + " " + str(
            data_folder + '\\derived_data\\tiles\\') + str('tile_') + str(n) + ".tif"
        n = n + 1
        os.system(com_string)

# Filter images
nodatas = []
for filename in os.listdir('data\\derived_data\\tiles'):
    with rasterio.open('data\\derived_data\\tiles' + '\\' + filename, crs={'EPSG:32637'}) as src:
        raster_array = src.read(1).ravel()

    num_zeros = (raster_array == 0).sum()

    if (num_zeros > 800).any():
        print('Found NA:', filename)
        nodatas.append('data\\derived_data\\tiles\\' + filename)

    else:
        continue
else:
    print(len(nodatas))

# Remove nodatas list from all tiles folder
for f in nodatas:
    os.remove(f)

# Loop through directory to detect the 'new' images containing already confirmed sites from dataset
findings = []
sites = gpd.GeoSeries(tell_sites['geometry'], crs='EPSG:32637')

for filename in os.listdir(r'data\\derived_data\\tiles'):
    raster = rasterio.open(r'data\\derived_data\\tiles' + '\\' + filename, crs={'init': 'epsg:32637'})
    corners = [Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[1]),
               Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[1]),
               Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[3]),
               Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[3])]

    bb = gpd.GeoSeries(Polygon(sum(map(list, (p.coords for p in corners)), [])), crs='EPSG:32637')
    bools = sites.within(bb.loc[0])

    if True in bools.values:
        print('Found raster containing tell:', filename)
        findings.append('data\\derived_data\\tiles\\' + filename)

    else:
        continue

# Buffer each point using a 900 meter circle radius
poly = tell_sites.copy()
poly['geometry'] = poly.geometry.buffer(900)
poly_g = gpd.GeoSeries(poly['geometry'], crs='EPSG:32637')

findings_negative = []

for filename in os.listdir(r'data/derived_data/tiles'):
    raster = rasterio.open(r'data/derived_data/tiles' + '/' + filename, crs={'init': 'epsg:32637'})
    points = [Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[1]),
              Point(np.asarray(raster.bounds)[2], np.asarray(raster.bounds)[3]),
              Point(np.asarray(raster.bounds)[0], np.asarray(raster.bounds)[3])]

    bb = gpd.GeoSeries(Polygon(sum(map(list, (p.coords for p in points)), [])), crs='EPSG:32637')
    bools = poly_g.overlaps(bb.loc[0])

    if True in bools.values:
        findings_negative.append('data\\derived_data\\tiles\\' + filename)

    else:
        continue

for i in findings:
    try:
        findings_negative.remove(i)
    except ValueError:
        pass

for f in findings_negative:
    shutil.copy(f, 'data/derived_data/negative_examples')

# Remove sites by hand
eliminated_sites = ['confirmed_7.tif', 'confirmed_9.tif', 'confirmed_5.tif',
                    'confirmed_6.tif', 'confirmed_15.tif', 'confirmed_18.tif',
                    'confirmed_19.tif', 'confirmed_21.tif', 'confirmed_23.tif',
                    'confirmed_24.tif', 'confirmed_25.tif', 'confirmed_26.tif',
                    'confirmed_28.tif', 'confirmed_29.tif', 'confirmed_32.tif',
                    'confirmed_36.tif', 'confirmed_38.tif', 'confirmed_39.tif', 'confirmed_46.tif']

for f in eliminated_sites:
    os.remove('data\\derived_data\\confirmed_sites\\'+f)

# Multiply the number of images
multi_list = []
for filename in os.listdir(r'data\\derived_data\\confirmed_sites'):
    raster = rasterio.open(r'data\\derived_data\\confirmed_sites' + '/' + filename, crs={'init': 'epsg:32637'})
    vert = (raster.bounds[3]-raster.bounds[1])/6  # One sixth of the total length
    hor = (raster.bounds[2]-raster.bounds[0])/6  # One sixth of the total height

    y = [raster.bounds[3]-vert,
         raster.bounds[3]-vert,
         raster.bounds[3]-vert,
         raster.bounds[3]-(3*vert),
         raster.bounds[3]-(3*vert),
         raster.bounds[1]+vert,
         raster.bounds[1]+vert,
         raster.bounds[1]+vert]

    x = [raster.bounds[0]+hor,
         raster.bounds[0]+(3*hor),
         raster.bounds[2]-hor,
         raster.bounds[0]+hor,
         raster.bounds[2]-hor,
         raster.bounds[0]+hor,
         raster.bounds[0]+(3*hor),
         raster.bounds[2]-hor]

    df = pd.DataFrame(list(zip(x, y)), columns=['X', 'Y'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))

    multi_list.append(gdf)

multiplied_sites = pd.concat(multi_list, ignore_index=True)

clip_by_point(POI=multiplied_sites,
              ras="data\\raw_data\\study_area_hillshade_32637_GT_saturated.tif")

# Remove all kinds of findings from the tiles folder to have a clean basis
for f in findings:
    os.remove(f)

for f in findings_negative:
    os.remove(f)
