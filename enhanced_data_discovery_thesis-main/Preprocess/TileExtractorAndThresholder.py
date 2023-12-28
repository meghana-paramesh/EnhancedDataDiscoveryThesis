import os
import rasterio
from rasterio.windows import from_bounds, intersection
from skimage.transform import resize
from matplotlib.image import imsave
import numpy as np
from rasterio.windows import Window
from rasterio.enums import Resampling
from Meta_Data_Extracter import get_maxar_meta_data, get_planet_meta_data
from shapely.geometry import Polygon
from rasterio.warp import transform_geom
from shapely.geometry import box
from rasterio.warp import transform_bounds
import shutil
import os
from scipy.ndimage import zoom
import sqlite3
from natsort import natsorted



def contains_mostly_zero(tile):
    threshold = 0.8
    return np.count_nonzero(tile == 0) / tile.size >= threshold


def tile_extractor():

    maxar_paths_u = "/home/CS/mp0157/dataset/IRR_NEW/MAXAR_IRR/"
    planet_paths_u = "/home/CS/mp0157/dataset/IRR_NEW/PLANET_IRR/"

    file_list_maxar_irrigation = natsorted(os.listdir(maxar_paths_u))
    file_list_planet_irrigation = natsorted(os.listdir(planet_paths_u))

    list_maxar = file_list_maxar_irrigation[400:450]

    for m_file in list_maxar:

        if m_file.endswith(".tif"):
            base_file_name = m_file[: -4]
            print(base_file_name)
            maxar_file_path = os.path.join(maxar_paths_u, base_file_name + '.tif')
            planet_file_path = os.path.join(planet_paths_u, base_file_name + '.tif')

            maxar_xml_path = os.path.join(maxar_paths_u, base_file_name + '.xml')
            planet_xml_path = os.path.join(planet_paths_u, base_file_name + '.xml')

            print("maxar_xml_path: ",maxar_xml_path)
            maxar_obj = get_maxar_meta_data(maxar_xml_path, maxar_file_path)
            planet_obj = get_planet_meta_data(planet_xml_path, planet_file_path)

            lower_left_latitude = min(maxar_obj.lower_left_latitude, planet_obj.lower_left_latitude)
            lower_left_longitude = min(maxar_obj.lower_left_longitude, planet_obj.lower_left_longitude)

            lower_right_latitude = min(maxar_obj.lower_right_latitude, planet_obj.lower_right_latitude)
            lower_right_longitude = min(maxar_obj.lower_right_longitude, planet_obj.lower_right_longitude)

            upper_right_latitude = min(maxar_obj.upper_right_latitude, planet_obj.upper_right_latitude)
            upper_right_longitude = min(maxar_obj.upper_right_longitude, planet_obj.upper_right_longitude)

            upper_left_latitude = min(maxar_obj.upper_left_latitude, planet_obj.upper_left_latitude)
            upper_left_longitude = min(maxar_obj.upper_left_longitude, planet_obj.upper_left_longitude)
            try:

                with rasterio.open(planet_file_path) as planet_ds, rasterio.open(maxar_file_path) as maxar_ds:
                    # Create a shapely box representing the bounding box in the original CRS
                    bbox_polygon = box(upper_left_longitude, lower_left_latitude, upper_right_longitude,
                                    upper_right_latitude)

                    bbox_polygon_proj = transform_bounds("EPSG:4326", planet_ds.crs, *bbox_polygon.bounds)

                    intersection_window = planet_ds.window(*bbox_polygon_proj)

                    planet_image_data = planet_ds.read(window=intersection_window)

                    bbox_polygon_proj = transform_bounds("EPSG:4326", maxar_ds.crs, *bbox_polygon.bounds)
                    intersection_window = maxar_ds.window(*bbox_polygon_proj)

                    maxar_image_data = maxar_ds.read(window=intersection_window)



                    desired_height = min(planet_image_data.shape[1], maxar_image_data.shape[1])
                    desired_width = min(planet_image_data.shape[2], maxar_image_data.shape[2])

                    if desired_height==0 or desired_width==0:
                        continue

                    # TODO: Resize both of them to the same zoom level and save the tiles in the database

                    # TODO: zoom both images to the required zoom level
                    resized_planet_data = zoom(planet_image_data, (
                    1, desired_height / planet_image_data.shape[1], desired_width / planet_image_data.shape[2]),
                                            order=3)

                    resized_maxar_data = zoom(maxar_image_data, (
                    1, desired_height / maxar_image_data.shape[1], desired_width / maxar_image_data.shape[2]), order=3)

                    print("before resized maxar shape: ", resized_maxar_data.shape)
                    print("before resized planet shape: ", resized_planet_data.shape)

                    if resized_maxar_data.shape[0] < 8:
                        continue


                    dest_folder_maxar = '/home/CS/mp0157/dataset/MAXAR_IRRIGATION_PREPROCESSED_NEW/'
                    dest_folder_planet = '/home/CS/mp0157/dataset/PLANET_IRRIGATION_PREPROCESSED_NEW/'
                    output_path_resized_planet = dest_folder_planet + base_file_name + '.tif'
                    output_path_resized_maxar = dest_folder_maxar + base_file_name + '.tif'

                    with rasterio.open(output_path_resized_planet, 'w', driver='GTiff', height=desired_height,
                                    width=desired_width, count=resized_planet_data.shape[0],
                                    dtype=resized_planet_data.dtype, crs=planet_ds.crs,
                                    transform=planet_ds.transform) as dst_resized_planet:
                        dst_resized_planet.write(resized_planet_data)

                    with rasterio.open(output_path_resized_maxar, 'w', driver='GTiff', height=desired_height,
                                    width=desired_width, count=resized_maxar_data.shape[0],
                                    dtype=resized_maxar_data.dtype, crs=maxar_ds.crs,
                                    transform=maxar_ds.transform) as dst_resized_maxar:
                        dst_resized_maxar.write(resized_maxar_data)

                    planet_xml_file_path = os.path.join(dest_folder_planet, f"{base_file_name}.xml")
                    maxar_xml_file_path = os.path.join(dest_folder_maxar, f"{base_file_name}.xml")

                    shutil.copy2(maxar_xml_path, maxar_xml_file_path)
                    shutil.copy2(planet_xml_path, planet_xml_file_path)


                    reorder_bands_planet = resized_planet_data[[0, 1, 3, 4, 5, 6, 7], :, :]
                    reorder_bands_maxar = resized_maxar_data[[0, 1, 2, 3, 4, 5, 6], :, :]

                    print("resized maxar shape: ", reorder_bands_maxar.shape)
                    print("resized planet shape: ", reorder_bands_planet.shape)

                    tile_size = (256, 256)
                    height, width = reorder_bands_maxar.shape[1:]
                    pad_height = (height // tile_size[0] + 1) * tile_size[0]
                    pad_width = (width // tile_size[1] + 1) * tile_size[1]

                    tile_no = 0

                    maxar_tiles_base_dir = '/home/CS/mp0157/dataset/new_tiles_test/maxar_irr_tiles/'
                    planet_tiles_base_dir = '/home/CS/mp0157/dataset/new_tiles_test/planet_irr_tiles/'
                    for y in range(0, height, tile_size[0]):
                        for x in range(0, width, tile_size[1]):
                            
                            m_tile_data = reorder_bands_planet[:, y:y + tile_size[0], x:x + tile_size[1]].astype(
                                np.float32)
                            p_tile_data = reorder_bands_maxar[:, y:y + tile_size[0], x:x + tile_size[1]].astype(
                                np.float32)
                            if m_tile_data.shape[1]==256 and p_tile_data.shape[1]==256 and m_tile_data.shape[2]==256 and p_tile_data.shape[2]==256:
                                if contains_mostly_zero(m_tile_data) or contains_mostly_zero(p_tile_data):
                                    continue

                                tile_name_maxar = maxar_tiles_base_dir + base_file_name + '_' + str(tile_no) + '.npy'
                                np.save(tile_name_maxar, m_tile_data)
                                tile_name_planet = planet_tiles_base_dir + base_file_name + '_' + str(tile_no) + '.npy'
                                np.save(tile_name_planet, p_tile_data)
                                tile_no+=1
            except rasterio.errors.RasterioIOError:
                print("The bounding box projection is invalid. Cannot perform the intersection.")
                continue
            

tile_extractor()




