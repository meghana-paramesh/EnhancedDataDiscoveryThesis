
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
from natsort import natsorted
import os


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    print_hi('PyCharm')
    # train()
    maxar_paths_d = "/home/CS/mp0157/dataset/maxar_scenes_desert/"
    planet_paths_d = "/home/CS/mp0157/dataset/planet_scenes_desert/"
    maxar_paths_f = "/home/CS/mp0157/dataset/maxar_scenes_forest/"
    planet_paths_f = "/home/CS/mp0157/dataset/planet_scenes_forest/"
    maxar_paths_s = "/home/CS/mp0157/dataset/maxar_scenes_snow/maxar_snow_required/"
    planet_paths_s = "/home/CS/mp0157/dataset/planet_scenes_snow/planet_snow_required/"

    file_list_maxar_desert = natsorted(os.listdir(maxar_paths_d))
    file_list_planet_desert = natsorted(os.listdir(planet_paths_d))
    file_list_maxar_forest = natsorted(os.listdir(maxar_paths_f))
    file_list_planet_forest = natsorted(os.listdir(planet_paths_f))
    file_list_maxar_snow = natsorted(os.listdir(maxar_paths_s))
    file_list_planet_snow = natsorted(os.listdir(planet_paths_s))

    for m_file in file_list_maxar_forest:

        if m_file.endswith(".tif"):
            base_file_name = m_file[: -4]
            print(base_file_name)

            maxar_file_path = os.path.join(maxar_paths_f, base_file_name+'.tif')
            planet_file_path = os.path.join(planet_paths_f, base_file_name+'.tif')

            maxar_xml_path = os.path.join(maxar_paths_f, base_file_name+'.xml')
            planet_xml_path = os.path.join(planet_paths_f, base_file_name+'.xml')

            m = get_maxar_meta_data(maxar_xml_path, maxar_file_path)
            p = get_planet_meta_data(planet_xml_path, planet_file_path)

            lower_left_latitude = min(m.lower_left_latitude, p.lower_left_latitude)
            lower_left_longitude = min(m.lower_left_longitude, p.lower_left_longitude)

            lower_right_latitude = min(m.lower_right_latitude, p.lower_right_latitude)
            lower_right_longitude = min(m.lower_right_longitude, p.lower_right_longitude)

            upper_right_latitude = min(m.upper_right_latitude, p.upper_right_latitude)
            upper_right_longitude = min(m.upper_right_longitude, p.upper_right_longitude)

            upper_left_latitude = min(m.upper_left_latitude, p.upper_left_latitude)
            upper_left_longitude = min(m.upper_left_longitude, p.upper_left_longitude)

            # Create a polygon representing the bounding box
            bounding_box_polygon = Polygon([(upper_left_longitude, upper_left_latitude),
                                            (upper_right_longitude, upper_right_latitude),
                                            (lower_right_longitude, lower_right_latitude),
                                            (lower_left_longitude, lower_left_latitude)])


            # Open the Planet and Maxar images using rasterio
            with rasterio.open(planet_file_path) as planet_ds, rasterio.open(maxar_file_path) as maxar_ds:
                # Create a shapely box representing the bounding box in the original CRS
                bbox_polygon = box(upper_left_longitude, lower_left_latitude, upper_right_longitude, upper_right_latitude)

                # Reproject the bounding box to the CRS of the Planet image
                try: 
                    # Reproject the bounding box to the CRS of the Planet image
                    bbox_polygon_proj = transform_bounds("EPSG:4326", planet_ds.crs, *bbox_polygon.bounds)

                    # Get the intersection of the bounding box and the Planet image extents
                    intersection_window = planet_ds.window(*bbox_polygon_proj)

                    # Read the intersection region from the Planet image
                    planet_image_array = planet_ds.read(window=intersection_window)

                    # Reproject the bounding box to the CRS of the Maxar image
                    bbox_polygon_proj = transform_bounds("EPSG:4326", maxar_ds.crs, *bbox_polygon.bounds)

                    # Get the intersection of the bounding box and the Maxar image extents
                    intersection_window = maxar_ds.window(*bbox_polygon_proj)

                    # Read the intersection region from the Maxar image
                    maxar_image_array = maxar_ds.read(window=intersection_window)
                



                    # #ordered bands: 0-CoastalBlue, 1-Blue, 3-Green, 4-Yellow, 5-Red, 6-Red Edge, 7-NIR
                    reorder_bands_planet = planet_image_array[[0, 1, 3, 4, 5, 6, 7], :, :]
                    reorder_bands_maxar = maxar_image_array[[0, 1, 2, 3, 4, 5, 6], :, :]


                    print("reorder_bands_planet: ",reorder_bands_planet.shape)
                    print("reorder_bands_maxar: ",reorder_bands_maxar.shape)

                    rgb_planet_resized = reorder_bands_planet[[5, 3, 1], :, :]
                    rgb_maxar_resized = reorder_bands_maxar[[5, 3, 1], :, :]

                    # Save the cropped images (you may need to specify the output profile for GeoTIFF)
                    dest_folder_maxar = '/home/CS/mp0157/dataset/maxar_f/'
                    dest_folder_planet = '/home/CS/mp0157/dataset/planet_f/'
                    planet_xml_file_path = os.path.join(dest_folder_planet, f"{base_file_name}.xml")
                    maxar_xml_file_path = os.path.join(dest_folder_maxar, f"{base_file_name}.xml")


                    output_path_planet = dest_folder_planet+base_file_name+'.tif'
                    output_path_maxar = dest_folder_maxar+base_file_name+'.tif'

                    with rasterio.open(output_path_planet, 'w', driver='GTiff', height=planet_image_array.shape[1],
                                    width=planet_image_array.shape[2], count=planet_image_array.shape[0],
                                    dtype=planet_image_array.dtype, crs=planet_ds.crs,
                                    transform=planet_ds.transform) as dst_planet:
                        dst_planet.write(planet_image_array)

                    with rasterio.open(output_path_maxar, 'w', driver='GTiff', height=maxar_image_array.shape[1],
                                    width=maxar_image_array.shape[2], count=maxar_image_array.shape[0],
                                    dtype=maxar_image_array.dtype, crs=maxar_ds.crs, transform=maxar_ds.transform) as dst_maxar:
                        dst_maxar.write(maxar_image_array)
                    shutil.copy2(maxar_xml_path, maxar_xml_file_path)
                    shutil.copy2(planet_xml_path, planet_xml_file_path)
                except rasterio.errors.RasterioIOError:
                    print("The bounding box projection is invalid. Cannot perform the intersection.")
                    continue

