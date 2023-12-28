from shapely.geometry import Polygon
from Meta_Data_Extracter import get_planet_meta_data, get_maxar_meta_data
import os
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
# from osgeo import gdal, gdal_array, osr, ogr
import tempfile


def check_intersection(m_xml, m_file, p_xml, p_file):

    m = get_maxar_meta_data(m_xml, m_file)
    p = get_planet_meta_data(p_xml, p_file)
    # print("For maxar: ", float(m.lower_left_latitude))
    # # print("For maxar: ", m.lower_left_latitude, m.lower_left_longitude, m.lower_right_latitude, m.lower_right_longitude,
    #       m.upper_right_latitude, m.upper_right_longitude, m.upper_left_latitude, m.upper_left_longitude)
    # print("For planet: ", p.lower_left_latitude, p.lower_left_longitude, p.lower_right_latitude,
    #       p.lower_right_longitude, p.upper_right_latitude, p.upper_right_longitude, p.upper_left_latitude,
    #       p.upper_left_longitude)
    # Define the coordinates of rectangle 1
    x = m.lower_left_latitude
    rect1_coords = [(float(x), m.lower_left_longitude), (m.lower_right_latitude, m.lower_right_longitude),
                    (m.upper_right_latitude, m.upper_right_longitude), (m.upper_left_latitude, m.upper_left_longitude)]

    # Define the coordinates of rectangle 2
    rect2_coords = [(p.lower_left_latitude, p.lower_left_longitude), (p.lower_right_latitude, p.lower_right_longitude),
                    (p.upper_right_latitude, p.upper_right_longitude), (p.upper_left_latitude, p.upper_left_longitude)]

    rect1 = Polygon(rect1_coords)
    rect2 = Polygon(rect2_coords)

    # Check for intersection
    intersection = rect1.intersects(rect2)

    return intersection

    # Example coordinates of two rectangles


def check_intersection_and_create_new_files(maxar_folder_dir, planet_folder_dir, dest_dir):
    i = 146
    for m_file in os.listdir(maxar_folder_dir):

        # if the maxar file is ntf them check intersection with all the tiff files
        if m_file.endswith('.ntf'):
            m_xml_file = m_file.replace(".ntf", ".xml")
            m_item_path_xml = os.path.join(maxar_folder_dir, m_xml_file)
            m_item_path_file = os.path.join(maxar_folder_dir, m_file)
            for p_file in os.listdir(planet_folder_dir):

                if p_file.endswith('.tif') and p_file.__contains__("AnalyticMS"):
                    p_xml_file = p_file.replace("SR_8b_clip.tif", "8b_metadata_clip.xml")
                    # print("p_xml_file: ", p_xml_file)
                    p_item_path_xml = os.path.join(planet_folder_dir, p_xml_file)
                    # print("p_item_path_xml: ", p_item_path_xml)
                    p_item_path_file = os.path.join(planet_folder_dir, p_file)
                    if check_intersection(m_item_path_xml, m_file, p_item_path_xml, p_file):
                        m_date = m_file[5:13]
                        m_new_filename = f"desert_{i}_{m_date}.ntf"
                        m_new_filename_xml = f"desert_{i}_{m_date}_maxar_metadata.xml"
                        p_date = p_file[:8]
                        p_new_filename = f"desert_{i}_{p_date}.tif"
                        p_new_filename_xml = f"desert_{i}_{p_date}_planet_metadata.xml"
                        i = i + 1

                        # for maxar
                        m_new_filepath = os.path.join(dest_dir, m_new_filename)
                        m_new_filepath_xml = os.path.join(dest_dir, m_new_filename_xml)

                        # for planet
                        p_new_filepath = os.path.join(dest_dir, p_new_filename)
                        p_new_filepath_xml = os.path.join(dest_dir, p_new_filename_xml)

                        # copy all 4
                        shutil.copy2(m_item_path_file, m_new_filepath)
                        shutil.copy2(m_item_path_xml, m_new_filepath_xml)

                        shutil.copy2(p_item_path_file, p_new_filepath)
                        shutil.copy2(p_item_path_xml, p_new_filepath_xml)


def plot_time_diff(maxar_paths, planet_paths):
    date_format = "%Y%m%d"
    diff=[]
    for i in range(len(maxar_paths)):
        for m_file in os.listdir(maxar_paths[i]):

            # if the maxar file is ntf them check intersection with all the tiff files
            if m_file.endswith('.ntf'):
                m_xml_file = m_file.replace(".ntf", ".xml")
                m_item_path_xml = os.path.join(maxar_paths[i], m_xml_file)
                m_item_path_file = os.path.join(maxar_paths[i], m_file)
                for p_file in os.listdir(planet_paths[i]):
                    if p_file.endswith('.tif') and p_file.__contains__("AnalyticMS") and not p_file.startswith("."):
                        p_xml_file = p_file.replace("SR_8b_clip.tif", "8b_metadata_clip.xml")
                        # print("p_xml_file: ", p_xml_file)
                        p_item_path_xml = os.path.join(planet_paths[i], p_xml_file)
                        # print("p_item_path_xml: ", p_item_path_xml)
                        # p_item_path_file = os.path.join(planet_folder_dir, p_file)
                        if check_intersection(m_item_path_xml, m_file, p_item_path_xml, p_file):
                            # m_date = m_file[5:13]
                            # p_date = p_file[:8]
                            # date1 = datetime.strptime(m_date, date_format).date()
                            # date2 = datetime.strptime(p_date, date_format).date()
                            # # print("date1: ",date1, "date2: ", date2)
                            # if date1 > date2:
                            #     difference = (date1 - date2).days
                            # else:
                            #     difference = (date2 - date1).days

                            # if difference > 15:
                            #     print("date1: ",date1, "date2: ", date2)
                            #     print("anamoly")
                            #     print("corresponding m_file: ",m_file)
                            #     print("corresponding p_file: ",planet_paths[i]+p_file)


                            # diff.append(difference)
                            print(maxar_paths[i])
                            m_file_path = os.path.join(maxar_paths[i], m_file)
                            m_dataset_prev = gdal.Open(m_file_path)
                            p_file_path = os.path.join(planet_paths[i], p_file)
                            p_dataset = gdal.Open(p_file_path)
                            p_projection = p_dataset.GetProjection()

                            # Create a temporary file for the in-memory dataset
                            temp_filename = tempfile.NamedTemporaryFile(suffix='.tif').name

                            # Perform the projection using gdal.Warp()
                            m_dataset = gdal.Warp(temp_filename, p_dataset, dstSRS=p_projection)
                            ntf_width = m_dataset.RasterXSize
                            ntf_height = m_dataset.RasterYSize

                            # Get the geospatial information of the TIF image
                            tif_geotransform = p_dataset.GetGeoTransform()
                            tif_width = p_dataset.RasterXSize
                            tif_height = p_dataset.RasterYSize

                            # Calculate the common size
                            common_width = min(ntf_width, tif_width)
                            common_height = min(ntf_height, tif_height)

                            # Calculate the window offset to center the cropping
                            x_offset = int((ntf_width - common_width) / 2)
                            y_offset = int((ntf_height - common_height) / 2)

                            # Read the NTF image subset
                            ntf_subset = m_dataset.ReadAsArray(x_offset, y_offset, common_width, common_height)

                            # Read the TIF image subset
                            tif_subset = p_dataset.ReadAsArray(0, 0, common_width, common_height)
                            _tile_image(ntf_subset, tif_subset, m_file_path, p_file_path, p_dataset)
    # return diff


def _tile_image(m_data, p_data, m_file, p_file, dataset):
    tile_size = 512
    overlap = 0

    data_shape = m_data.shape
    band = data_shape[0]
    height = data_shape[1]
    width = data_shape[2]
    print(data_shape)

    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    i = 0
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Compute tile coordinates
            tile_x = i * tile_size
            tile_y = j * tile_size

            # Extract tile data
            m_tile_data = m_data[ :, tile_y:tile_y + tile_size, tile_x:tile_x + tile_size]
            p_tile_data = m_data[ :, tile_y:tile_y + tile_size, tile_x:tile_x + tile_size]

            # maxar_tiles.append(ntf_tile)
            # planet_tiles.append(tif_tile)
            
