import os
from osgeo import gdal
import shutil
from shapely.geometry import Polygon
from Meta_Data_Extracter import get_planet_meta_data, get_maxar_meta_data

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

def get_all_tiles_to_the_same_projection_and_save(maxar_paths, planet_paths, dest_maxar_dir, dest_planet_dir):
    j=0
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
                        print("p_item_path_xml: ", p_item_path_xml)
                        # p_item_path_file = os.path.join(planet_folder_dir, p_file)

                        # TODO: Once the intersection is found then project it to the same projection
                        # TODO: Open each image using gdal and attach a label
                        # TODO: If the intersection is present then make the tiles out of them and attach the type: desert(0), forest(1), snow(2) with each

                        if check_intersection(m_item_path_xml, m_file, p_item_path_xml, p_file):
                            m_file_path = os.path.join(maxar_paths[i], m_file)
                            print(m_file_path)
                            m_dataset_prev = gdal.Open(m_file_path)
                            p_file_path = os.path.join(planet_paths[i], p_file)
                            p_dataset = gdal.Open(p_file_path)
                            p_projection = p_dataset.GetProjection()

                            # Create a temporary file for the in-memory dataset
                            # Perform the projection using gdal.Warp()
                            maxar_file_path = os.path.join(dest_maxar_dir, f"snow_{j}.tif")
                            maxar_xml_file_path = os.path.join(dest_maxar_dir, f"snow_{j}.xml")
                            gdal.Warp(maxar_file_path, m_dataset_prev, dstSRS=p_projection)
                            planet_file_path = os.path.join(dest_planet_dir, f"snow_{j}.tif")
                            planet_xml_file_path = os.path.join(dest_planet_dir, f"snow_{j}.xml")
                            j=j+1
                            shutil.copy2(p_file_path, planet_file_path)

                            # TODO: save the xml file for the maxar and the planet file
                            shutil.copy2(m_item_path_xml, maxar_xml_file_path)
                            shutil.copy2(p_item_path_xml, planet_xml_file_path)

                            # Close the datasets
                            ntf_dataset = None
                            tif_dataset = None
                            output_dataset = None

                            
