import rasterio
from scipy.ndimage import zoom
import shutil
from natsort import natsorted
import os
import sqlite3
import numpy as np

def resize():


    maxar_paths_f = "/home/CS/mp0157/dataset/maxar_f/"
    planet_paths_f = "/home/CS/mp0157/dataset/planet_f/"
    # maxar_paths_s = "/home/CS/mp0157/dataset/maxar_scenes_snow/maxar_snow_required/"
    # planet_paths_s = "/home/CS/mp0157/dataset/planet_scenes_snow/planet_snow_required/"

    # file_list_maxar_desert = natsorted(os.listdir(maxar_paths_d))
    # file_list_planet_desert = natsorted(os.listdir(planet_paths_d))
    file_list_maxar_forest = natsorted(os.listdir(maxar_paths_f))
    file_list_planet_forest = natsorted(os.listdir(planet_paths_f))
    # file_list_maxar_snow = natsorted(os.listdir(maxar_paths_s))
    # file_list_planet_snow = natsorted(os.listdir(planet_paths_s))

    conn = sqlite3.connect('/home/CS/mp0157/dataset/DB/embeddings_forest.db')
    cursor = conn.cursor()

    # Create a table to store embeddings and related information
    # TODO: Create a tabel in this format
    # TODO: idx, maxar_tile, m_type, planet_tile, p_type, maxar_metadata, planet_metadata, maxar_tile_name, planet_tile_name
    # cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS embeddings_trail (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         class_type INTEGER,
    #         tile_type VARCHAR(64),
    #         file_name VARCHAR(128),
    #         tile_name VARCHAR(128)
    #     )
    # ''')

    # Create a table to store embeddings and related information
    # TODO: Create a tabel in this format
    # TODO: idx, maxar_tile, m_type, planet_tile, p_type, maxar_metadata, planet_metadata, maxar_tile_name, planet_tile_name

    new_list = file_list_maxar_forest[220: ]

    for m_file in new_list:

        if m_file.endswith(".tif"):
            base_file_name = m_file[: -4]
            print("base_file_name: ",base_file_name)
            # Load the Planet and Maxar images
            maxar_file_path = os.path.join(maxar_paths_f, base_file_name+'.tif')
            planet_file_path = os.path.join(planet_paths_f, base_file_name+'.tif')

            maxar_xml_path = os.path.join(maxar_paths_f, base_file_name+'.xml')
            planet_xml_path = os.path.join(planet_paths_f, base_file_name+'.xml')

            # TODO: Resize both of them to the same zoom level and save the tiles in the database

            # Open the Planet and Maxar images using rasterio
            with rasterio.open(planet_file_path) as planet_ds, rasterio.open(maxar_file_path) as maxar_ds:
                # Read the Planet image data and resize it to the desired size
                maxar_image_data = maxar_ds.read()
                planet_image_data = planet_ds.read()
                desired_height = min(maxar_image_data.shape[1], planet_image_data.shape[1])
                desired_width = min(maxar_image_data.shape[2], planet_image_data.shape[2])

                #TODO: zomm both images to the required zoom level
                resized_planet_data = zoom(planet_image_data, (1, desired_height / planet_image_data.shape[1], desired_width / planet_image_data.shape[2]), order=3)

                resized_maxar_data = zoom(maxar_image_data, (1, desired_height / maxar_image_data.shape[1], desired_width / maxar_image_data.shape[2]), order=3)

            dest_folder_maxar = '/home/CS/mp0157/dataset/maxar_forest_remaining/'
            dest_folder_planet = '/home/CS/mp0157/dataset/planet_forest_remaining/'
            output_path_resized_planet = dest_folder_planet+base_file_name+'.tif'
            output_path_resized_maxar = dest_folder_maxar+base_file_name+'.tif'

            with rasterio.open(output_path_resized_planet, 'w', driver='GTiff', height=desired_height, width=desired_width, count=resized_planet_data.shape[0], dtype=resized_planet_data.dtype, crs=planet_ds.crs, transform=planet_ds.transform) as dst_resized_planet:
                dst_resized_planet.write(resized_planet_data)

            with rasterio.open(output_path_resized_maxar, 'w', driver='GTiff', height=desired_height, width=desired_width, count=resized_maxar_data.shape[0], dtype=resized_maxar_data.dtype, crs=maxar_ds.crs, transform=maxar_ds.transform) as dst_resized_maxar:
                dst_resized_maxar.write(resized_maxar_data)
            
            planet_xml_file_path = os.path.join(dest_folder_planet, f"{base_file_name}.xml")
            maxar_xml_file_path = os.path.join(dest_folder_maxar, f"{base_file_name}.xml")

            shutil.copy2(maxar_xml_path, maxar_xml_file_path)
            shutil.copy2(planet_xml_path, planet_xml_file_path)

            reorder_bands_planet = resized_planet_data[[0, 1, 3, 4, 5, 6, 7], :, :]
            reorder_bands_maxar = resized_maxar_data[[0, 1, 2, 3, 4, 5, 6], :, :]

            print("resized maxar shape: ",resized_maxar_data.shape)
            print("resized planet shape: ",resized_planet_data.shape)
            
            tile_size = (256, 256)
            height, width = resized_maxar_data.shape[1:]
            pad_height = (height // tile_size[0] + 1) * tile_size[0]
            pad_width = (width // tile_size[1] + 1) * tile_size[1]

            # TODO: function using the mode='reflect' parameter in np.pad(). This mode reflects the image values at the boundaries to create padding, resulting in a smoother transition compared to zero padding.
            reorder_bands_maxar = np.pad(reorder_bands_maxar, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode='reflect')
            reorder_bands_planet = np.pad(reorder_bands_planet, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode='reflect')

            tile_no = 0
            maxar_tiles_base_dir = '/home/CS/mp0157/dataset/tiles_maxar_rest/'
            planet_tiles_base_dir = '/home/CS/mp0157/dataset/tiles_planet_rest/'
            for y in range(0, height, tile_size[0]):
                for x in range(0, width, tile_size[1]):
                    m_tile_data = reorder_bands_planet[:, y:y + tile_size[0], x:x + tile_size[1]].astype(np.float32)
                    p_tile_data = reorder_bands_maxar[:, y:y + tile_size[0], x:x + tile_size[1]].astype(np.float32)

                    tile_name_maxar = maxar_tiles_base_dir+base_file_name+'_'+str(tile_no)+'.npy'
                    np.save(tile_name_maxar, m_tile_data)
                    tile_name_planet = planet_tiles_base_dir+base_file_name+'_'+str(tile_no)+'.npy'
                    np.save(tile_name_planet, p_tile_data)
                    data_maxar = (1, "MAXAR", str(output_path_resized_maxar), tile_name_maxar)
                    cursor.execute('INSERT INTO embeddings_trail (class_type, tile_type, file_name, tile_name) VALUES (?,?,?,?)', data_maxar)

                    data_planet = (1, "PLANET", str(output_path_resized_planet), tile_name_planet)
                    cursor.execute('INSERT INTO embeddings_trail (class_type, tile_type, file_name, tile_name) VALUES (?,?,?,?)', data_maxar)

                    tile_no+=1
    conn.commit()
    conn.close()

resize()
