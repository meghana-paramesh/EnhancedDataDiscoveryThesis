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

    list_maxar = file_list_maxar_irrigation[450:700]

    #train tiles
    create_tiles(list_maxar, '/home/CS/mp0157/dataset/new_tiles_original_inference/maxar_irrigation_tiles/', '/home/CS/mp0157/dataset/new_tiles_original_inference/planet_irrigation_tiles/', maxar_paths_u, planet_paths_u)

    #inference tiles
    # list_maxar = file_list_maxar_irrigation[500: 700]
    # create_tiles(list_maxar, '/home/CS/mp0157/dataset/maxar_tiles_original_inference/', '/home/CS/mp0157/dataset/planet_tiles_original_inference/', maxar_paths_u, planet_paths_u)

    # #test tiles
    list_maxar = file_list_maxar_irrigation[400: 450]
    create_tiles(list_maxar, '/home/CS/mp0157/dataset/new_tiles_original_inference/maxar_irrigation_tiles/', '/home/CS/mp0157/dataset/new_tiles_original_inference/planet_irrigation_tiles/', maxar_paths_u, planet_paths_u)



def create_tiles(list_maxar, maxar_tiles_base_dir, planet_tiles_base_dir, maxar_paths_u, planet_paths_u):
    for m_file in list_maxar:

        if m_file.endswith(".tif"):
            base_file_name = m_file[: -4]
            print(base_file_name)
            maxar_file_path = os.path.join(maxar_paths_u, base_file_name + '.tif')
            planet_file_path = os.path.join(planet_paths_u, base_file_name + '.tif')

            with rasterio.open(planet_file_path) as planet_ds, rasterio.open(maxar_file_path) as maxar_ds:
                maxar_image_data = maxar_ds.read()
                planet_image_data = planet_ds.read()
                tile_no = 0

                tile_size = (256, 256)
                height, width = maxar_image_data.shape[1:]
                pad_height = (height // tile_size[0] + 1) * tile_size[0]
                pad_width = (width // tile_size[1] + 1) * tile_size[1]

                for y in range(0, height, tile_size[0]):
                    for x in range(0, width, tile_size[1]):
                        m_tile_data = planet_image_data[:, y:y + tile_size[0], x:x + tile_size[1]].astype(np.float32)
                        p_tile_data = maxar_image_data[:, y:y + tile_size[0], x:x + tile_size[1]].astype(np.float32)
                        if m_tile_data.shape[1]==256 and p_tile_data.shape[1]==256 and m_tile_data.shape[2]==256 and p_tile_data.shape[2]==256:
                            if m_tile_data.size>0:
                                if not contains_mostly_zero(m_tile_data):
                                    tile_name_maxar = maxar_tiles_base_dir + base_file_name + '_' + str(tile_no) + '.npy'
                                    np.save(tile_name_maxar, m_tile_data)
                            if p_tile_data.size>0:
                                if not contains_mostly_zero(p_tile_data):
                                    tile_name_planet = planet_tiles_base_dir + base_file_name + '_' + str(tile_no) + '.npy'
                                    np.save(tile_name_planet, p_tile_data)
                            tile_no+=1
            

tile_extractor()




