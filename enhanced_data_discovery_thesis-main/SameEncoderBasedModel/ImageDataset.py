import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import os
from Meta_Data_Extracter import get_planet_meta_data, get_maxar_meta_data
import tempfile
from rasterio.windows import from_bounds, intersection


## TODO 1: accept both planet and maxar image assign metadata for each of them and return everything as
# a key value pair

## TODO 2: resign planet and maxar based on the max size of both of them before applying the tiling
## TODO 3: Reorder the images to match the band. First one - Planet, second= Maxar
## 1	Coastal Blue	443 (20)	Yes - with Sentinel-2 band 1  Coastal Blue
## 2	Blue	490 (50)	Yes - with Sentinel-2 band 2 Blue
## 3	Green I	531 (36)	No equivalent with Sentinel-2 Green
## 4	Green	565 (36)	Yes - with Sentinel-2 band 3 Yellow
## 5	Yellow	610 (20)	No equivalent with Sentinel-2 Red
## 6	Red	665 (31)	Yes - with Sentinel-2 band 4 RedEdge
## 7	Red Edge	705 (15)	Yes - with Sentinel-2 band 5  NIR1
## 8	NIR NIR2

class ImageDataset(Dataset):
    def __init__(self, load_train_images, load_test_images, maxar_desert, maxar_desert_list, planet_desert, planet_desert_list, maxar_snow, maxar_snow_list, planet_snow, planet_snow_list, maxar_forest, maxar_forest_list, planet_forest, planet_forest_list):
        self.maxar_tiles = []
        self.planet_tiles = []

        self.m_types = []
        self.p_types = []
        self.load_test_images = load_test_images
        self.load_train_images = load_train_images

        if self.load_test_images:
            self.maxar_files = []
            self.planet_files = []
            self.maxar_tile_names = []
            self.planet_tile_names = []

        self._load_tiles("desert", maxar_desert, maxar_desert_list, planet_desert, planet_desert_list)
        self._load_tiles("snow", maxar_snow, maxar_snow_list, planet_snow, planet_snow_list)
        self._load_tiles("forest", maxar_forest, maxar_forest_list, planet_forest, planet_forest_list)

    def __len__(self):
        return len(self.maxar_tiles)

    def __getitem__(self, idx):
        maxar_tile = self.maxar_tiles[idx]
        planet_tile = self.planet_tiles[idx]

        m_type = self.m_types[idx]
        p_type = self.p_types[idx]

        # maxar_metadata = self.maxar_metadata[idx]
        # planet_metadata = self.planet_metadata[idx]

        if self.load_test_images:
            maxar_file_path = self.maxar_files[idx]
            planet_file_path = self.planet_files[idx]

            maxar_tile_name = self.maxar_tile_names[idx]
            planet_tile_name = self.planet_tile_names[idx]


            return (maxar_tile, m_type, planet_tile, p_type, maxar_file_path, planet_file_path, maxar_tile_name, planet_tile_name)
        else:
            return (maxar_tile, m_type, planet_tile, p_type)

    def _load_tiles(self, folder_type, maxar_folder, maxar_list, planet_folder, planet_list):
        i = 0
        for m_file in maxar_list:
            #TODO: Get the maxar path

            if m_file.endswith(".tif"):
            
                maxar_file_path = os.path.join(maxar_folder, m_file)

                planet_file_path = os.path.join(planet_folder, m_file)

                print("maxar_file_path: ",maxar_file_path, "i is", i)
                print("planet_file_path: ",planet_file_path)
                i=i+1
                with rasterio.open(maxar_file_path) as maxar_dataset, rasterio.open(planet_file_path) as planet_dataset:
                    # Read the raster data and metadata for both images
                    maxar_data = maxar_dataset.read()
                    planet_data = planet_dataset.read()

                    # Calculate the common window
                    window_maxar = from_bounds(*maxar_dataset.bounds, transform=maxar_dataset.transform)
                    window_planet = from_bounds(*planet_dataset.bounds, transform=planet_dataset.transform)
                    common_window = intersection(window_maxar, window_planet)

                    # Crop the images to the common window
                    cropped_maxar = maxar_dataset.read(window=common_window)
                    cropped_planet = planet_dataset.read(window=common_window)

                    reorder_bands_planet = cropped_planet[[0, 1, 3, 4, 5, 6, 7], :, :]
                    reorder_bands_maxar = cropped_maxar[[0, 1, 2, 3, 4, 5, 6], :, :]

                # Reshape the cropped arrays into tiles of size 512x512
                tile_size = (256, 256)
                if self.load_test_images:
                    # maxar_metadata = self._load_metadata("maxar", maxar_file_path)
                    # planet_metadata = self._load_metadata("planet", planet_file_path)
                    self.create_tiles_save_with_metadata(m_file, reorder_bands_maxar, reorder_bands_planet, tile_size, maxar_file_path, planet_file_path)
                else:
                    self.create_tiles(folder_type, reorder_bands_maxar, reorder_bands_planet, tile_size)
            # TODO: For the test purpose only. Needs removal
    
    def create_tiles_save_with_metadata(self, file_name, maxar_data, planet_data, tile_size, maxar_file, planet_file):
        height, width = maxar_data.shape[1:]
        pad_height = (height // tile_size[0] + 1) * tile_size[0]
        pad_width = (width // tile_size[1] + 1) * tile_size[1]

        # TODO: function using the mode='reflect' parameter in np.pad(). This mode reflects the image values at the boundaries to create padding, resulting in a smoother transition compared to zero padding.
        maxar_data = np.pad(maxar_data, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode='reflect')
        planet_data = np.pad(planet_data, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode='reflect')
        tile_no=0
        for y in range(0, height, tile_size[0]):
            for x in range(0, width, tile_size[1]):
                m_tile_data = maxar_data[:, y:y+tile_size[0], x:x+tile_size[1]].astype(np.float32)
                p_tile_data = planet_data[:, y:y+tile_size[0], x:x+tile_size[1]].astype(np.float32)
                file_name = file_name.removesuffix('.tif') 
                tile_name = 'tiles_all_proper_bands_order_new/'+file_name+'_'+str(tile_no)
                if "desert" in file_name:
                    self.maxar_tiles.append(m_tile_data)
                    np.save(tile_name+'_maxar.npy', p_tile_data)
                    self.m_types.append(0)
                    self.maxar_files.append(maxar_file)
                    self.maxar_tile_names.append(tile_name+'_maxar.npy')

                    self.planet_tiles.append(p_tile_data)
                    np.save(tile_name+'_planet.npy', p_tile_data)
                    self.p_types.append(0)
                    self.planet_files.append(planet_file)
                    self.planet_tile_names.append(tile_name+'_planet.npy')
                elif "snow" in file_name:
                    self.maxar_tiles.append(m_tile_data)
                    np.save(tile_name+'_maxar.npy', p_tile_data)
                    self.maxar_files.append(maxar_file)
                    self.m_types.append(2)
                    self.maxar_tile_names.append(tile_name+'_maxar.npy')

                    self.planet_tiles.append(p_tile_data)
                    np.save(tile_name+'_planet.npy', p_tile_data)
                    self.p_types.append(2)
                    self.planet_files.append(planet_file)
                    self.planet_tile_names.append(tile_name+'_planet.npy')
                else:
                    self.maxar_tiles.append(m_tile_data)
                    np.save(tile_name+'_maxar.npy', p_tile_data)
                    self.maxar_files.append(maxar_file)
                    self.m_types.append(1)
                    self.maxar_tile_names.append(tile_name+'_maxar.npy')

                    self.planet_tiles.append(p_tile_data)
                    np.save(tile_name+'_planet.npy', p_tile_data)
                    self.p_types.append(1)
                    self.planet_files.append(planet_file)
                    self.planet_tile_names.append(tile_name+'_planet.npy')
                tile_no+=1

    def create_tiles(self, file_type, maxar_data, planet_data, tile_size):
        height, width = maxar_data.shape[1:]
        pad_height = (height // tile_size[0] + 1) * tile_size[0]
        pad_width = (width // tile_size[1] + 1) * tile_size[1]

        # TODO: function using the mode='reflect' parameter in np.pad(). This mode reflects the image values at the boundaries to create padding, resulting in a smoother transition compared to zero padding.
        maxar_data = np.pad(maxar_data, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode='reflect')
        planet_data = np.pad(planet_data, ((0, 0), (0, pad_height - height), (0, pad_width - width)), mode='reflect')
        for y in range(0, height, tile_size[0]):
            for x in range(0, width, tile_size[1]):
                m_tile_data = maxar_data[:, y:y+tile_size[0], x:x+tile_size[1]].astype(np.float32)
                p_tile_data = planet_data[:, y:y+tile_size[0], x:x+tile_size[1]].astype(np.float32)
                if file_type == "desert":
                    self.maxar_tiles.append(m_tile_data)
                    self.m_types.append(0)
                    self.planet_tiles.append(p_tile_data)
                    self.p_types.append(0)
                elif file_type == "snow":
                    self.maxar_tiles.append(m_tile_data)
                    self.m_types.append(1)
                    self.planet_tiles.append(p_tile_data)
                    self.p_types.append(1)
                else:
                    self.maxar_tiles.append(m_tile_data)
                    self.m_types.append(2)
                    self.planet_tiles.append(p_tile_data)
                    self.p_types.append(2)


    def _load_metadata(self, type, filename):

        if type=="maxar" and filename.endswith(".tif"):
            xml_file = filename.replace(".tif", ".xml")
            data = get_maxar_meta_data(xml_file, filename)
        else:
            xml_file = filename.replace(".tif", ".xml")
            data = get_planet_meta_data(xml_file, filename)

        return data

def max_min_normalize(tensor):
    # Min-max normalization
    min_value = tensor.min()
    max_value = tensor.max()
    tensor = (tensor - min_value) / (max_value - min_value)
    return tensor


# Example usage
# "/Users/meghananp/Documents/Summer2023/dataset/snow_covered_planet/snow_covered_10300100A40CB600_2020_04_08_original_April4_instead_psscene_analytic_8b_sr_udm2/PSScene/20200404_204056_90_2263_3B_AnalyticMS_SR_8b.tif"
# maxar_filenames = [
#     "/Users/meghananp/Documents/Summer2023/dataset/snow_covered_maxar/Meghana/10300100A40CB600_M1BS/WV02_20200408214137_10300100A40CB600_20APR08214137-M1BS-504232271090_01_P002.ntf"]
# planet_filenames = [
#     "/Users/meghananp/Documents/Summer2023/dataset/snow_covered_planet/snow_covered_10300100A40CB600_2020_04_08_original_April4_instead_psscene_analytic_8b_sr_udm2/PSScene/20200404_204056_90_2263_3B_AnalyticMS_SR_8b.tif"]
# dataset = ImageDataset(maxar_filenames, planet_filenames)
# print(dataset.__len__())
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
# print(dataloader.__len__())
