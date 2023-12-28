from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os


##1 	Coastal Blue 	443 (20) 	Yes - with Sentinel-2 band 1
##2 	Blue 	490 (50) 	Yes - with Sentinel-2 band 2
##3 	Green I 	531 (36) 	No equivalent with Sentinel-2
##4 	Green 	565 (36) 	Yes - with Sentinel-2 band 3
##5 	Yellow 	610 (20) 	No equivalent with Sentinel-2
##6 	Red 	665 (31) 	Yes - with Sentinel-2 band 4
##7 	Red Edge 	705 (15) 	Yes - with Sentinel-2 band 5
##8 	NIR 	865 (40) 	Yes - with Sentinel-2 band 8a
def display_tiles():
    planet_image_path = '/Volumes/Untitled/Meghana/REQUIRED/SEGREGATED/dataset/snow_covered_planet/snow_covered_10300100A40CB600_2020_04_08_original_April4_instead_psscene_analytic_8b_sr_udm2/PSScene/20200404_204101_29_2263_3B_AnalyticMS_SR_8b.tif'
    maxar_image_path = '/Volumes/Untitled/Meghana/REQUIRED/SEGREGATED/dataset/snow_covered_maxar/Meghana/10300100A40CB600_M1BS/WV02_20200408214139_10300100A40CB600_20APR08214139-M1BS-504232271090_01_P004.ntf'

    (p_data, planet_x, planet_y, planet_bands) = get_num_tile_xy(planet_image_path, "P")
    (m_data, maxar_x, maxar_y, maxar_bands) = get_num_tile_xy(maxar_image_path, "M")
    overlap = 0
    tile_size = 512


    i=0
    for i in range(planet_x):
        for j in range(planet_y):
            print(i)
            i=i+1
            # fig = plt.figure(figsize=(10, 7))
            # rows = 2
            # columns = 2
            x = i * (planet_x - overlap)
            y = j * (planet_y - overlap)
            p_tile = p_data.ReadAsArray(x, y, tile_size, tile_size)
            m_tile = p_data.ReadAsArray(x, y, tile_size, tile_size)

            # Extract RGB bands (assuming bands 1, 2, 3 are RGB)
            rgb_P_bands = p_tile[planet_bands, :, :]
            rgb_M_bands = m_tile[maxar_bands, :, :]

            # Display the tile using Matplotlib
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(rgb_P_bands.transpose(1, 2, 0))
            # plt.title("Planet")
            plt.axis('off')
            # fig.add_subplot(rows, columns, 2)
            axs[1].imshow(rgb_M_bands.transpose(1, 2, 0))
            # plt.title("Maxar")
            plt.axis('off')
            plt.show()
            # break

        dataset = None


def get_num_tile_xy(image_path, type):
    dataset = gdal.Open(image_path)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    num_bands = dataset.RasterCount

    tile_size = 512
    overlap = 0
    num_tiles_x = (width - overlap) // (tile_size - overlap)
    num_tiles_y = (height - overlap) // (tile_size - overlap)

    if type=="P":
        selected_bands = [2, 4, 6]

    if type=="M":
        selected_bands = [2, 3, 5]

    return dataset, num_tiles_x, num_tiles_y, selected_bands

def display_tiles(tif_path, ntf_path, output_dir):
    tif_dataset = gdal.Open(tif_path)
    ntf_dataset = gdal.Open(ntf_path)

    tile_size = 256
    overlap = 0

    tif_width = tif_dataset.RasterXSize
    tif_height = tif_dataset.RasterYSize
    ntf_width = ntf_dataset.RasterXSize
    ntf_height = ntf_dataset.RasterYSize

    num_tiles_x = (tif_width - overlap) // (tile_size - overlap)
    num_tiles_y = (tif_height - overlap) // (tile_size - overlap)

    fig, axes = plt.subplots(num_tiles_y, num_tiles_x, figsize=(12, 8))

    for y in range(0, tif_height, tile_size - overlap):
        for x in range(0, tif_width, tile_size - overlap):
            if x + tile_size > tif_width:
                x = tif_width - tile_size
            if y + tile_size > tif_height:
                y = tif_height - tile_size

            tif_tile = tif_dataset.ReadAsArray(x, y, tile_size, tile_size)[[2, 4, 6], :, :]
            ntf_tile = ntf_dataset.ReadAsArray(x, y, tile_size, tile_size)[[2, 3, 5], :, :]

            x_idx = x // (tile_size - overlap)
            y_idx = y // (tile_size - overlap)

            composite_image = np.concatenate((tif_tile, ntf_tile), axis=1)

            # Normalize pixel values to [0, 1]
            composite_image = composite_image / 255.0
            output_path = os.path.join(output_dir, f"tile_{x_idx}_{y_idx}.png")

            plt.imshow(np.transpose(composite_image, (1, 2, 0)))
            plt.axis('off')
            plt.savefig(output_path, dpi=300)
            plt.close()

    plt.tight_layout()
    plt.show()

    tif_dataset = None
    ntf_dataset = None


