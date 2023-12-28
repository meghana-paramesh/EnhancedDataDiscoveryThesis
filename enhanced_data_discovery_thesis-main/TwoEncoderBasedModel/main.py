# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from TrainModel import train
import shutil
import rasterio
from rasterio.windows import from_bounds, intersection
from skimage.transform import resize
from matplotlib.image import imsave
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    print_hi('PyCharm')
    # train()
    maxar_file_path = "/home/CS/mp0157/dataset/maxar_scenes_desert/desert_187.tif"
    planet_file_path = "/home/CS/mp0157/dataset/planet_scenes_desert/desert_187.tif"

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

        # Define the downsampling factors in each dimension
        downsample_factor_width = 2
        downsample_factor_height = 2

        # Calculate the new dimensions after downsampling
        new_width = reorder_bands_planet.shape[2] // downsample_factor_width
        new_height = reorder_bands_maxar.shape[1] // downsample_factor_height

        # Perform bilinear interpolation for downsampling
        downsampled_planet_image = resize(reorder_bands_planet, (7, new_height, new_width), order=1, mode='reflect', anti_aliasing=False)
        downsampled_maxar_image = resize(reorder_bands_maxar, (7, new_height, new_width), order=1, mode='reflect', anti_aliasing=False)

        rgb_planet_resized = downsampled_planet_image[[5, 3, 1], :, :]
        rgb_maxar_resized = downsampled_planet_image[[4, 2, 1], :, :]

        # Convert the downsampled image from numpy array to uint8 (0-255 range)
        downsampled_planet_image_uint8 = (rgb_planet_resized * 255).astype(np.uint8)
        downsampled_maxar_image_uint8 = (rgb_maxar_resized * 255).astype(np.uint8)

        # View the downsampled image using Matplotlib
        imsave("planet.png", np.transpose(downsampled_planet_image_uint8, (1, 2, 0)))
        imsave("maxar.png", np.transpose(downsampled_maxar_image_uint8, (1, 2, 0)))   # Transpose to (height, width, channels) format
        # plt.axis('off')  # Turn off axis
        # plt.show()
    
    print(downsampled_planet_image.shape)
    print(downsampled_maxar_image.shape)

   #  des_base_dir="/media/hdd2/Meghana/dataset_in_pairs/desert/"
   #  forest_base_dir="/media/hdd2/Meghana/dataset_in_pairs/forest/"
   #  snow_base_dir="/home/CS/mp0157/dataset/dataset_in_pairs/snow/"


   #  maxar_paths_d=[des_base_dir+"pair_10300100BED53E00/10300100BED53E00/", des_base_dir+"pair_104001005B272100/104001005B272100/",
   #               des_base_dir+"pair_10300100C151E500/10300100C151E500/", des_base_dir+"pair_10300100BB781000/10300100BB781000/",
   #               des_base_dir+"pair_10300100BB524A00/10300100BB524A00/", des_base_dir+"pair_10300100BB45C000/10300100BB45C000/",
   #               des_base_dir+"pair_10300100C7A45600/10300100C7A45600/", des_base_dir+"pair_10300100C62D1400/10300100C62D1400/",
   #               des_base_dir + "pair_10300100C1A65D00/10300100C1A65D00/",
   #               des_base_dir + "pair_10300100C0088100/10300100C0088100/",
   #               des_base_dir + "pair_10300100C187DF00/10300100C187DF00/",
   #               des_base_dir + "pair_10300100A7297800/10300100A7297800/",
   #               des_base_dir + "pair_10300110A511CE00/10300110A511CE00_M1BS/",
   #               des_base_dir + "pair_10300100C099A900/10300100C099A900/",
   #               ]
   #  planet_paths_d=[des_base_dir+"pair_10300100BED53E00/desert_10300100BED53E00_psscene_analytic_8b_sr_udm2/PSScene/", des_base_dir+"pair_104001005B272100/desert_104001005B272100_2020_06_09_partial_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir+"pair_10300100C151E500/desert_10300100C151E500_psscene_analytic_8b_sr_udm2/PSScene/", des_base_dir+"pair_10300100BB781000/desert_10300100BB781000_2021_03_18_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir+"pair_10300100BB524A00/desert_10300100BB524A00_2021_03_18_psscene_analytic_8b_sr_udm2/PSScene/", des_base_dir+"pair_10300100BB45C000/desert_10300100BB45C000_2021_03_18_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir+"pair_10300100C7A45600/desert_10300100C7A45600_2021_10_10_psscene_analytic_8b_sr_udm2/PSScene/", des_base_dir+"pair_10300100C62D1400/desert_10300100C62D1400_2021_10_10_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir + "pair_10300100C1A65D00/desert_10300100C1A65D00_2021_06_24_psscene_analytic_8b_sr_udm2/PSScene",
   #                des_base_dir + "pair_10300100C0088100/desert_10300100C0088100_2021_06_19_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir + "pair_10300100C187DF00/desert_10300100C187DF00_2021_06_19_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir + "pair_10300100A7297800/desert_20200620184346_10300100A7297800_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir + "pair_10300110A511CE00/desert_10300110A511CE00_2020_03_12_psscene_analytic_8b_sr_udm2/PSScene/",
   #                des_base_dir + "pair_10300100C099A900/desert_10300100C099A900_2021_06_08_psscene_analytic_8b_sr_udm2/PSScene/",
   #                ]
   #  maxar_paths_f=[forest_base_dir+"pair_10300100C5460200/10300100C5460200/",
   #                 forest_base_dir+"pair_10300100A5258A00/10300100A5258A00/",
   #                 forest_base_dir+"pair_10300100A4AC1F00/10300100A4AC1F00/",
   #                 forest_base_dir+"pair_10300100A972A400/10300100A972A400/",
   #                 forest_base_dir + "pair_10300100A736A000/10300100A736A000/",
   #                 forest_base_dir + "pair_1040010060728B00/1040010060728B00/",
   #                 forest_base_dir + "pair_10300100AB567000/10300100AB567000/",
   #                 forest_base_dir + "pair_104001005ED61B00/104001005ED61B00/",
   #                 forest_base_dir + "pair_104001005D624F00/104001005D624F00/",
   #                 forest_base_dir + "pair_104001005EAE5800/104001005EAE5800/",
   #                 forest_base_dir + "pair_10300100A848DE00/10300100A848DE00/",
   #                 # forest_base_dir + "pair_10300100AA078700/10300100AA078700/",
   #                 forest_base_dir + "pair_104001005C74C400/104001005C74C400/",
   #                 forest_base_dir + "pair_10300100AD7CD000/10300100AD7CD000/",
   #                 forest_base_dir + "pair_104001006010BB00/104001006010BB00/",
   #                 forest_base_dir + "pair_10300100A90E2200/10300100A90E2200/",
   #                 forest_base_dir + "pair_10300100A6495D00/10300100A6495D00/",
   #                 forest_base_dir + "pair_10300100A60C3B00/10300100A60C3B00/",
   #                 forest_base_dir + "pair_10300100AABCC900/10300100AABCC900/",
   #                 forest_base_dir + "pair_104001005CBDD200/104001005CBDD200/",
   #                 ]

   #  planet_paths_f=[forest_base_dir+"pair_10300100C5460200/forest_10300100C5460200_2021_09_12_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir+"pair_10300100A5258A00/forest_10300100A5258A00_2020_05_29_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100A4AC1F00/forest_10300100A4AC1F00_2020_05_29_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100A972A400/forest_10300100A972A400_2020_05_29_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100A736A000/forest_10300100A736A000_2020_05_29_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_1040010060728B00/forest_1040010060728B00_2020_09_13_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100AB567000/forest_10300100AB567000_2020_09_06_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_104001005ED61B00/forest_104001005ED61B00_2020_07_26_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_104001005D624F00/forest_104001005D624F00_2020_07_20_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_104001005EAE5800/forest_104001005EAE5800_2020_07_20_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100A848DE00/forest_10300100A848DE00_2020_07_11_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  # forest_base_dir + "pair_10300100AA078700/forest_10300100AA078700_2020_06_03_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_104001005C74C400/forest_104001005C74C400_2020_05_11_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100AD7CD000/forest_10300100AD7CD000_2020_09_30_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_104001006010BB00/forest_104001006010BB00_2020_09_22_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100A90E2200/forest_10300100A90E2200_2020_06_02_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100A6495D00/forest_10300100A6495D00_2020_05_23_partial_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100A60C3B00/forest_10300100A60C3B00_2020_05_11_partial_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_10300100AABCC900/forest_10300100AABCC900_2020_08_04_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  forest_base_dir + "pair_104001005CBDD200/forest_104001005CBDD200_2020_05_01_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  ]

   #  maxar_paths_s=[
   #      snow_base_dir+"pair_10300100A40CB600/10300100A40CB600_M1BS/",
   #              #    snow_base_dir+"pair_10300100A37F6E00/10300100A37F6E00_M1BS/",
   #                 snow_base_dir+"pair_1040010059962200/1040010059962200_M1BS/",
   #                 snow_base_dir+"pair_104001005CBED700/104001005CBED700_M1BS/",
   #              #    snow_base_dir + "pair_104001005A9BC500/104001005A9BC500_M1BS/",
   #                 snow_base_dir + "pair_10400100669A8F00/10400100669A8F00_M1BS/",
   #                 snow_base_dir + "pair_10300100B218AE00/10300100B218AE00/",
   #                 snow_base_dir + "pair_10300100C67FC000/10300100C67FC000/",
   #              #    TODO: testdata "pair_10300100C1371400/10300100C1371400/",
   #                 snow_base_dir + "pair_10300100B3A54300/10300100B3A54300/",
   #                 snow_base_dir + "pair_1040010064A8B100/1040010064A8B100/",
   #                 snow_base_dir + "pair_10300100B3369300/10300100B3369300/",
   #                 snow_base_dir + "pair_10300100B38E1700/10300100B38E1700/",
   #                 snow_base_dir + "pair_10300100B3466E00/10300100B3466E00/",
   #              #    snow_base_dir + "pair_1040010065714900/1040010065714900/",
   #                 snow_base_dir + "pair_10300100BDAEA100/10300100BDAEA100/",
   #                 snow_base_dir + "pair_10300100B61E7A00/10300100B61E7A00/",
   #                 snow_base_dir + "pair_10300100B551EB00/10300100B551EB00/",
   #                 snow_base_dir + "pair_10300100B333CE00/10300100B333CE00/",
   #                 snow_base_dir + "pair_1040010067594B00/1040010067594B00/",
   #                 ]

   #  planet_paths_s=[
   #      snow_base_dir+"pair_10300100A40CB600/snow_10300100A40CB600_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  # snow_base_dir+"pair_10300100A37F6E00/snow_10300100A37F6E00_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_1040010059962200/snow_1040010059962200_April_2020_1st_13th_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_104001005CBED700/snow_104001005CBED700_2020_04_14_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  # snow_base_dir + "pair_104001005A9BC500/snow_104001005A9BC500_April_2020_2nd_16th_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10400100669A8F00/snow_10400100669A8F00_2021_02_25_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B218AE00/snow_10300100B218AE00_2021_01_24_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100C67FC000/snow_10300100C67FC000_2021_09_22_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  # TODO: testdata: snow_base_dir + "pair_10300100C1371400/snow_10300100C1371400_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B3A54300/snow_10300100B3A54300_2021_02_27_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_1040010064A8B100/snow_1040010064A8B100_2021_01_26_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B3369300/snow_10300100B3369300_2021_01_25_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B38E1700/snow_10300100B38E1700_2021_01_25_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B3466E00/snow_10300100B3466E00_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  # snow_base_dir + "pair_1040010065714900/snow_1040010065714900_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100BDAEA100/snow_10300100BDAEA100_2021_04_14_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B61E7A00/snow_10300100B61E7A00_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B551EB00/snow_10300100B551EB00_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_10300100B333CE00/snow_10300100B333CE00_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  snow_base_dir + "pair_1040010067594B00/snow_1040010067594B00_psscene_analytic_8b_sr_udm2/PSScene/",
   #                  ]
   #  get_all_tiles_to_the_same_projection_and_save(maxar_paths_s, planet_paths_s,
   #                                                "/home/CS/mp0157/dataset/maxar_scenes_snow",
   #                                                "/home/CS/mp0157/dataset/planet_scenes_snow")
   #  # print(check_intersection_and_create_new_files(maxar_path, planet_path, dest_path))
   #  plot_time_diff(maxar_paths_d, planet_paths_d)
   #  desert_diff = plot_time_diff(maxar_paths_d, planet_paths_d)
   #  forest_diff = plot_time_diff(maxar_paths_f, planet_paths_f)
   #  snow_diff = plot_time_diff(maxar_paths_s, planet_paths_s)
   #  plot_date_diff(desert_diff, forest_diff, snow_diff)

    # plot_diff_in_days()
    # train()
    # Display and store tiles from the first image (TIF)
    # tif_path = '/Volumes/Untitled/Meghana/REQUIRED/SEGREGATED/dataset/snow_covered_planet/snow_covered_10300100A40CB600_2020_04_08_original_April4_instead_psscene_analytic_8b_sr_udm2/PSScene/20200404_204101_29_2263_3B_AnalyticMS_SR_8b.tif'
    # output_path_tif = 'composite_tif.png'
    #
    #
    # # Display and store tiles from the second image (NTF)
    # ntf_path = '/Volumes/Untitled/Meghana/REQUIRED/SEGREGATED/dataset/snow_covered_maxar/Meghana/10300100A40CB600_M1BS/WV02_20200408214139_10300100A40CB600_20APR08214139-M1BS-504232271090_01_P004.ntf'
    # output_path_ntf = 'composite_ntf.png'
    # display_tiles(ntf_path, ntf_path, "/Users/meghananp/Documents/Summer2023/code")
    # display_data_distribution()
    # Start recursive traversal

