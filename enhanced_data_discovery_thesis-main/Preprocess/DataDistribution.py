import matplotlib.pyplot as plt
import glob
import os


def display_data_distribution():
    # Directory containing the images
    base_dir_snow_M = "/Volumes/Untitled/Meghana/REQUIRED/SEGREGATED/dataset/snow_covered_maxar/Meghana/"
    base_dir_snow_P = "/Volumes/Untitled/Meghana/REQUIRED/SEGREGATED/dataset/snow_covered_planet"

    image_directory = [base_dir_snow_M + "10300100A37F6E00_M1BS", base_dir_snow_M + "10300100B333CE00",
                       base_dir_snow_M + "10300100B38E1700", base_dir_snow_M + "10300100B61E7A00",
                       base_dir_snow_M + "10300100C67FC000",
                       base_dir_snow_M + "104001005CBED700_M1BS", base_dir_snow_M + "10400100669A8F00_M1BS",
                       base_dir_snow_M + "10300100A40CB600_M1BS", base_dir_snow_M + "10300100B3369300",
                       base_dir_snow_M + "10300100B3A54300",
                       base_dir_snow_M + "10300100BDAEA100", base_dir_snow_M + "1040010059962200_M1BS",
                       base_dir_snow_M + "1040010064A8B100", base_dir_snow_M + "1040010067594B00",
                       base_dir_snow_M + "10300100B218AE00",
                       base_dir_snow_M + "10300100B3466E00", base_dir_snow_M + "10300100B551EB00",
                       base_dir_snow_M + "10300100C1371400", base_dir_snow_M + "104001005A9BC500_M1BS",
                       base_dir_snow_M + "1040010065714900",
                       base_dir_snow_P + "snow_10300100A40CB600_2020_04_08_psscene_analytic_8b_sr_udm2",
                       base_dir_snow_P + "snow_10300100B218AE00_2021_01_24_psscene_analytic_8b_sr_udm2",
                       base_dir_snow_P + "snow_10300100B333CE00_2021_02_15_psscene_analytic_8b_sr_udm2",
                       base_dir_snow_P + "snow_10300100B3369300_2021_01_25_psscene_analytic_8b_sr_udm2",
                       base_dir_snow_P + "snow_10300100B3466E00_2021_01_19_psscene_analytic_8b_sr_udm2"]

    n_files = []
    t_files = []
    for dir in image_directory:
        # Search for NTF and TIF image files
        ntf_files = glob.glob(dir + "/*.ntf")
        tif_files = glob.glob(dir + "/*.tif")
        n_files.append(ntf_files)
        t_files.append(tif_files)

    print("n_files.__len__(): ", n_files.__len__())
    print("t_files.__len__() :", t_files.__len__())

    for f in n_files:
        print(f)

    # Get the dates for NTF and TIF images
    # ntf_dates = [ntf_file.split("/")[-1].split(".")[0] for ntf_file in ntf_files]
    # tif_dates = [tif_file.split("/")[-1].split(".")[0] for tif_file in tif_files]

    # # Count the frequency of each date
    # ntf_date_counts = {}
    # tif_date_counts = {}

    # for date in ntf_dates:
    #     ntf_date_counts[date] = ntf_date_counts.get(date, 0) + 1

    # for date in tif_dates:
    #     tif_date_counts[date] = tif_date_counts.get(date, 0) + 1

    # # Create the plot
    # plt.bar(ntf_date_counts.keys(), ntf_date_counts.values(), label='NTF')
    # plt.bar(tif_date_counts.keys(), tif_date_counts.values(), label='TIF')

    # # Customize the plot
    # plt.xlabel("Date")
    # plt.ylabel("Frequency")
    # plt.title("Image Frequency Distribution")
    # plt.legend()

    # # Rotate the x-axis labels for better visibility
    # plt.xticks(rotation=45)

    # # Display the plot
    # plt.show()


dates = []
file_types = []


# Recursive function to process NTF and TIF files
def process_directory(directory, depth=0):
    global dates

    # Maximum depth for recursive traversal
    max_depth = 10
    if depth > max_depth:
        return

    try:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if file.endswith(".tif") and not file.startswith("."):
                    # Process the file as needed
                    print("Processing file:", file_path)
                    res_t = file.rsplit(', ', 1)
                    file_name_t = res_t[-1]
                    print(file_name_t[:8])
                    dates.append(file_name_t[:8])
                    file_types.append("p")
                if file.endswith(".ntf"):
                    print("Processing file:", file_path)
                    res_n = file.rsplit(', ', 1)
                    file_name_n = res_n[-1]
                    print(file_name_n[5:13])
                    dates.append(file_name_n[5:13])
                    file_types.append("m")

            elif os.path.isdir(file_path):
                # Recursively process subdirectories
                process_directory(file_path, depth + 1)
    except PermissionError:
        # Handle permission errors for certain directories
        print("Permission denied for directory:", directory)
    return dates, file_types
