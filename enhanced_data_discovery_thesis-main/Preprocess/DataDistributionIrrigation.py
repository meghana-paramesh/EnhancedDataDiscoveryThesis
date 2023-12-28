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
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import matplotlib.dates as mdates 



def contains_mostly_zero(tile):
    threshold = 0.8
    return np.count_nonzero(tile == 0) / tile.size >= threshold


def data_distribution():

    maxar_paths_u = "/home/CS/mp0157/dataset/IRR_NEW/MAXAR_IRR/"
    planet_paths_u = "/home/CS/mp0157/dataset/IRR_NEW/PLANET_IRR/"

    file_list_maxar_irrigation = natsorted(os.listdir(maxar_paths_u))
    file_list_planet_irrigation = natsorted(os.listdir(planet_paths_u))

    list_maxar = file_list_maxar_irrigation[0: ]
    dates_maxar = []
    dates_planet = []

    for m_file in list_maxar:
        if m_file.endswith(".xml"):
            meta_maxar = get_maxar_meta_data(maxar_paths_u+m_file, "dummy")

            dates_maxar.append(meta_maxar.date_time.split("T")[0])
            meta_planet = get_planet_meta_data(planet_paths_u+m_file, "dummy")
            print(meta_planet.date_time.split("T")[0])
            if "2020-04" in meta_planet.date_time:
                print(meta_planet.date_time.split("T")[0])
                dates_planet.append(meta_planet.date_time.split("T")[0].replace("2020", "2022"))
            else:
                dates_planet.append(meta_planet.date_time.split("T")[0])
 

    date_counts1 = Counter(dates_maxar)
    date_counts2 = Counter(dates_planet)

    # Extract dates and their corresponding counts
    unique_dates1 = list(date_counts1.keys())
    data_points1 = list(date_counts1.values())

    unique_dates2 = list(date_counts2.keys())
    data_points2 = list(date_counts2.values())

    # Convert the date strings to datetime objects
    date_objects1 = pd.to_datetime(unique_dates1)
    date_objects2 = pd.to_datetime(unique_dates2)

    # Create a bar graph for the first set of dates (blue bars)
    plt.figure(figsize=(10, 6))
    plt.bar(date_objects1, data_points1, width=5, align='center', color='blue', label='Maxar Data')

    # Create a bar graph for the second set of dates (orange bars)
    plt.bar(date_objects2, data_points2, width=5, align='edge', color='orange', label='Planet Data')

    # Format the x-axis as dates using DateFormatter
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Number of Multispectral 8 band images")
    plt.title("Date vs. Number of Multispectral 8 band images (Irrigation Area)")

    # Show a legend to differentiate between the sets
    plt.legend()

    # Show the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("data_distribution_irrigation.jpg")

data_distribution()
