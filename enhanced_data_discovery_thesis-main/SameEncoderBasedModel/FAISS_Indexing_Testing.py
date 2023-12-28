import sqlite3
import faiss
import torch
import numpy as np
from natsort import natsorted
import os
from torchvision import models
from ImageDataset import ImageDataset
from SimsiamSameEncoderModel import SimSiam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from Meta_Data_Extracter import get_planet_meta_data, get_maxar_meta_data
from shapely.geometry import Polygon

def add_index_and_test():
    conn = sqlite3.connect('DB/embeddings_proper_bands_alligned_new.db')
    cursor = conn.cursor()

    # cursor.execute('SELECT embedding FROM embeddings_trail')
    # results = cursor.fetchall()

    cursor.execute('SELECT * FROM embeddings_trail')

    complete_result = cursor.fetchall()

    print(str(complete_result[0][4])[2:-3])
    near_neighbors = open("near_neighbors1_new_tiles.txt", "a")

    # TODO: Convert the blob to NumPy arrays and create PyTorch tensors
    retrieved_embeddings_np = [np.frombuffer(result[1], dtype=np.float32) for result in complete_result]
    retrieved_embeddings_torch = [torch.tensor(embedding) for embedding in retrieved_embeddings_np]

    print(len(complete_result))
    index = faiss.IndexFlatL2(len(complete_result))
    conn.close()

    # TODO: Convert the retrieved embeddings back to NumPy arrays
    retrieved_embeddings_np = np.array([embedding.numpy() for embedding in retrieved_embeddings_torch])

    # TODO: Create an index for the retrieved embeddings
    index = faiss.IndexFlatL2(retrieved_embeddings_np.shape[1])
    index.add(retrieved_embeddings_np)
    i=450
   # min_len=550
    min_len=452
    batch_files = 1
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

    accuracy=0

    batch_files_maxar_desert = file_list_maxar_desert[i:i+batch_files]
    batch_files_planet_desert = file_list_planet_desert[i:i+batch_files]
    batch_files_maxar_forest = file_list_maxar_forest[i:i+batch_files]
    batch_files_planet_forest = file_list_planet_forest[i:i+batch_files]
    batch_files_maxar_snow = file_list_maxar_snow[i:i+batch_files]
    batch_files_planet_snow = file_list_planet_snow[i:i+batch_files]


    dataset = ImageDataset(False, True, maxar_paths_d, batch_files_maxar_desert, planet_paths_d, batch_files_planet_desert, maxar_paths_s, batch_files_maxar_snow, planet_paths_s, batch_files_planet_snow, maxar_paths_f, batch_files_maxar_forest, planet_paths_f, batch_files_planet_snow)
    print("dataset length: ",len(dataset))
    batch_size=1
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    embeddings = np.zeros((len(dataset),  2048))
    checkpoint_path = "model_trail_same_order_bands.pt"
    checkpoint = torch.load(checkpoint_path)
    model = SimSiam(2048)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    j=0
    for batch in data_loader:
        maxar_tile, m_type, planet_tile, p_type, maxar_file, planet_file, maxar_tile_name, planet_tile_name = batch
        # maxar_tile = maxar_tile.to(device)
        # planet_tile = planet_tile.to(device)
        # Forward pass
        p1, p2, z1, z2 = model(maxar_tile, planet_tile)


        # TODO: Perform similarity search using Faiss
        query_embeddings_p1 = p1.detach().numpy()
        query_embeddings_p2 = p2.detach().numpy()
        k = 10  

        distances1, indices1 = index.search(query_embeddings_p1, k)
        distances2, indices2 = index.search(query_embeddings_p2, k)


        classes_maxar = []

        for idx_m in indices1[0]:
            corresponding_tile_type = complete_result[idx_m][3]
        
        # maxar_class = np.argmax(np.bincount(np.array(classes_maxar)))
        # print("maxar class: ",maxar_class)
        # print("actual maxar class: ",m_type.item())

        #TODO: First get the nearest planet image. Check if the bounding boxes intersect

            if corresponding_tile_type == "PLANET":
                if m_type.item() == int(complete_result[idx_m][2]):
                    corresponding_planet_file = str(complete_result[idx_m][4][2:-3])
                    print("corresponding_planet_file: ",corresponding_planet_file.replace(".tif", ".xml"))
                    print("maxar_file: ",maxar_file[0].replace(".tif", ".xml"))
                    if check_intersection(maxar_file[0].replace(".tif", ".xml"), maxar_file[0], corresponding_planet_file.replace(".tif", ".xml"), corresponding_planet_file):
                        accuracy+=1
                        break
                    # visualize_k_near_neighbors(indices1[0], complete_result, "MAXAR", maxar_tile_name, j)

            # Extract RGB bands (assuming bands 1, 2, 3 are RGB)
            # rgb_P_bands = p_tile[planet_bands, :, :]
            # rgb_M_bands = m_tile[maxar_bands, :, :]

            # # Display the tile using Matplotlib
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(rgb_P_bands.transpose(1, 2, 0))
            # # plt.title("Planet")
            # plt.axis('off')
            # # fig.add_subplot(rows, columns, 2)
            # axs[1].imshow(rgb_M_bands.transpose(1, 2, 0))
            # # plt.title("Maxar")
            # plt.axis('off')
            # plt.show()
        

        classes_planet = []
        
        for idx_p in indices2[0]:
            # classes_planet.append(int(complete_result[idx_p][2]))
            corresponding_tile_type = complete_result[idx_p][3]
            if corresponding_tile_type == "MAXAR":
                if p_type.item() == int(complete_result[idx_p][2]):
                    corresponding_maxar_file = str(complete_result[idx_p][4][2:-3])
                    print("corresponding_maxar_file: ",corresponding_maxar_file.replace(".tif", ".xml"))
                    print("planet_file: ",planet_file[0].replace(".tif", ".xml"))
                    if check_intersection(corresponding_maxar_file.replace(".tif", ".xml"), corresponding_maxar_file, planet_file[0].replace(".tif", ".xml"), planet_file[0]):
                        print("found planet nearly")
                        accuracy+=1
                        break
        
        # planet_class = np.argmax(np.bincount(np.array(classes_planet)))
        # print("planet class: ",planet_class)
        # print("actual planet class: ",p_type.item())

        
        # if p_type.item() == planet_class:
        #     accuracy+=1
        #     display_neighbors(indices2[0], complete_result, planet_tile_name)
        #     # visualize_k_near_neighbors(indices2[0], complete_result, "PLANET", planet_tile_name, j)
        #     j+=1
        # for idx in indices[0]:
        #         print(dataset[idx])
        #     nearest_distances = distances.squeeze()

        #     print("nearest distance: ",nearest_distances)
    
    print("number of correct counts: ",accuracy)

    #TODO: from the dataset accuracy is calculated as one for planet and one for maxar
    print("accuracy = ",accuracy/(len(dataset)*2))
    near_neighbors.close()

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

def display_neighbors(index, complete_result, original_tile):
    near_neighbor = open("near_neighbors1_new_tiles.txt.txt", "a")
    print(original_tile)
    near_neighbor.write("\n")
    near_neighbor.write("original_tile: "+original_tile[0])
    for ind in index:
        near_neighbor.write("\n")
        file_name = complete_result[ind][4]
        print(file_name)
        near_neighbor.write("file_name: "+str(file_name))
        near_neighbor.write("\n")
        tile_name = complete_result[ind][5][2:-3]
        near_neighbor.write("tile_name: "+tile_name)
        near_neighbor.write("\n")
    near_neighbor.write("=============================================")

def visualize_k_near_neighbors(index, complete_result, original_type, original_image, k):
    plt.figure()
    fig, axes = plt.subplots(1, 20, figsize=(20, 4))
    data = np.load(original_image[0])
    if original_type=="PLANET":
        selected_bands = [2, 4, 6]
        data = data / 255.0
        rgb_bands_data = data[selected_bands, :, :]
        plt.imshow(np.transpose(rgb_bands_data, (1, 2, 0)))
        axes[0].axis('off')

    if original_type=="MAXAR":
        selected_bands = [2, 3, 5]
        data = data / 255.0
        rgb_bands_data = data[selected_bands, :, :]
        plt.imshow(np.transpose(rgb_bands_data, (1, 2, 0)))
        axes[1].axis('off')

    i=2
    for idx_m in index:
        file = complete_result[idx_m][5]
        file = file[2:-3]
        print("complete_result[idx_m][5]: ",file)
        data = np.load(file)
        print("complete_result[idx_m][5]: ",complete_result[idx_m][5])
        if complete_result[idx_m][3]=="PLANET":
            selected_bands = [2, 4, 6]
            data = data / 255.0
            rgb_bands_data = data[selected_bands, :, :]
            plt.imshow(np.transpose(rgb_bands_data, (1, 2, 0)))
            axes[i].axis('off')
        i+=1

        if complete_result[idx_m][3]=="MAXAR":
            selected_bands = [2, 3, 5]
            data = data / 255.0
            rgb_bands_data = data[selected_bands, :, :]
            plt.imshow(np.transpose(rgb_bands_data, (1, 2, 0)))
            axes[i].axis('off')
        i+=1
    plt.savefig('near_neighbor/'+str(k)+'.png')


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


    

add_index_and_test()