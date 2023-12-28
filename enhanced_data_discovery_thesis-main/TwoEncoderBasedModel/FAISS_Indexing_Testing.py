import sqlite3
import faiss
import torch
import numpy as np
from natsort import natsorted
import os
from torchvision import models
from ImageDataset import NpyDataset
from SimsiamSameEncoderModel import SimSiam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from Meta_Data_Extracter import get_planet_meta_data, get_maxar_meta_data
from shapely.geometry import Polygon
from torch.utils.data import DataLoader, random_split
from ReadCorrespondingData import CorrespondingNpyDataset
from Transformer import Transformer
from Transforms import get_transform
from PlotConfusionMatrix import plot_mertics
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def add_index_and_test():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    conn = sqlite3.connect('/home/CS/mp0157/dataset/DB/indivisual_norm.db')
    cursor = conn.cursor()

    # cursor.execute('SELECT embedding FROM embeddings_trail')
    # results = cursor.fetchall()

    cursor.execute('SELECT * FROM maxar_table')
    accurately_retrieved = 0

    complete_result_maxar = cursor.fetchall()
    print(len(complete_result_maxar))

    cursor.execute('SELECT * FROM planet_table')

    complete_result_planet = cursor.fetchall()
    print(len(complete_result_planet))


    base_dir = '/home/CS/mp0157/dataset/'
    transform1, transform2 = get_transform()


    dataset = CorrespondingNpyDataset('/home/CS/mp0157/dataset/maxar_tiles_test/', '/home/CS/mp0157/dataset/planet_tiles_test/', False, True, transform1, transform2)
    # train_size = int(0.55 * len(dataset))  
    # # test_size = int(0.1 * len(dataset))  
    # inference_size = int(0.4 * len(dataset))
    # test_size = len(dataset) - train_size - inference_size  
    # batch_size=1
    # test_dataset = torch.utils.data.Subset(dataset, range(inference_size, inference_size + test_size))
    # test_dataset = torch.utils.data.Subset(dataset, range(test_size))
    batch_size=1
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    retrieved_embeddings_np_maxar = [np.frombuffer(result[1], dtype=np.float32) for result in complete_result_maxar]
    retrieved_embeddings_torch_maxar = [torch.tensor(embedding) for embedding in retrieved_embeddings_np_maxar]

    retrieved_embeddings_np_planet = [np.frombuffer(result[1], dtype=np.float32) for result in complete_result_planet]
    retrieved_embeddings_torch_planet = [torch.tensor(embedding) for embedding in retrieved_embeddings_np_planet]


    # TODO: Create an index for the retrieved embeddings
    retrieved_embeddings_np_maxar = np.array([embedding.numpy() for embedding in retrieved_embeddings_torch_maxar])
    retrieved_embeddings_np_planet = np.array([embedding.numpy() for embedding in retrieved_embeddings_torch_planet])
    conn.close()

    maxar_model = SimSiam(7, 2048)  
    maxar_model.load_state_dict(torch.load('maxar_model_normalized.pth'))
    maxar_model.to(device)
    maxar_model.eval() 

    planet_model = SimSiam(7, 2048)  
    planet_model.load_state_dict(torch.load('planet_model_normalized.pth'))
    planet_model.to(device)
    planet_model.eval() 

    # TODO: Create an index for the retrieved embeddings
    index_maxar = faiss.IndexFlatL2(retrieved_embeddings_np_maxar.shape[1])
    index_maxar.add(retrieved_embeddings_np_maxar)
    accuracy=0.0
    total_ssim=0.0

    index_planet = faiss.IndexFlatL2(retrieved_embeddings_np_planet.shape[1])
    index_planet.add(retrieved_embeddings_np_planet)
    embedding_size=2048
    
    maxar_transformer = Transformer(input_size=embedding_size, output_size=embedding_size, num_layers=4)
    planet_transformer = Transformer(input_size=embedding_size, output_size=embedding_size, num_layers=4)

    maxar_transformer.to(device)
    maxar_transformer.load_state_dict(torch.load('maxar_transformer_normalized.pth'))

    planet_transformer.to(device)
    planet_transformer.load_state_dict(torch.load('planet_transformer_normalized.pth'))
    k = 5 
    maxar_transformer.eval()
    planet_transformer.eval()
    pred_class = []
    actual_class = []

    for inputs in test_loader:
        maxar1, maxar2, planet1, planet2, file_name = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device), inputs[4]
        p1_maxar, p2_maxar, z1_maxar, z2_maxar = maxar_model(maxar1, maxar2)
        p1_planet, p2_planet, z1_planet, z2_planet = planet_model(planet1, planet2)

        maxar_transformed_1 = maxar_transformer(z1_maxar)
        maxar_transformed_2 = maxar_transformer(z2_maxar)

        query_embeddings_z1_maxar = maxar_transformed_1.detach().cpu().numpy()
        query_embeddings_z2_maxar = maxar_transformed_2.detach().cpu().numpy()

        distances1_m, indices1_m = index_planet.search(query_embeddings_z1_maxar, k)
        distances2_m, indices2_m = index_planet.search(query_embeddings_z2_maxar, k)

        accuracy_maxar, actual, pred, ssim = get_accuracy(indices1_m, complete_result_planet, file_name, query_embeddings_z1_maxar)
        actual_class.append(actual)
        pred_class.append(pred)
        accuracy+=accuracy_maxar
        total_ssim+=ssim
        accuracy_maxar, actual, pred, ssim = get_accuracy(indices2_m, complete_result_planet, file_name, query_embeddings_z2_maxar)
        actual_class.append(actual)
        pred_class.append(pred)
        accuracy+=accuracy_maxar
        total_ssim+=ssim
        

        planet_transformed_1 = planet_transformer(z2_maxar)
        planet_transformed_2 = planet_transformer(z2_planet)

        query_embeddings_z1_planet = planet_transformed_1.detach().cpu().numpy()
        query_embeddings_z2_planet = planet_transformed_2.detach().cpu().numpy()

        distances1_p, indices1_p = index_maxar.search(query_embeddings_z1_planet, k)
        distances2_p, indices2_p = index_maxar.search(query_embeddings_z2_planet, k)
        accuracy_planet, actual, pred, ssim = get_accuracy(indices1_p, complete_result_maxar, file_name, query_embeddings_z1_planet)
        actual_class.append(actual)
        pred_class.append(pred)
        accuracy+=accuracy_planet
        total_ssim+=ssim
        accuracy_planet, actual, pred, ssim = get_accuracy(indices2_p, complete_result_maxar, file_name, query_embeddings_z2_planet)
        actual_class.append(actual)
        pred_class.append(pred)
        accuracy+=accuracy_planet
        total_ssim+=ssim
                        

    plot_mertics(actual_class, pred_class)
    #TODO: from the dataset accuracy is calculated as one for planet and one for maxar
    print("accuracy value: ",accuracy)
    print("accuracy = ",(accuracy)/(len(dataset)*4))
    print("total ssim accuracy = ",total_ssim)
    print("ssim accuracy = ",total_ssim/(len(dataset)*4))

def get_accuracy(indices, complete_result, filename, query_embedding):
    threshold = 0.8
    accuracy=0.0
    accurately_retrieved=0.0
    # for idx_m in indices[0]:
    
    # # maxar_class = np.argmax(np.bincount(np.array(classes_maxar)))
    # # print("maxar class: ",maxar_class)
    # # print("actual maxar class: ",m_type.item())

    # #TODO: First get the nearest planet image. Check if the bounding boxes intersect

    #     corresponding_tile_name = complete_result[idx_m][2][2:-3]
    #     cor_type, cor_filename = get_file_and_type(corresponding_tile_name)
    #     cur_tile_name = str(filename)[2:-3]
    #     cur_type, cur_filename = get_file_and_type(cur_tile_name)
    #     print("\n")

    #     reshaped_array = np.frombuffer(complete_result[idx_m][1], dtype=np.float32).reshape(1, -1)
    #     if complete_result[idx_m][3]=="PLANET":
    #         file1 = "/home/CS/mp0157/dataset/planet_tiles_inference/"+corresponding_tile_name
    #         file2 = "/home/CS/mp0157/dataset/maxar_tiles_test/"+cur_tile_name
        
    #     if complete_result[idx_m][3]=="MAXAR":
    #         file1 = "/home/CS/mp0157/dataset/maxar_tiles_infernce/"+corresponding_tile_name
    #         file2 = "/home/CS/mp0157/dataset/planet_tiles_test/"+cur_tile_name

    
    max_class = []

    for idx_m in indices[0]:
        corresponding_tile_name = complete_result[idx_m][2][2:-3]
        cor_type, cor_filename = get_file_and_type(corresponding_tile_name)
        cur_tile_name = str(filename)[2:-3]
        cur_type, cur_filename = get_file_and_type(cur_tile_name)
        print("\n")

        reshaped_array = np.frombuffer(complete_result[idx_m][1], dtype=np.float32).reshape(1, -1)
        if complete_result[idx_m][3]=="PLANET":
            file1 = "/home/CS/mp0157/dataset/planet_tiles_inference/"+corresponding_tile_name
            file2 = "/home/CS/mp0157/dataset/maxar_tiles_test/"+cur_tile_name
        
        if complete_result[idx_m][3]=="MAXAR":
            file1 = "/home/CS/mp0157/dataset/maxar_tiles_infernce/"+corresponding_tile_name
            file2 = "/home/CS/mp0157/dataset/planet_tiles_test/"+cur_tile_name
        max_class.append(cor_type)
    cor_req = max(max_class,key=max_class.count)

    if cor_req == cur_type:
        print(cur_type)
        print(cor_req)
        accuracy+=1
    ssim_score = ssim(normalize_image(np.load(file1)), normalize_image(np.load(file2)), data_range=np.load(file1) - np.load(file2), multichannel=True)
    print("ssim_score: ",ssim_score)
    if ssim_score >= threshold:
        accurately_retrieved += 1

    return accuracy, cur_type, cor_req, accurately_retrieved

def normalize_image(image):
    min_value = np.min(image)
    max_value = np.max(image)
    normalized_image = (image - min_value) / (max_value - min_value)
    return normalized_image

def get_file_and_type(corresponding_tile_name):
    corresponding_tile_split = corresponding_tile_name.split('_')
    corresponding_type = corresponding_tile_split[0]
    file_name = corresponding_tile_split[0]+'_'+corresponding_tile_split[1]
    return corresponding_type, file_name


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