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
from Transforms import get_transform
from PlotConfusionMatrix import plot_mertics
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from PlotConfusionMatrix import plot_mertics
from Transformer import GeospatialTransformer
from SimpleTransformer import Transformer

def add_index_and_test():
    conn = sqlite3.connect('/home/CS/mp0157/dataset/DB/maxar_contrastive_loss.db')
    cursor = conn.cursor()

    # cursor.execute('SELECT embedding FROM embeddings_trail')
    # results = cursor.fetchall()

    cursor.execute('SELECT * FROM maxar_table')
    accurately_retrieved = 0

    complete_result_maxar = cursor.fetchall()
    print(len(complete_result_maxar))

    retrieved_embeddings_np_maxar = [np.frombuffer(result[1], dtype=np.float32) for result in complete_result_maxar]
    retrieved_embeddings_torch_maxar = [torch.tensor(embedding) for embedding in retrieved_embeddings_np_maxar]


    # TODO: Create an index for the retrieved embeddings
    retrieved_embeddings_np_maxar = np.array([embedding.numpy() for embedding in retrieved_embeddings_torch_maxar])
    conn.close()
    print("retrieved_embeddings_np_maxar.shap: ",retrieved_embeddings_np_maxar.shape)

    transform1, transform2 = get_transform()
    base_dir = "/home/CS/mp0157/dataset/new_tiles_original_inference/"
    batch_size=1

    dataset_planet = NpyDataset(base_dir+"planet_irrigation_tiles", base_dir+"planet_urban_tiles", base_dir+"planet_snow_cap_mountains_tiles", False, True, 10500, 10550, transform1, transform2)
    test_loader = DataLoader(dataset_planet, batch_size=batch_size, shuffle=True)
    print(len(test_loader))

    planet_model = SimSiam(8, 2048)  
    checkpoint = torch.load('important/planet_contrastive_loss.pth')
    planet_model.load_state_dict(checkpoint['model_state_dict'])
    planet_model.eval() 

    # TODO: Create an index for the retrieved embeddings
    index_maxar = faiss.IndexFlatL2(retrieved_embeddings_np_maxar.shape[1])
    index_maxar.add(retrieved_embeddings_np_maxar)
    accuracy=0.0
    total_ssim=0.0

    embedding_size=256
    threshold = 0.8
    
    k = 5 
    preds = []
    actuals = []
    embedding_size = 256
    planet_transformer = Transformer(input_size=embedding_size)
    checkpoint_planet_transformer = torch.load('important/planet_transformer.pth')
    planet_transformer.load_state_dict(checkpoint_planet_transformer['model_state_dict'])
    planet_transformer.eval()

    for inputs in test_loader:
        x1, x2, filename = inputs[0], inputs[1], inputs[2]
        z1_online, z2_online, z1_target, z2_target  = planet_model(x1, x2)
        print(z1_online.shape)

        transformer_online_maxar1 = planet_transformer(z1_online)
        transformer_online_maxar2 = planet_transformer(z2_online)

        query_embeddings_z1_planet = transformer_online_maxar1.detach().cpu().numpy()
        query_embeddings_z2_planet = transformer_online_maxar2.detach().cpu().numpy()

        print(query_embeddings_z1_planet.shape)

        distances1, indices1 = index_maxar.search(query_embeddings_z1_planet, k)
        distances2, indices2 = index_maxar.search(query_embeddings_z2_planet, k)
        i_m = 0
        corr_class1 = []
        for idx_m in indices1[0]:
            corresponding_tile_type = complete_result_maxar[idx_m][3]
                
            corresponding_tile_name = complete_result_maxar[idx_m][2][2:-3]
            cor_type, cor_filename = get_file_and_type(corresponding_tile_name)
            cur_tile_name = str(filename)[2:-3]
            cur_type, cur_filename = get_file_and_type(cur_tile_name)
            print("\n")
            print(cur_type)
            print(cor_type)              
            corr_class1.append(cor_type)


        if corr_class1:
            planet_class = max(corr_class1,key=corr_class1.count)
            if planet_class == cur_type:
                print(cur_type)
                print(planet_class)
                accuracy+=1
                if "irrigation" in cor_type:
                    dir_req_cor = "maxar_irrigation_tiles/"
                
                if "snow" in cor_type:
                    dir_req_cor = "maxar_snow_cap_mountain_tiles/"

                if "urban" in cor_type:
                    dir_req_cor = "maxar_urban_tiles/"

                if "irrigation" in cur_type:
                    dir_req_cur = "planet_irrigation_tiles/"
                
                if "snow" in cur_type:
                    dir_req_cur = "planet_snow_cap_mountains_tiles/"

                if "urban" in cur_type:
                    dir_req_cur = "planet_urban_tiles/"

                file1 = "/home/CS/mp0157/dataset/new_tiles_original_inference/"+dir_req_cur+cur_tile_name
                file2 = "/home/CS/mp0157/dataset/new_tiles_original_inference/"+dir_req_cor+corresponding_tile_name
                ssim_score = ssim(normalize_image(np.load(file1)), normalize_image(np.load(file2)), data_range=np.load(file1) - np.load(file2), multichannel=True)
                print("ssim_score: ",ssim_score)
                if ssim_score >= threshold:
                    total_ssim += 1
            
            preds.append(planet_class)
            actuals.append(cur_type)        

        corr_class2 = []
        i_p = 0
        for idx_p in indices2[0]:   
            corresponding_tile_name = complete_result_maxar[idx_p][2][2:-3]
            cor_type, cor_filename = get_file_and_type(corresponding_tile_name)
            cur_tile_name = str(filename)[2:-3]
            cur_type, cur_filename = get_file_and_type(cur_tile_name)
            print("\n")
                            
            corr_class2.append(cor_type)

        if corr_class2:
            planet_class = max(corr_class2,key=corr_class2.count)
            if planet_class == cur_type:
                print(cur_type)
                print(planet_class)
                accuracy+=1
                
                if "irrigation" in cor_type:
                    dir_req_cor = "maxar_irrigation_tiles/"
                
                if "snow" in cor_type:
                    dir_req_cor = "maxar_snow_cap_mountain_tiles/"

                if "urban" in cor_type:
                    dir_req_cor = "maxar_urban_tiles/"

                if "irrigation" in cur_type:
                    dir_req_cur = "planet_irrigation_tiles/"
                
                if "snow" in cur_type:
                    dir_req_cur = "planet_snow_cap_mountains_tiles/"

                if "urban" in cur_type:
                    dir_req_cur = "planet_urban_tiles/"

                file1 = "/home/CS/mp0157/dataset/new_tiles_original_inference/"+dir_req_cur+cur_tile_name
                file2 = "/home/CS/mp0157/dataset/new_tiles_original_inference/"+dir_req_cor+corresponding_tile_name
                ssim_score = ssim(normalize_image(np.load(file1)), normalize_image(np.load(file2)), data_range=np.load(file1) - np.load(file2), multichannel=True)
                print("ssim_score: ",ssim_score)
                if ssim_score >= threshold:
                    total_ssim += 1

            preds.append(planet_class)
            actuals.append(cur_type)  


    #TODO: from the dataset accuracy is calculated as one for planet and one for planet
    print("accuracy = ",accuracy/(len(dataset_planet)))
    print("ssim = ",total_ssim/(len(dataset_planet)))
    print("preds length: ",len(preds))
    print("actuals length: ",len(actuals))
    plot_mertics(actuals, preds, "Two Encoder Based (Planet Only)")
    precision = precision_score(actuals, preds, average="macro")
    recall = recall_score(actuals, preds, average="macro")
    f1 = f1_score(actuals, preds, average="macro")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def get_file_and_type(corresponding_tile_name):
    corresponding_tile_split = corresponding_tile_name.split('_')
    corresponding_type = corresponding_tile_split[0]
    file_name = corresponding_tile_split[0]+'_'+corresponding_tile_split[1]
    return corresponding_type, file_name

def min_max_scale_band(band):
    min_value = np.min(band)
    max_value = np.max(band)
    
    # Check if the range is zero
    if max_value == min_value:
        # Handle zero range (you can choose an appropriate epsilon value)
        epsilon = 1e-9
        normalized_band = (band - min_value) / (max_value - min_value + epsilon)
    else:
        normalized_band = (band - min_value) / (max_value - min_value)
    
    return normalized_band

def normalize_image(image):
    normalized_image = np.zeros_like(image, dtype=np.float32)
    for band_idx in range(image.shape[0]):
        normalized_image[band_idx:, :, ] = min_max_scale_band(image[band_idx:, :, ])
    return normalized_image

    

add_index_and_test()