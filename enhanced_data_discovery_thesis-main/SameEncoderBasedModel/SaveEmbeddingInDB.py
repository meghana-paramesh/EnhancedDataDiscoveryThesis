import sqlite3
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import torch.optim as optim
from torchvision import models
import torch

from ImageDataset import ImageDataset
from SimsiamSameEncoderModel import SimSiam
import torch.nn as nn
import faiss
import numpy as np
import os
from natsort import natsorted
from Plot import plot_training_validation_loss

# embeddings.db contains both embeddings together
# embeddings_each contains each embeddings, table embeddings_trail
# embeddings_all contains all embeddings, table embeddings_trail, tile_all
def add_embeddings_to_the_database():
    # Connect to the database or create a new one if it doesn't exist
    conn = sqlite3.connect('DB/embeddings_proper_bands_alligned_new.db')
    cursor = conn.cursor()

    # Create a table to store embeddings and related information
    # TODO: Create a tabel in this format
    # TODO: idx, maxar_tile, m_type, planet_tile, p_type, maxar_metadata, planet_metadata, maxar_tile_name, planet_tile_name
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings_trail (
            id INTEGER PRIMARY KEY,
            embedding BLOB,
            class_type VARCHAR(64),
            tile_type VARCHAR(64),
            file_name VARCHAR(128),
            tile_name VARCHAR(64)
        )
    ''')
    test_start=400
    min_len=520
    # min_len=420
    batch_files = 1
    maxar_paths_d = "/home/CS/mp0157/dataset/maxar_scenes_desert/"
    planet_paths_d = "/home/CS/mp0157/dataset/planet_scenes_desert/"
    maxar_paths_f = "/home/CS/mp0157/dataset/maxar_scenes_forest/"
    planet_paths_f = "/home/CS/mp0157/dataset/planet_scenes_forest/"
    maxar_paths_s = "/home/CS/mp0157/dataset/maxar_scenes_snow/maxar_snow_required/"
    planet_paths_s = "/home/CS/mp0157/dataset/planet_scenes_snow/planet_snow_required/"

        # Get the list of files in the folder and sort them
    file_list_maxar_desert = natsorted(os.listdir(maxar_paths_d))
    file_list_planet_desert = natsorted(os.listdir(planet_paths_d))
    file_list_maxar_forest = natsorted(os.listdir(maxar_paths_f))
    file_list_planet_forest = natsorted(os.listdir(planet_paths_f))
    file_list_maxar_snow = natsorted(os.listdir(maxar_paths_s))
    file_list_planet_snow = natsorted(os.listdir(planet_paths_s))

    idx=0

    for i in range(test_start, min_len, batch_files):
        # Get the batch_files of files
        batch_files_maxar_desert = file_list_maxar_desert[i:i+batch_files]
        batch_files_planet_desert = file_list_planet_desert[i:i+batch_files]
        batch_files_maxar_forest = file_list_maxar_forest[i:i+batch_files]
        batch_files_planet_forest = file_list_planet_forest[i:i+batch_files]
        batch_files_maxar_snow = file_list_maxar_snow[i:i+batch_files]
        batch_files_planet_snow = file_list_planet_snow[i:i+batch_files]

        checkpoint_path = "model_trail_same_order_bands.pt"
        checkpoint = torch.load(checkpoint_path)
        model = SimSiam(2048)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        batch_size=1

        dataset = ImageDataset(False, True, maxar_paths_d, batch_files_maxar_desert, planet_paths_d, batch_files_planet_desert, maxar_paths_s, batch_files_maxar_snow, planet_paths_s, batch_files_planet_snow, maxar_paths_f, batch_files_maxar_forest, planet_paths_f, batch_files_planet_snow)
        
        if(len(dataset)==0):
            continue
        print("dataset length: ",len(dataset))
        data_loader = DataLoader(dataset, batch_size, shuffle=False)
        embeddings = np.zeros((len(dataset),  2048))

        i=0
        for batch in data_loader:
            maxar_tile, m_type, planet_tile, p_type, maxar_file, planet_file, maxar_tile_name, planet_tile_name = batch
            # maxar_tile = maxar_tile.to(device)
            # planet_tile = planet_tile.to(device)
            # Forward pass
            p1, p2, z1, z2 = model(maxar_tile, planet_tile)
            print(planet_tile_name)
            """            
                id INTEGER PRIMARY KEY,
                embedding BLOB,
                class_type VARCHAR(64),
                tile_type VARCHAR(64),
                file_name VARCHAR(128),
                tile_name VARCHAR(64),
            """    

            data_maxar = (idx, p1.detach().numpy().tobytes(), m_type.item(), "MAXAR", str(maxar_file), str(maxar_tile_name))
            cursor.execute('INSERT INTO embeddings_trail VALUES (?,?,?,?,?,?)', data_maxar)
            idx += 1

            data_planet = (idx, p2.detach().numpy().tobytes(), p_type.item(), "PLANET", str(planet_file), str(planet_tile_name))
            cursor.execute('INSERT INTO embeddings_trail VALUES (?,?,?,?,?,?)', data_planet)
            idx += 1



            # TODO: save the embeddings to a database

            # embeddings[i] = np.mean(p1.detach().squeeze().cpu().numpy(), axis=0)
            # i=i+1
            # index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
            # index.add(embeddings)
    conn.commit()
    conn.close()

def test_select():
    # Connect to the SQLite database
    conn = sqlite3.connect('DB/embeddings_proper_band.db')
    cursor = conn.cursor()

    # Retrieve data from the table
    cursor.execute('SELECT * FROM embeddings_trail1')
    results = cursor.fetchall()

    # Process the results
    for row in results:
        print(row)

    # Close the connection
    conn.close()

add_embeddings_to_the_database()
# test_select()

