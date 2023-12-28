import torch
from SimsiamSameEncoderModel import SimSiam
import sqlite3
from ImageDataset import NpyDataset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
import torch.nn as nn
from Transforms import get_transform

def add_embeddings_to_the_database():
    # Connect to the database or create a new one if it doesn't exist
    conn = sqlite3.connect('/home/CS/mp0157/dataset/DB/maxar_contrastive_loss.db')
    cursor = conn.cursor()
    device = torch.device("cuda:0")

    # Create a table to store embeddings and related information
    # TODO: Create a tabel in this format
    # TODO: idx, maxar_tile, m_type, planet_tile, p_type, maxar_metadata, planet_metadata, maxar_tile_name, planet_tile_name
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maxar_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB,
            file_name VARCHAR(64),
            vendor_type VARCHAR(64)
        )
    ''')

    # cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS planet_table (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         embedding BLOB,
    #         file_name VARCHAR(64),
    #         vendor_type VARCHAR(64)
    #     )
    # ''')

    batch_size=1

    transform1, transform2 = get_transform()
    num_epochs = 50

    base_dir = "/home/CS/mp0157/dataset/new_tiles_original_inference/"

    dataset_maxar = NpyDataset(base_dir+"maxar_irrigation_tiles", base_dir+"maxar_urban_tiles", base_dir+"maxar_snow_cap_mountain_tiles", False, True, 10000, transform1, transform2)

    inference_loader = DataLoader(dataset_maxar, batch_size=batch_size, shuffle=True)

    maxar_model = SimSiam(8, 2048)  
    checkpoint = torch.load('maxar_contrastive_loss.pth')
    # model= nn.DataParallel(model)
    maxar_model.to(device)
    

    # planet_model = SimSiam(7, 2048)  
    # # model= nn.DataParallel(model)
    # planet_model.to(device)
    # planet_model.load_state_dict(torch.load('planet_model_normalized.pth'))

    # Initialize the image encoders and transformers
    embedding_size = 2048
    
    # maxar_transformer = Transformer(input_size=embedding_size, output_size=embedding_size, num_layers=4)
    # planet_transformer = Transformer(input_size=embedding_size, output_size=embedding_size, num_layers=4)

    # maxar_transformer.to(device)
    # maxar_transformer.load_state_dict(torch.load('maxar_transformer.pth'))

    # planet_transformer.to(device)
    # planet_transformer.load_state_dict(torch.load('planet_transformer.pth'))
    maxar_model.load_state_dict(checkpoint['model_state_dict'])
    maxar_model.eval() 
    # planet_model.eval()
    # planet_transformer.eval()
    # maxar_transformer.eval()

    for inputs in inference_loader:
        maxar1, maxar2, file_name = inputs[0].to(device), inputs[1].to(device), inputs[2]
        z1_online, z2_online, z1_target, z2_target = maxar_model(maxar1, maxar2)
        # p1_planet, p2_planet, z1_planet, z2_planet = planet_model(planet1, planet2)

        data_maxar = (z1_online.clone().detach().cpu().numpy().tobytes(), str(file_name), "MAXAR")
        cursor.execute('INSERT INTO maxar_table (embedding, file_name, vendor_type) VALUES (?,?,?)', data_maxar)

        # data_planet = (z1_planet.clone().detach().cpu().numpy().tobytes(), str(file_name), "PLANET")
        # cursor.execute('INSERT INTO planet_table (embedding, file_name, vendor_type) VALUES (?,?,?)', data_planet)

        data_maxar = (z2_online.clone().detach().cpu().numpy().tobytes(), str(file_name), "MAXAR")
        cursor.execute('INSERT INTO maxar_table (embedding, file_name, vendor_type) VALUES (?,?,?)', data_maxar)

        # data_planet = (z2_planet.clone().detach().cpu().numpy().tobytes(), str(file_name), "PLANET")
        # cursor.execute('INSERT INTO planet_table (embedding, file_name, vendor_type) VALUES (?,?,?)', data_planet)
    conn.commit()
    conn.close()


add_embeddings_to_the_database()

