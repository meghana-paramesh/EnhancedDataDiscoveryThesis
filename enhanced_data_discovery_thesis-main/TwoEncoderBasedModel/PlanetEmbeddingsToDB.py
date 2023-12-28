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
    conn = sqlite3.connect('/home/CS/mp0157/dataset/DB/planet_contrastive_loss_new.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS planet_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB,
            file_name VARCHAR(64),
            vendor_type VARCHAR(64)
        )
    ''')

    batch_size=1

    device = torch.device("cuda:0")
    transform1, transform2 = get_transform()
    num_epochs = 50

    base_dir = "/home/CS/mp0157/dataset/new_tiles_original_inference/"

    dataset_planet = NpyDataset(base_dir+"planet_irrigation_tiles", base_dir+"planet_urban_tiles", base_dir+"planet_snow_cap_mountains_tiles", False, True, 0, 10000, transform1, transform2)

    inference_loader = DataLoader(dataset_planet, batch_size=batch_size, shuffle=True)

    planet_model = SimSiam(8, 2048)  
    # model= nn.DataParallel(model)
    checkpoint = torch.load('planet_contrastive_loss.pth')
    planet_model.to(device)
    
    # Initialize the image encoders and transformers
    embedding_size = 2048
    planet_model.load_state_dict(checkpoint['model_state_dict'])
    planet_model.eval()


    for inputs in inference_loader:
        planet1, planet2, file_name = inputs[0].to(device), inputs[1].to(device), inputs[2]
        p1_planet, p2_planet, z1_planet, z2_planet = planet_model(planet1, planet2)

        data_planet = (z1_planet.clone().detach().cpu().numpy().tobytes(), str(file_name), "PLANET")
        cursor.execute('INSERT INTO planet_table (embedding, file_name, vendor_type) VALUES (?,?,?)', data_planet)

        data_planet = (z2_planet.clone().detach().cpu().numpy().tobytes(), str(file_name), "PLANET")
        cursor.execute('INSERT INTO planet_table (embedding, file_name, vendor_type) VALUES (?,?,?)', data_planet)
    conn.commit()
    conn.close()


add_embeddings_to_the_database()

