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



def train():
   # TODO: test the model
   # Initialize the SimSiam model
   torch.cuda.empty_cache()
   print("Is cuda available: ",torch.cuda.is_available())
   device = torch.device("cuda")

   checkpoint_path = 'model_trail_same_order_bands.pt'
   model = SimSiam(2048)
   model.to(device)
   # Loss function and optimizer
   criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
   optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-6, momentum=0.9)
    

   batch_size = 16
   num_epochs = 100
   best_loss = float('inf')

   training_loss = []
   validation_loss = []
   for epoch in range(num_epochs):
      total_loss = 0.0
      val_total_loss = 0.0
      train_dataset_size = 0.0
      validation_dataset_size = 0.0
    
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

      # Number of files to load in each batch
      batch_files = 20

      min_len = 400

      validation_files_maxar_desert = file_list_maxar_desert[401: 420]
      validation_files_planet_desert = file_list_planet_desert[401: 420]
      validation_files_maxar_forest = file_list_maxar_forest[401: 420]
      validation_files_planet_forest = file_list_planet_forest[401: 420]
      validation_files_maxar_snow = file_list_maxar_snow[401: 420]
      validation_files_planet_snow = file_list_planet_snow[401: 420]
      training_loss_tracker = open("training_loss_tracker_same_bands.txt", "a")
      validation_loss_tracker = open("validation_loss_tracker_same_bands.txt", "a")

      # Iterate over the file list in batch_fileses
      for i in range(0, min_len, batch_files):
         # Get the batch_files of files
         batch_files_maxar_desert = file_list_maxar_desert[i:i+batch_files]
         batch_files_planet_desert = file_list_planet_desert[i:i+batch_files]
         batch_files_maxar_forest = file_list_maxar_forest[i:i+batch_files]
         batch_files_planet_forest = file_list_planet_forest[i:i+batch_files]

         batch_files_maxar_snow = file_list_maxar_snow[i:i+batch_files]
         batch_files_planet_snow = file_list_planet_snow[i:i+batch_files]


         # TODO: transfer the snow covered maxar and planet images. SHUFFLE EVERYTIME WE LOAD
         dataset = ImageDataset(True, False, maxar_paths_d, batch_files_maxar_desert, planet_paths_d, batch_files_planet_desert, maxar_paths_s, batch_files_maxar_snow, planet_paths_s, batch_files_planet_snow, maxar_paths_f, batch_files_maxar_forest, planet_paths_f, batch_files_planet_snow)
         train_dataset_size += len(dataset)
         data_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
         
         #TODO: load validation dataset. SHUFFLE EVERYTIME WE LOAD
         validation_dataset = ImageDataset(True, False, maxar_paths_d, validation_files_maxar_desert, planet_paths_d, validation_files_planet_desert, maxar_paths_s, validation_files_maxar_snow, planet_paths_s, validation_files_planet_forest, maxar_paths_f, validation_files_maxar_forest, planet_paths_f, validation_files_planet_forest)
         validation_dataset_size += len(validation_dataset)
         validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=True, drop_last=True)



         # TODO: Check if the model file exists, if yes load the parameters
         if os.path.isfile(checkpoint_path):
            # Load the existing model
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint['best_loss']
            print('Existing model loaded successfully.')
         else:
            # Create a new model
            model = SimSiam(2048)
            print('New model created successfully.')
         model.to(device)


         model.train()
         if os.path.isfile(checkpoint_path):
             # Load the existing model
             checkpoint = torch.load(checkpoint_path)
             model.load_state_dict(checkpoint['model_state_dict'])
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             best_loss = checkpoint['best_loss']
             print('Existing model loaded successfully.')
         else:
            # Create a new model
            model = SimSiam(2048)
            print('New model created successfully.')
         model.to(device)
         correct_predictions = 0

         for batch in data_loader:
            maxar_tile, m_type, planet_tile, p_type = batch
               
            maxar_tile = maxar_tile.to(device)
            planet_tile = planet_tile.to(device)
            # Forward pass

            p1, p2, z1, z2 = model(maxar_tile, planet_tile)


            # Compute the loss
            loss = 1 - criterion(p1, z2).mean() + 1 - criterion(p2, z1).mean()

             # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

             # Accumulate loss
            total_loss += loss.item()

          # Calculate average loss
      average_loss = total_loss / train_dataset_size

      # TODO: calculate validation loss
      model.eval()
      for val_batch in validation_data_loader:
         val_maxar_tile, val_m_type, val_planet_tile, val_p_type = val_batch
         val_maxar_tile = val_maxar_tile.to(device)
         val_planet_tile = val_planet_tile.to(device)
         # Forward pass
         val_p1, val_p2, val_z1, val_z2 = model(val_maxar_tile, val_planet_tile)

         # Compute the loss
         val_loss = 1 - criterion(val_p1, val_z2).mean() + 1 - criterion(val_p2, val_z1).mean()
         val_total_loss += val_loss.item()
      val_avg_loss = val_total_loss / validation_dataset_size
             
      training_loss.append(average_loss)
      training_loss_tracker.write(str(average_loss))
      training_loss_tracker.write("\n")

      validation_loss.append(val_avg_loss)
      validation_loss_tracker.write(str(val_avg_loss))
      validation_loss_tracker.write("\n")
      print("Validation Loss: ", val_avg_loss)
      print("Training Loss: ", average_loss)
      print("Current epoch: ", epoch)
      print(f"Epoch [{epoch}], Loss: {average_loss:.4f}")
      if average_loss<best_loss:
         best_loss = average_loss
         
         # Save the model checkpoint
         torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'best_loss': best_loss,
         }, checkpoint_path)
      torch.cuda.empty_cache()
   training_loss_tracker.close()
   validation_loss_tracker.close()
   plot_training_validation_loss(training_loss, validation_loss)



def test():
   # test_start=400
   # # min_len=520
   # min_len=420
   # batch_files = 20
   # maxar_paths_d = "/home/CS/mp0157/dataset/maxar_scenes_desert/"
   # planet_paths_d = "/home/CS/mp0157/dataset/planet_scenes_desert/"
   # maxar_paths_f = "/home/CS/mp0157/dataset/maxar_scenes_forest/"
   # planet_paths_f = "/home/CS/mp0157/dataset/planet_scenes_forest/"
   # maxar_paths_s = "/home/CS/mp0157/dataset/maxar_scenes_snow/maxar_snow_required/"
   # planet_paths_s = "/home/CS/mp0157/dataset/planet_scenes_snow/planet_snow_required/"

   #     # Get the list of files in the folder and sort them
   # file_list_maxar_desert = natsorted(os.listdir(maxar_paths_d))
   # file_list_planet_desert = natsorted(os.listdir(planet_paths_d))
   # file_list_maxar_forest = natsorted(os.listdir(maxar_paths_f))
   # file_list_planet_forest = natsorted(os.listdir(planet_paths_f))
   # file_list_maxar_snow = natsorted(os.listdir(maxar_paths_s))
   # file_list_planet_snow = natsorted(os.listdir(planet_paths_s))

   # for i in range(test_start, min_len, batch_files):
   #    # Get the batch_files of files
   #    batch_files_maxar_desert = file_list_maxar_desert[i:i+batch_files]
   #    batch_files_planet_desert = file_list_planet_desert[i:i+batch_files]
   #    batch_files_maxar_forest = file_list_maxar_forest[i:i+batch_files]
   #    batch_files_planet_forest = file_list_planet_forest[i:i+batch_files]
   #    batch_files_maxar_snow = file_list_maxar_snow[i:i+batch_files]
   #    batch_files_planet_snow = file_list_planet_snow[i:i+batch_files]

   #    checkpoint_path = "model_trail1.pt"
   #    checkpoint = torch.load(checkpoint_path)
   #    model = SimSiam(512)
   #    model.load_state_dict(checkpoint['model_state_dict'])
   #    model.eval()

   #    batch_size=16

   #    dataset = ImageDataset(False, True, maxar_paths_d, batch_files_maxar_desert, planet_paths_d, batch_files_planet_desert, maxar_paths_s, batch_files_maxar_snow, planet_paths_s, batch_files_planet_snow, maxar_paths_f, batch_files_maxar_forest, planet_paths_f, batch_files_planet_snow)
   #    data_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
   #    embeddings = np.zeros((len(dataset),  2048))

   #    i=0
   #    for batch in data_loader:
   #       maxar_tile, m_type, planet_tile, p_type = batch
   #       # maxar_tile = maxar_tile.to(device)
   #       # planet_tile = planet_tile.to(device)
   #       # Forward pass
   #       p1, p2 = model(maxar_tile, planet_tile)
   #       embeddings[i] = np.mean(p1.detach().squeeze().cpu().numpy(), axis=0)
   #       i=i+1
   #       index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
   #       index.add(embeddings)


   test_start=520
   # min_len=550
   min_len=525
   batch_files = 10
   maxar_paths_d = "/home/CS/mp0157/dataset/maxar_scenes_desert/"
   planet_paths_d = "/home/CS/mp0157/dataset/planet_scenes_desert/"
   maxar_paths_f = "/home/CS/mp0157/dataset/maxar_scenes_forest/"
   planet_paths_f = "/home/CS/mp0157/dataset/planet_scenes_forest/"
   maxar_paths_s = "/home/CS/mp0157/dataset/maxar_scenes_snow/maxar_snow_required/"
   planet_paths_s = "/home/CS/mp0157/dataset/planet_scenes_snow/planet_snow_required/"

   # Extract the embedding for the query image
   for i in range(test_start, min_len, batch_files):
      # Get the batch_files of files
      batch_files_maxar_desert = file_list_maxar_desert[i:i+batch_files]
      batch_files_planet_desert = file_list_planet_desert[i:i+batch_files]
      batch_files_maxar_forest = file_list_maxar_forest[i:i+batch_files]
      batch_files_planet_forest = file_list_planet_forest[i:i+batch_files]
      print("batch[0]: ",batch[0])
      print("batch[1]: ",batch[1])
      maxar_tile, m_type, planet_tile, p_type = batch
      with torch.no_grad():
         print("here")
         p1,p2 = model(maxar_tile, planet_tile)

         # Perform similarity search using Faiss
         query_embedding = np.mean(p1.detach().squeeze().cpu().numpy(), axis=0)
         k = 5  # Number of nearest neighbors to retrieve
         distances, indices = index.search(query_embedding.reshape(1, -1), k)
         print("here")

         # Retrieve the nearest neighbors and their distances
         print("indices: ",indices)
         print("indices.shape: ",indices.shape)
         # print(dataset[0])
         for idx in indices[0]:
            print(dataset[idx])
         nearest_distances = distances.squeeze()

         print("nearest distance: ",nearest_distances)






   # Load the saved model
   # model = SimSiam(encoder, num_classes=3)
   # model.load_state_dict(torch.load("simsiam_classification_model.pt"))
   # model.eval()

   # des_base_dir = "/media/hdd2/Meghana/dataset_in_pairs/desert/"
   # test_des_folder_maxar = des_base_dir + "pair_10300100C1A65D00/10300100C1A65D00/"

   # test_des_folder_planet = des_base_dir + "pair_10300100C1A65D00/desert_10300100C1A65D00_2021_06_24_psscene_analytic_8b_sr_udm2/PSScene"

   # # Load the dataset for KNN search
   # knn_dataset = ImageDataset(test_des_folder_maxar, test_des_folder_planet, "f", "f", "s", "s")
   # knn_loader = DataLoader(knn_dataset, batch_size=32)

   # embeddings = []
   # labels = []
   # for images, image_labels in knn_loader:
   #    images = images.to(device)
   #    embeddings.append(model.encoder(images))
   #    labels.extend(image_labels)

   # embeddings = torch.cat(embeddings)
   # embeddings = embeddings.cpu().numpy()

   # # Perform KNN search
   # k = 5  # Number of nearest neighbors to retrieve
   # nn = NearestNeighbors(n_neighbors=k, metric="cosine")
   # nn.fit(embeddings)

   # # Get the feature embeddings for the query image
   # query_image = ...  # Load the query image
   # query_image = transform(query_image).unsqueeze(0).to(device)
   # query_embedding = model.encoder(query_image)

   # # Perform KNN search for the query image
   # distances, indices = nn.kneighbors(query_embedding.cpu().numpy())

   # # Print the nearest neighbors
   # for i, index in enumerate(indices[0]):
   #    neighbor_image = knn_dataset[index][0]  # Get the image of the neighbor
   #    neighbor_label = labels[index]  # Get the label of the neighbor

   #    print(f"Nearest Neighbor {i + 1}: Image Index {index}, Distance {distances[0][i]}, Label {neighbor_label}")

      
      