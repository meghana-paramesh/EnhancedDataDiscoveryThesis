# import matplotlib.pyplot as plt
# import torch
# from SimsiamSameEncoderModel import SimSiam
# import numpy as np
# import torch
# import torch.optim as optim
# def normalize(array):
#     array_min, array_max = array.min(), array.max()
#     return (array - array_min) / (array_max - array_min)

# def visualize_feature_map():
#     maxar_path = "/home/CS/mp0157/dataset/maxar_tiles_test/irrigation_398_178.npy"
#     planet_path = "/home/CS/mp0157/dataset/planet_tiles_test/urban_area_398_51.npy"
#     # model_path = 'model_rest_plotting.pth'

#     # checkpoint = torch.load(model_path)
    
#     # model = SimSiam(2048)  
#     # model.load_state_dict(checkpoint['model_state_dict'])
#     # model.eval() 

#     maxar_data = np.load(maxar_path)
#     planet_data = np.load(planet_path)
#     nir = maxar_data[4]
#     red = maxar_data[5]
#     green = maxar_data[3]
#     nir_norm = normalize(nir)
#     red_norm = normalize(red)
#     green_norm = normalize(green)
#     nrg = np.dstack((nir_norm, red_norm, green_norm))
#     plt.imsave("maxar_irr1.png", nrg)
#     # print(maxar_data.shape)
#     # print(planet_data.shape)
#     # p1, p2, z1, z2 = model(torch.from_numpy(np.expand_dims(maxar_data, axis=0)), torch.from_numpy(np.expand_dims(planet_data, axis=0)))
#     # target_layer = model.projector
#     # input_image = torch.from_numpy(np.expand_dims(maxar_data, axis=0))
#     # optimizer = optim.RAdam([input_image], lr=0.001)
#     # num_iterations = 100  # Adjust the number of iterations as needed
#     # for _ in range(num_iterations):
#     #     optimizer.zero_grad()
#     #     output = target_layer(model.encoder(input_image))
#     #     loss = activation_maximization_loss(output)
#     #     loss.backward()
#     #     optimizer.step()
#     # optimized_image = input_image.detach().squeeze().permute(1, 2, 0).cpu().numpy()
#     # print("optimized_image: ",optimized_image.shape)
#     # for band_idx in range(optimized_image.shape[2]):
#     #     plt.subplot(1, 7, band_idx + 1)
#     #     plt.imshow(optimized_image[:, :, band_idx], cmap='gray')
#     #     plt.title(f'Band {band_idx + 1}')
#     #     plt.axis('off')
#     # plt.savefig('salient_feature_visualization.png')

# def activation_maximization_loss(output):
#     return -output.mean() 

# def visualize_activations():
#     model.eval()
#     sample_input = torch.randn(1, num_channels, height, width)

#         # Get the activations of a specific layer
#     target_layer = model.hidden_layer  # Replace with the actual layer

#     # Forward pass
#     activations = model(sample_input)

#     # Visualize the activations using a heatmap
#     activation_map = activations.squeeze().detach().cpu().numpy()
#     plt.imshow(activation_map, cmap='viridis')
#     plt.show()

# def visualize_all_layers_encoder():
#     model_path = 'model_rest_plotting.pth'
#     checkpoint = torch.load(model_path)
    
#     model = SimSiam(2048)  
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval() 
#     encoder = model.encoder
#     maxar_path = "/home/CS/mp0157/dataset/maxar_tiles_test/urban_area_398_51.npy"
#     planet_path = "/home/CS/mp0157/dataset/planet_tiles_test/urban_area_398_51.npy"
#     transform1, transform2 = get_transform()

#     # Choose the intermediate layers for visualization (e.g., layers 2, 3, and 4)
#     target_layers = [encoder.layer2, encoder.layer3, encoder.layer4]

#     # Load and preprocess an example image
#     image_path = 'example_image.jpg'
#     image = Image.open(image_path)
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_image = preprocess(image).unsqueeze(0)

#     # Function to hook into the intermediate layers and extract feature maps
#     activations = {}
#     def hook_fn(module, input, output):
#         activations[module] = output
#     for layer in target_layers:
#         layer.register_forward_hook(hook_fn)

#     # Forward pass to compute feature maps
#     with torch.no_grad():
#         model(input_image)

#     # Visualize the feature maps from multiple layers together
#     for i, layer in enumerate(target_layers):
#         feature_maps = activations[layer][0]
#         num_feature_maps = feature_maps.size(0)
        
#         plt.figure(figsize=(10, 4))
#         plt.suptitle(f'Layer {i + 2} Feature Maps')
        
#         for j in range(num_feature_maps):
#             plt.subplot(4, num_feature_maps // 4, j + 1)
#             plt.imshow(feature_maps[j].cpu(), cmap='viridis')
#             plt.axis('off')

#     plt.show()


# visualize_feature_map()



        




import matplotlib.pyplot as plt
import torch
from SimsiamSameEncoderModel import SimSiam
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from skimage import exposure 
import os

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def visualize_feature_map():
    maxar_path = "/home/CS/mp0157/dataset/new_tiles/maxar_irr_tiles/irrigation_182_110.npy"
    # planet_path = "/home/CS/mp0157/dataset/planet_tiles_test/irrigation_398_178.npy"
    # model_path = 'models/maxar_model.pth'

    # checkpoint = torch.load(model_path)
    
    # model = SimSiam(8, 2048)  
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval() 

    maxar_data = np.load(maxar_path)
    # planet_data = np.load(planet_path)
    nir = maxar_data[4]
    red = maxar_data[5]
    green = maxar_data[3]
    nir_norm = normalize(nir)
    red_norm = normalize(red)
    green_norm = normalize(green)
    nrg = np.dstack((nir_norm, red_norm, green_norm))
    plt.imsave("maxar_irr1.png", nrg)
    # print(maxar_data.shape)
    # print(planet_data.shape)
    # p1, p2, z1, z2 = model(torch.from_numpy(np.expand_dims(maxar_data, axis=0)), torch.from_numpy(np.expand_dims(maxar_data, axis=0)))
    # target_layer = model.projector
    # input_image = torch.from_numpy(np.expand_dims(maxar_data, axis=0))
    # optimizer = optim.RAdam([input_image], lr=0.001)
    # num_iterations = 100  # Adjust the number of iterations as needed
    # for _ in range(num_iterations):
    #     optimizer.zero_grad()
    #     output = target_layer(model.encoder(input_image))
    #     loss = activation_maximization_loss(output)
    #     loss.backward()
    #     optimizer.step()
    # optimized_image = input_image.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    # print("optimized_image: ",optimized_image.shape)
    # for band_idx in range(optimized_image.shape[2]):
    #     plt.subplot(1, 7, band_idx + 1)
    #     plt.imshow(optimized_image[:, :, band_idx], cmap='gray')
    #     plt.title(f'Band {band_idx + 1}')
    #     plt.axis('off')
    # plt.savefig('salient_feature_visualization_irr.png')

visualize_feature_map()