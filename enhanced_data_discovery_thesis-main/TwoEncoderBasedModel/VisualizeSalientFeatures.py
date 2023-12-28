import matplotlib.pyplot as plt
import torch

def visualize_feature_map():
    model.eval()
    sample_input = torch.randn(1, num_channels, height, width)

    # Get the feature maps of a specific layer
    target_layer = model.conv_layer  # Replace with the actual layer

    # Forward pass
    activations = model(sample_input)

    # Visualize the feature maps using a grid of images
    num_features = activations.size(1)
    fig, axes = plt.subplots(1, num_features, figsize=(15, 2))
    for i in range(num_features):
        feature_map = activations[0, i].detach().cpu().numpy()
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].axis('off')
    plt.show()

def visualize_activations():
    model.eval()
    sample_input = torch.randn(1, num_channels, height, width)

    # Get the activations of a specific layer
    target_layer = model.hidden_layer  # Replace with the actual layer

    # Forward pass
    activations = model(sample_input)

    # Visualize the activations using a heatmap
    activation_map = activations.squeeze().detach().cpu().numpy()
    plt.imshow(activation_map, cmap='viridis')
    plt.show()

        
