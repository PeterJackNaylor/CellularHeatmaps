
from segmentation_net.unet_distance import DistanceUnet

def create_model(log, mean_array, n_features):

    variables_model = {
        ## Model basics
        "log": log, 
        "num_channels": 3,
        'mean_array': mean_array,
        "n_features": n_features
    }

    model = DistanceUnet(**variables_model)

    return model
