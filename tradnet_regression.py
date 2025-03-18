# Basic python libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Libraries for ML
import torch
import torch.nn as nn
import torch.optim as optim
# import seaborn as sn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


######################################################################################
######################## NEURAL NETWORK ARCHITECTURE #################################
######################################################################################

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        # activation = nn.Tanh
        # activation = nn.ReLU
        # activation = nn.ELU
        # activation = nn.LeakyReLU
        activation = nn.Sigmoid
        # activation = nn.Softplus
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
##################################################################################
######################## PREPROCESSING FUNCTIONS #################################
##################################################################################
    
# Function to set the random seed for reproducibility
def set_seed(seed_value):
    # Set the seed for random number generation
    torch.manual_seed(seed_value)
    
    
## Function to prepare the ytraining data

def create_train_val_test_splits(predictors_all, n_samples, split_frac, random_seed):
    set_seed(random_seed)
    
    predictors = predictors_all.sample(n = n_samples, random_state = random_seed)
    predictors_test = predictors_all[~predictors_all.index.isin(predictors.index)]

    predictors = predictors.reset_index(drop = True)
    predictors_test = predictors_test.reset_index(drop = True)
    
    val_start = n_samples * split_frac

    train_data = predictors.loc[:val_start-1]
    val_data = predictors.loc[val_start:].reset_index(drop = True)
    test_data = predictors_test.copy()
    
    return train_data, val_data, test_data

def create_scaled_inputs_outputs(train_data, val_data, test_data, cols2scale, cols_gm, cols_predicted = ['tip_displacement']):
    
    cols_all = cols2scale + cols_gm

    x_to_scale_train = torch.tensor(train_data[cols2scale].values)
    x_to_scale_val = torch.tensor(val_data[cols2scale].values)
    x_to_scale_test = torch.tensor(test_data[cols2scale].values)
        
    # Scale Data
    scaler = StandardScaler()
    scaler.fit(x_to_scale_train)

    x_train_scaled = torch.tensor(scaler.transform(x_to_scale_train))
    x_val_scaled = torch.tensor(scaler.transform(x_to_scale_val))
    x_test_scaled = torch.tensor(scaler.transform(x_to_scale_test))
    
    train_data_scaled = train_data.copy()
    val_data_scaled = val_data.copy()
    test_data_scaled = test_data.copy()
    
    train_data_scaled[cols2scale] = x_train_scaled
    val_data_scaled[cols2scale] = x_val_scaled
    test_data_scaled[cols2scale] = x_test_scaled
    
    x_data = torch.tensor(train_data_scaled[cols_all].values)
    x_val = torch.tensor(val_data_scaled[cols_all].values)
    x_test = torch.tensor(test_data_scaled[cols_all].values)
    
    y_data = torch.tensor(train_data_scaled[cols_predicted].values)
    y_val = torch.tensor(val_data_scaled[cols_predicted].values)
    y_test = torch.tensor(test_data_scaled[cols_predicted].values)
    
    return x_data, x_val, x_test, y_data, y_val, y_test



########################################################################################
######################## MODEL TRAINING AND VALIDATION #################################
########################################################################################

def train_tradnet_model(x_data, 
                        x_val, 
                        y_data, 
                        y_val, 
                        train_data, 
                        val_data,
                        max_epochs = 2000,
                        interval = 10):
    
    # NN parameters
    n_input_nn = x_data.shape[1]
    n_output = y_data.shape[1]
    
    # train standard neural network to fit training data
    model = FCN(n_input_nn, n_output, 24, 3)
    alpha = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    error_evolution_training = []
    error_evolution_val = []

    for i in range(max_epochs):
        model.train()

        optimizer.zero_grad()

        yh = model(x_data.float())

        loss_train = torch.mean((yh-y_data)**2)

        loss = loss_train

        loss.backward()

        optimizer.step()

        # plot the result as training progresses
        if ((i+1) % interval != 0) and (i!=0):
            continue

        model.eval()
        with torch.no_grad():
            y_pred_val = predict_tradnet(model, x_val)

            loss_val = torch.mean((y_pred_val-y_val)**2)

            err_train = np.sqrt(loss_train)
            err_val = np.sqrt(loss_val)

            error_evolution_training.append(float(err_train))
            error_evolution_val.append(float(err_val))

        print("Epoch %d: Training RMSE Loss %.4f, Validation RMSE Loss %.4f" % (i+1, err_train, err_val))
        
    return model, error_evolution_training, error_evolution_val


###################################################################################
######################## POSTPROCESSING FUNCTIONS #################################
###################################################################################

def predict_tradnet(model, x_data):
    
    y_pred = model(x_data.float())
    
    return y_pred


# Plotting functions
# Plotting functions    
def plot_scatter_results(data, title_text):
    fig, ax = plt.subplots(1,1, figsize = (5, 5))

    x_truth = np.arange(0, 0.5+data['tip_displacement'].max(), 0.5)
    y_truth = x_truth

    ax.scatter(data['tip_displacement'], data['tip_displacement_pred'], label = 'Trained Model')
    ax.plot(x_truth, y_truth, color = 'r', label = 'Perfect Model')

    error = ((data['tip_displacement_pred']-data['tip_displacement'])**2).mean()
    r2 = r2_score(data['tip_displacement'], data['tip_displacement_pred'])
    error = error**0.5

    ax.legend()
    ax.set_xlabel('Ground Truth Tip Disp (in)', fontsize = 18)
    ax.set_ylabel('Predicted Tip Disp (in)', fontsize = 18)
    ax.set_title(title_text, fontsize = 18)
    
    ax.grid()
    plt.tight_layout()
    plt.show()
    
    print('RMSE = {}, R2 = {}'.format(round(error, 3), round(r2, 3)))
