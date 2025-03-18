# Basic python libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Libraries for ML
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


######################################################################################
######################## NEURAL NETWORK ARCHITECTURE #################################
######################################################################################

# Define the architecture of the Neural Network Model
class nu_1(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        # activation = nn.Tanh
        # activation = nn.ReLU
        activation = nn.ELU
        # activation = nn.LeakyReLU
        # activation = nn.Sigmoid
        # activation = nn.Softplus
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        # self.fce = nn.Sequential(*[
        #                 nn.Linear(N_HIDDEN, N_OUTPUT),
        #                 activation()])
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x     
    
    
class nu_2(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        # activation = nn.Tanh
        # activation = nn.ReLU
        activation = nn.ELU
        # activation = nn.LeakyReLU
        # activation = nn.Sigmoid
        # activation = nn.Softplus
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        # self.fce = nn.Sequential(*[
        #                 nn.Linear(N_HIDDEN, N_OUTPUT),
        #                 activation()])
        
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
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    # If using CUDA, set the seed for all devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
## Function to prepare the ytraining data

# Split dataset into train, test, and validation data
def create_train_val_test_splits(predictors_all, 
                                 n_samples, 
                                 split_frac, 
                                 random_seed
):
    
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

# Create inputs and outputs
def create_scaled_inputs_outputs(train_data, 
                                 val_data, 
                                 test_data, 
                                 cols2scale, 
                                 cols_predicted = ['tip_displacement']
):

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
    
    y_data = torch.tensor(train_data_scaled[cols_predicted].values)
    y_val = torch.tensor(val_data_scaled[cols_predicted].values)
    y_test = torch.tensor(test_data_scaled[cols_predicted].values)
    
    return x_train_scaled, x_val_scaled, x_test_scaled, y_data, y_val, y_test

# Split the ground motions based on train, test, and validation based on index
def split_ground_motions(x_th_all, 
                         train_data, 
                         val_data, 
                         test_data
):
    
    gm_sequence_train = np.sort(train_data['gm_index'].unique())
    gm_sequence_train_df = pd.DataFrame(gm_sequence_train, columns = ['gm_index_old'])
    gm_sequence_train_df['gm_ind_new'] = gm_sequence_train_df.index
    train_data = pd.merge(train_data, gm_sequence_train_df, left_on = 'gm_index', right_on = 'gm_index_old')
    
    gm_sequence_val = np.sort(val_data['gm_index'].unique())
    gm_sequence_val_df = pd.DataFrame(gm_sequence_val, columns = ['gm_index_old'])
    gm_sequence_val_df['gm_ind_new'] = gm_sequence_val_df.index
    val_data = pd.merge(val_data, gm_sequence_val_df, left_on = 'gm_index', right_on = 'gm_index_old')
    
    gm_sequence_test = np.sort(test_data['gm_index'].unique())
    gm_sequence_test_df = pd.DataFrame(gm_sequence_test, columns = ['gm_index_old'])
    gm_sequence_test_df['gm_ind_new'] = gm_sequence_test_df.index
    test_data = pd.merge(test_data, gm_sequence_test_df, left_on = 'gm_index', right_on = 'gm_index_old')
    
    x_th = x_th_all[gm_sequence_train, :]
    x_th_val = x_th_all[gm_sequence_val, :]
    x_th_test = x_th_all[gm_sequence_test, :]
    
    return train_data, val_data, test_data, x_th, x_th_val, x_th_test



########################################################################################
######################## MODEL TRAINING AND VALIDATION #################################
########################################################################################

# Train the MetaIMNet Model
def train_metaimnet_model(x_train_scaled, 
                          x_val_scaled, 
                          y_data, y_val, 
                          x_th, 
                          x_th_val, 
                          train_data, 
                          val_data, 
                          cols2scale, 
                          num_metaims,
                          max_epochs,
                          interval,
                          random_seed
):
    
    set_seed(random_seed)
    
    # LSTM parameters
    n = x_th.shape[0]
    input_d = x_th.shape[1]
    output_d = num_metaims
    
    # NN parameters
    n_input_nn = output_d + len(cols2scale)
    n_output = y_data.shape[1]
    
    # train standard neural network to fit training data
    model1 = nu_1(input_d, output_d, 256, 4)
    model2 = nu_2(n_input_nn, n_output, 24, 3)
    alpha1 = 1e-3
    alpha2 = 1e-3

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=alpha1)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=alpha2)

    criterion = nn.BCELoss()

    error_evolution_training = []
    error_evolution_val = []

    for i in range(max_epochs):
        model1.train()
        model2.train()

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        y_th = model1(x_th.float())

        x_data_th = y_th[train_data['gm_ind_new'].astype(int), :]

        x_data = torch.cat((x_data_th, x_train_scaled), dim = 1)

        yh = model2(x_data.float())

        loss_train = torch.mean((yh-y_data)**2)

        loss = loss_train

        loss.backward()

        optimizer1.step()
        optimizer2.step()

        # plot the result as training progresses
        if ((i+1) % interval != 0) and (i!=0):
            continue

        model1.eval()
        model2.eval()
        with torch.no_grad():
            y_pred_val = predict_metaimnet(model1, model2, x_th_val, x_val_scaled, val_data)

            loss_val = torch.mean((y_pred_val-y_val)**2)

            err_train = np.sqrt(loss_train)
            err_val = np.sqrt(loss_val)

            error_evolution_training.append(float(err_train))
            error_evolution_val.append(float(err_val))


        print("Epoch %d: Training RMSE Loss %.4f, Validation RMSE Loss %.4f" % (i+1, err_train, err_val))
        
    return model1, model2, error_evolution_training, error_evolution_val


###################################################################################
######################## POSTPROCESSING FUNCTIONS #################################
###################################################################################

# Predict using the trained model
def predict_metaimnet(model1, 
                      model2, 
                      x_th, 
                      x_scaled, 
                      data
):
    
    y_th = model1(x_th.float())
    x_th = y_th[data['gm_ind_new'].astype(int), :]
    x_train = torch.cat((x_th, x_scaled), dim = 1)
    y_pred = model2(x_train.float())
    
    return y_pred


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


