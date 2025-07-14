# Basic python libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Libraries for ML
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
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
        activation1 = nn.ELU
        # activation1 = nn.LeakyReLU
        # activation1 = nn.Sigmoid
        activation2 = nn.Sigmoid
        # activation = nn.Linear
        # activation = nn.Softplus
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation1()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation1()]) for _ in range(N_LAYERS-1)])
        # self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.fce = nn.Sequential(*[
                        nn.Linear(N_HIDDEN, N_OUTPUT),
                        activation2()])
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
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
                                 cols_predicted = ['damage_state']
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
                          max_epochs = 250,
                          automate_stopping = True,
                          interval = 50,
                          random_seed = 7,
                          alpha1 = 1e-3,
                          alpha2 = 1e-3,
                          nlayers1 = 4,
                          nhidden1 = 256,
                          nlayers2 = 3,
                          nhidden2 = 24
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
    model1 = nu_1(input_d, output_d, nhidden1, nlayers1)
    model2 = nu_2(n_input_nn, n_output, nhidden2, nlayers2)
    # alpha1 = 1e-3
    # alpha2 = 1e-3

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=alpha1)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=alpha2)

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss() 

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

        loss_train = criterion(yh, y_data.float())

        loss = loss_train

        loss.backward()

        optimizer1.step()
        optimizer2.step()

        # # plot the result as training progresses
        # if ((i+1) % interval != 0) and (i!=0):
        #     continue

        model1.eval()
        model2.eval()
        with torch.no_grad():
            _, y_pred_val = predict_metaimnet(model1, model2, x_th_val, x_val_scaled, val_data)

            loss_val = criterion(y_pred_val, y_val.float())

            err_train = np.sqrt(loss_train)
            err_val = np.sqrt(loss_val)

            error_evolution_training.append(float(err_train))
            error_evolution_val.append(float(err_val))


        if ((i+1) % interval == 0) or (i==0):
            print("Epoch %d: Training BCE Loss %.4f, Validation BCE Loss %.4f" % (i+1, err_train, err_val))
        
        if automate_stopping & (i > 100):
            err_window = error_evolution_val[-30:]
            err_diff = np.diff(err_window)
            global_diff = error_evolution_val[i] - error_evolution_val[i-30]
            if np.all(err_diff>0) or (global_diff > 0.04*err_val):
                print("Epoch %d: Final Training BCE Loss %.4f, Final Validation BCE Loss %.4f" % (i+1, err_train, err_val))
                break
    
    return model1, model2, error_evolution_training, error_evolution_val


# Train the MetaIMNet Model
def train_metaimml_model(x_train_scaled, 
                          x_val_scaled, 
                          y_data, y_val, 
                          x_th, 
                          x_th_val, 
                          train_data, 
                          val_data, 
                          cols2scale, 
                          num_metaims,
                          max_epochs = 250,
                          regularization = True,
                          reg_type = 'Elastic Nets',
                          automate_stopping = True,
                          interval = 50,
                          random_seed = 7,
                          alpha1 = 1e-3,
                          alpha2 = 1e-3,
                          nlayers1 = 4,
                          nhidden1 = 256,
                          nlayers2 = 3,
                          nhidden2 = 24
):
    
    set_seed(random_seed)
    
    # LSTM parameters
    n = x_th.shape[0]
    input_d = x_th.shape[1]
    output_d = num_metaims

    l2_lambda = 0.1
    l1_lambda = 0.01
    
    # NN parameters
    n_input_nn = output_d + len(cols2scale)
    n_output = y_data.shape[1]
    
    # train standard neural network to fit training data
    model1 = nu_1(input_d, output_d, nhidden1, nlayers1)
    # model2 = nu_2(n_input_nn, n_output, nhidden2, nlayers2)
    model2 = LogisticRegressionModel(input_dim=n_input_nn)
    # alpha1 = 1e-3
    # alpha2 = 1e-3

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=alpha1)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=alpha2)
    optimizer2 = optim.SGD(model2.parameters(), lr=alpha2)

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss() 

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

        loss_train = criterion(yh, y_data.float())

        loss = loss_train
        if regularization:
        # Add L2 penalty: sum of squared weights (exclude bias)
            if reg_type == 'L2':
                l2_norm = sum(param.pow(2.0).sum() for name, param in model2.named_parameters() if 'bias' not in name)
                loss = loss + l2_lambda * l2_norm
            elif reg_type == 'L1':
                l1_norm = sum(param.abs().sum() for name, param in model2.named_parameters() if 'bias' not in name)
                loss = loss + l1_lambda * l1_norm
            else:
                l2_norm = sum(param.pow(2.0).sum() for name, param in model2.named_parameters() if 'bias' not in name)
                l1_norm = sum(param.abs().sum() for name, param in model2.named_parameters() if 'bias' not in name)
                loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
        
        loss.backward()

        optimizer1.step()
        optimizer2.step()

        # # plot the result as training progresses
        # if ((i+1) % interval != 0) and (i!=0):
        #     continue

        model1.eval()
        model2.eval()
        with torch.no_grad():
            _, y_pred_val = predict_metaimnet(model1, model2, x_th_val, x_val_scaled, val_data)

            loss_val = criterion(y_pred_val, y_val.float())

            err_train = np.sqrt(loss_train)
            err_val = np.sqrt(loss_val)

            error_evolution_training.append(float(err_train))
            error_evolution_val.append(float(err_val))


        if ((i+1) % interval == 0) or (i==0):
            print("Epoch %d: Training BCE Loss %.4f, Validation BCE Loss %.4f" % (i+1, err_train, err_val))
        
        if automate_stopping & (i > 100):
            err_window = error_evolution_val[-10:]
            err_diff = np.diff(err_window)
            global_diff = error_evolution_val[i] - error_evolution_val[i-10]
            if np.all(err_diff>0) or (global_diff > 0.04*err_val):
                print("Epoch %d: Final Training BCE Loss %.4f, Final Validation BCE Loss %.4f" % (i+1, err_train, err_val))
                break
    
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
    
    return x_th, y_pred


# Plotting functions
def plot_confusion(data, true_label, pred_label, classes):

    # Build confusion matrix
    cf_matrix = confusion_matrix(data[true_label], data[pred_label])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])

    plt.figure(figsize = (10,6))
    g = sn.heatmap(df_cm, annot=True, annot_kws={"size":24}, cmap = 'crest')
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 20)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 20)
    plt.xlabel("Predicted", fontsize = 24)
    plt.ylabel("True", fontsize = 24)
    # sn.set(font_scale=2.5)
    return df_cm

def plot_auc_roc(data, true_label, pred_label):
    """
    plots the roc curve based of the probabilities
    """
    
    fpr, tpr, thresholds = roc_curve(np.array(data[true_label]), np.array(data[pred_label]))
    plt.figure(figsize = (5,4))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate', fontsize = 16)
    plt.ylabel('True Positive Rate', fontsize = 16)
    auc_roc = roc_auc_score(np.array(data[true_label]), np.array(data[pred_label]))
    plt.title('ROC Curve, AUC_ROC = {}'.format(round(auc_roc, 3), fontsize = 16))
    plt.grid()


###################################################################################
######################## TRAIN FRAGILITY MODELS ###################################
###################################################################################

def develop_fragility_model_stripe(data, 
                                   im_col, 
                                   ds_col, 
                                   bin_width = 0.1, 
                                   max_im = 1.5,
                                   mu = 0.6,
                                   beta = 0.15):
    
    fragility_df = pd.DataFrame(columns = ['IM', 'Probability']) 
    
    # Loop over stripes
    im_start = 0.0
    im_end = im_start
    stripe = 0
    
    # Calculate probability of damage for each stripe
    while im_end <= max_im:
        im_end += bin_width
        
        data_stripe = data[(data[im_col] > im_start) & (data[im_col] <= im_end)]
        rep_im = 0.5*(im_start + im_end)
        prob_damage = data_stripe[ds_col].sum()/data_stripe.shape[0]
        
        fragility_df.loc[stripe, 'IM'] = rep_im
        fragility_df.loc[stripe, 'Probability'] = prob_damage
        
        stripe += 1
        im_start = im_end
    
    return fragility_df
