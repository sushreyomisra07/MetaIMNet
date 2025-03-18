# MetaIMNet
Codes and Input Datasets required to train and test MetaIMNet surrogate models and compare to Traditional Surrogate Models.
The codes are in python and the demonstrations are provided in the form of Jupyter Notebooks. It uses "pytorch" and "sklearn" libraries for the machine learning tasks.

The Jupyter Notebooks are annotated as follows. 
  * TradNet_classification_train.ipynb: Train the TradNet classification model and check model performance
  * TradNet_regression_train.ipynb: Train the TradNet regression model and check model performance
  * MetaIMNet_classification_train.ipynb: Train the MetaIMNet classification model and check model performance
  * MetaIMNet_regression_train.ipynb: Train the MetaIMNet regression model and check model performance

The input python codes for gathering the data and features, preprocessing, training the model, and postporcessing the outputs are provided in the form of four python scripts, each of which corresponds to one of the above Jupyter Notebooks.

The structural and some of the ground motion features which are required inputs are stored in a parquet file titled 'predictors.parquet'. The computed ground motion features referenced by a unique ground motion id contained in 'predictors.parquet' are provided in a separate parquet file called 'pga_all_new.parquet'.

##################################################

IMPORTANT NOTE ABOUT REQUIRED INPUT TIME HISTORIES

##################################################

For running the Meta-IMNet models, the full time histories are required. The time histories are stacked into a 2D tensor in pytorch, with each row representing a unique time history. Since the tensor structure requires all rows to have the same length, all time histories shorter than the longest time history are padded with zeros at the end. This input file is a large 3.72 GB file, and it cannot be loaded into GitHub. As a result, the time histories were copied into a DropBox folder and the link is provided here: https://www.dropbox.com/scl/fi/86hyuxr59btve1ekwnuey/x_gm?rlkey=f0dgl18b5favyecmrovjmf2ik&st=oafrzb8l&dl=0

Please download the dataset and update the file path in your Notebook to incorporate it (you may need to update other file paths as well).
