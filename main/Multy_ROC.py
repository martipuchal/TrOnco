import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,accuracy_score


import joblib
import xgboost as xgb
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def class_round(num:float, threshold:float):

    """ Rounds the result from the classifier with a threshold.

    :param num: Probability which is going to be rounded.
    :param threshold: Probability is going to be use as a threshold.
    :return: Returns a string classifying the probability.
    """
    if num < threshold:
        return 0
    else:
        return 1

# Seed 
seed = 1999

# Path
Homedir = "tmp/"
path_models = "resources/common/"

#! Load the models

config_df = pd.read_csv(path_models+"config.txt",sep="\t")
RF_threshold, XGB_threshold, TF_threshold = config_df["thresholds"]

# Load RF classifier
RF_model = joblib.load(path_models+"modelRF.bin")

# Load XGBoost model
XGB_model = xgb.XGBClassifier()
XGB_model.load_model(path_models+"modelXGB.json")

# Load TensorFlow model
TF_model = tf.keras.models.load_model(path_models+"modelTF.h5")

# Test with curated data
not_vector_df = pd.read_csv(Homedir+"normal_vector_train",sep=",")
yes_vector_df = pd.read_csv(Homedir+"tumor_vector_train",sep=",")

# Add labels to the dataframes
not_vector_df["target"] = 0
yes_vector_df["target"] = 1
df = pd.concat([not_vector_df,yes_vector_df],ignore_index=True)

# Separate the control variable and the vector
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)

# Evaluate the RF method
y_pred_prob = RF_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = roc_auc_RF = roc_auc_score(y_test, [class_round(val,RF_threshold) for val in y_pred_prob])
print(f'AUC score of the RandomForest: {roc_auc}')

# Plot the ROC curve
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for the RandomForest model')
plt.legend()
plt.savefig(Homedir+'graph/ROC.pdf')
#print("\nClassification Report:\n", classification_rep)


# Evaluate the XGB method
y_pred_prob  = XGB_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = fpr_XGB, tpr_XGB, thresholds_XGB = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = roc_auc_XGB = roc_auc_score(y_test, [class_round(float(val),XGB_threshold) for val in y_pred_prob])
print(f'AUC score of the XGBoost: {roc_auc}')

# Plot the ROC curve
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for th XGBoost model')
plt.legend()
plt.savefig(Homedir+'graph/ROC.pdf')
#print("\nClassification Report:\n", classification_rep)


# TensorFlow model (CNN)
y_pred_prob = []
for vector in X_test:
    input_X = np.array(vector, dtype='float32').reshape(-1, 28, 1)
    #prob_positive = TF_model.predict(input_X)[0][0]
    prob_positive = TF_model.predict(input_X)[0, 1]
    y_pred_prob.append(prob_positive)

fpr, tpr, thresholds = fpr_TF, tpr_TF, thresholds_TF = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = roc_auc_TF = roc_auc_score(y_test, [class_round(float(val),TF_threshold) for val in y_pred_prob])
print(f'AUC score of the TensorFlow: {roc_auc}')

# Plot the ROC curve
plt.figure()  
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for th TensorFlow model')
plt.legend()
plt.savefig(Homedir+'graph/ROC.pdf')
#print("\nClassification Report:\n", classification_rep)

# Plot the ROC curve for all models
plt.figure()  
plt.plot(fpr_RF, tpr_RF, label='ROC curve RF(area = %0.2f)' % roc_auc_RF)
plt.plot(fpr_XGB, tpr_XGB, label='ROC curve XGB(area = %0.2f)' % roc_auc_XGB)
plt.plot(fpr_TF, tpr_TF, label='ROC curve TF(area = %0.2f)' % roc_auc_TF)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for the different models')
plt.legend()
plt.savefig(Homedir+'graph/Multy_ROC.pdf')