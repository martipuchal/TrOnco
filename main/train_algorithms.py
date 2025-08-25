import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import joblib
import random

# Evaluate the model:
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.callback import TrainingCallback

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten,Dense, Dropout, LeakyReLU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import (
    Conv1D, Dense, LeakyReLU, MaxPooling1D,
    Dropout, Flatten, BatchNormalization,
    SpatialDropout1D, GlobalAveragePooling1D,Lambda
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# For LeakyReLU
from keras.layers import LeakyReLU  
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Automatize the modifications
import argparse
parser = argparse.ArgumentParser(
    prog= "Algorithm train",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-s","--save",default="False")
args = parser.parse_args()

seed = 1999

# Directories
path = "tmp/"

save_path = "tmp/"

# Functions:
def round(num:int, threshold:int):
    if num<threshold:
        return 0
    else:
        return 1
    
def ROC ():
    # Plot the ROC curve 
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for the different models')
    plt.legend()
    plt.savefig(path+'Multy_ROC.png')
    #print("\nClassification Report:\n", classification_rep)



notoncofusions = pd.read_csv(path+"normal_vector_train",sep=",")
oncofusions = pd.read_csv(path+"tumor_vector_train",sep=",")


# Build the datafrmaes
oncofusions["target"] = 1
notoncofusions["target"] = 0
df = pd.concat([notoncofusions,oncofusions],ignore_index=True)

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split teh data in to train/test
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.1, random_state=seed, stratify=y)

## Random forest ##
print("\n","##Random Forest moodel:##")
classifier = RandomForestClassifier(n_estimators=200, random_state=seed)  
classifier.fit(X_train, y_train)

# Evaluate the RF method
y_pred_prob_RF = classifier.predict_proba(X_test)[:, 1]

# Save the RF
if args.save == "False":
    joblib.dump(classifier, save_path+"modelRF.bin")


## XGBoost: ##
print("\n","##XGBoost moodel:##")
# Prepare the data for the model
xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# Set up the parameters is going to use to classify
n=1000

param = {'max_depth': 6, 'eta': 0.6,"min_child_weight":12, "subsample": 1, "colsample_bytree":0.8,
         "seed_per_iteration":True,"base_score":0.8}

param['nthread'] = 4
param['eval_metric'] = 'auc' # auc

# Train the model
evallist = [(xgb_train, 'train'), (xgb_test, 'eval')]
model = xgb.train(params=param,dtrain=xgb_train,num_boost_round=n,evals = evallist,verbose_eval=True, early_stopping_rounds=3)

# Evaluate the model
y_pred_prob_XG =  model.predict(xgb_test)

# Save the XGBoost
if args.save == "False":
    model.save_model(save_path+"modelXGB.json")


## CNN: ##

print("\n","##CNN moodel:##")
# prepare the model
model = Sequential([
    Conv1D(5, kernel_size=5, activation='linear', padding='same', input_shape=(28, 1),kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
    MaxPooling1D(),
    Conv1D(5, kernel_size=5, activation='linear', padding='same',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),
    MaxPooling1D(),
    Dropout(0.1,seed=seed),
    Flatten(),
    LeakyReLU(alpha=0.1),
    Dense(2, activation='tanh',kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)),  # 2 neurons for binary classification
    Lambda(lambda x: (x + 1) / 2)
])

# Compile with binary settings
model.compile(
    optimizer="adam",
    loss='binary_crossentropy',
    metrics=[
        'accuracy',  # Standard accuracy
        tf.keras.metrics.Recall(name='sensitivity'),  # Sensitivity = Recall
        tf.keras.metrics.Precision(),
        tf.keras.metrics.AUC()
    ]
)
X_train = X_train.reshape(-1, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 1).astype('float32')
y_train = y_train.astype('float32')  
y_test = y_test.astype('float32')
# Convert labels to one-hot
y_train_onehot = to_categorical(y_train, num_classes=2)
y_test_onehot = to_categorical(y_test, num_classes=2)

early_stopping = EarlyStopping(
    monitor='val_auc',  # Monitor validation AUC
    patience=2,         # Stop after 4 epochs without improvement
    mode='max',         # Maximize AUC (default for AUC)
    verbose=1
)



history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=30,
    verbose=1, callbacks=early_stopping,batch_size=10,validation_split=0.1,
    class_weight={0: 0.666, 1: 2.0}
)

if args.save == "False":
    model.save(save_path+'modelTF.h5')

y_pred_prob_CNN = []
for vector in X_test:
    input_X = np.array(vector, dtype='float32').reshape(-1, 28, 1)
    #prob_positive = TF_model.predict(input_X)[0][0]
    prob_positive = model.predict(input_X)[0, 1]
    y_pred_prob_CNN.append(prob_positive)



with open(save_path+"config.txt", "w") as f:
    f.write("\t".join(["fpr", "tpr", "thresholds"])+"\n")
    plt.figure()  
    for pred,model in zip([y_pred_prob_RF, y_pred_prob_XG, y_pred_prob_CNN],["RandomForest","XGBoost","CNN"]):
        fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)
        def obtain_t ():
            fpr_max = max(fpr)
            for f, t, th in zip(fpr, tpr, thresholds):
                if t >= max(tpr) and f<=fpr_max :

                    fpr_max = f

                    f_return = f
                    t_return = t
                    th_return = th
                    
            return [f_return,t_return,th_return]
                
        line = obtain_t()
        roc_auc = roc_auc_score(y_test, [round(float(val),line[2]) for val in pred])
        plt.plot(fpr, tpr, label='ROC curve '+model+' (area = %0.2f)' % roc_auc)

        f.write("\t".join([str(e) for e in line])+"\n")
    ROC()


for pred,model in zip([y_pred_prob_RF, y_pred_prob_XG, y_pred_prob_CNN],["RandomForest","XGBoost","CNN"]):
    plt.figure()
    cm = confusion_matrix(y_test, [round(float(val),line[2]) for val in pred])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix of {model}')
    plt.savefig(path+f'confuM_{model}.png')