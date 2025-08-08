import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,accuracy_score
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

class EvalHistoryCallback(TrainingCallback):
    def __init__(self):
        self.history = {}
    
    def after_iteration(self, model, epoch, evals_log):
        # Store evaluation metrics
        for data_name, metrics in evals_log.items():
            if data_name not in self.history:
                self.history[data_name] = {}
            for metric_name, values in metrics.items():
                if metric_name not in self.history[data_name]:
                    self.history[data_name][metric_name] = []
                self.history[data_name][metric_name].append(values[-1])
        return False  # Return False to continue training


seed = 1999

# Automatize the modifications
import argparse
parser = argparse.ArgumentParser(
    prog= "Algorithm train",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Arguments to add parameters to the different algorithms. 
parser.add_argument("-d","--depth",default=9)
parser.add_argument("-c","--child",default=35) # 20
#parser.add_argument("-t","--tree",default="approx",choices=("hist", "approx"))

# Store the parameters in a variable.
args = parser.parse_args()

depth = int(args.depth)
child = int(args.child)
#tree_m = str(args.tree)

# Directories
path = "tmp/"
save_path = "resources/common/"

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
y_pred = classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
classification_rep = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Sensitivity: {sensitivity * 100:.2f}%')
#print("\nClassification Report:\n", classification_rep)

# Save the RF
joblib.dump(classifier, save_path+"modelRF.bin")

def round(num:int):
    if num<0.5:
        return 0
    else:
        return 1

# ROC curve for the RF model
y_pred_prob = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = evalRF = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = roc_auc_score(y_test, [round(val) for val in y_pred_prob])

print("ROC AUC of the random forest model",roc_auc)


## XGBoost: ##

print("\n","##XGBoost moodel:##")
# Prepare the data for the model
xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# Set up the parameters is going to use to classify
n=1000

param = {'max_depth': depth, 'eta': 0.131,"min_child_weight":child, "subsample": 1, "colsample_bytree":0.949,"colsample_bylevel":0.94,
         "base_score":0.8,"seed_per_iteration":True}

param['nthread'] = 4
param['eval_metric'] = 'auc' # auc

# Train the model
eval_history = EvalHistoryCallback()
evallist = [(xgb_train, 'train'), (xgb_test, 'eval')]
model_base = xgb.train(params=param,dtrain=xgb_train,num_boost_round=n,evals = evallist,verbose_eval=True, early_stopping_rounds=3,callbacks=[eval_history])

# AUC evaluation
df = pd.DataFrame({
    'iteration': range(len(eval_history.history['train']['auc'])),
    'train_auc': eval_history.history['train']['auc'],
    'eval_auc': eval_history.history['eval']['auc']
})
plt.plot(df["eval_auc"])
#plt.show()

# Select itineration
max_val = 0
iteration = 0
for i,val in enumerate(df["eval_auc"]):
    val = float(val)
    if val>max_val:
        max_val = val
        iteration = i
    if val<max_val:
        max_val = 2
print("Selected itineration: ",iteration)
#iteration = int(input("Select iteration to use: "))

model = model_base[iteration]

# Evaluate the model
preds = noroundpreds =  model.predict(xgb_test)

def round (num:float):
    """Modify the threshold of the function.

	:param num: A float, between 0,1. 
	:return: An int 1 or 0.
	"""
    if num < 0.8:
        return 0
    else:
        return 1
    
preds =  [round(e) for e in preds]
accuracy= accuracy_score(y_test,preds)

conf_matrix = confusion_matrix(y_test, preds)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel().tolist()

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Sensitivity: {sensitivity * 100:.2f}%')

roc_auc = roc_auc_score(y_test, preds)
print("ROC AUC of the XGBoost model",roc_auc)

classification_rep = classification_report(y_test, preds)

#print("\nClassification Report:\n", classification_rep)
# Save the model
model.save_model(save_path+"modelXGB.json")

fpr, tpr, thresholds = evalXGB = roc_curve(y_test, noroundpreds, pos_label=1)

## Binary Classification Model (pos/neg) with CNN (TensorFlow)
## test
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
#y_train_onehot = y_train.astype('float32')  
#y_test_onehot =y_test.astype('float32')

early_stopping = EarlyStopping(
    monitor='val_auc',  # Monitor validation AUC
    patience=2,         # Stop after 4 epochs without improvement
    mode='max',         # Maximize AUC (default for AUC)
    verbose=1
)

checkpoint = ModelCheckpoint(
    save_path+'modelTF.h5',    # Save path
    monitor='val_auc',  # Save the best model based on val_auc
    save_best_only=True,
    mode='max',         # Save when AUC increases
    verbose=1
)



history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=30,
    verbose=1, callbacks=[early_stopping,checkpoint],batch_size=10,validation_split=0.1,
    class_weight={0: 0.666, 1: 2.0}
)


train_auc = history.history['auc']
val_auc = history.history['val_auc']
# Extract metrics
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_sensitivity = history.history['sensitivity']  # Recall for positive class
val_sensitivity = history.history['val_sensitivity']

# Print final values
print(f"\nFinal Training Accuracy: {train_accuracy[-1]:.4f}")
print(f"Final Validation Accuracy: {val_accuracy[-1]:.4f}")
print(f"\nFinal Training Sensitivity: {train_sensitivity[-1]:.4f}")
print(f"Final Validation Sensitivity: {val_sensitivity[-1]:.4f}")

# Print final AUC scores
print(f"\nTraining AUC: {train_auc[-1]:.4f}")
print(f"Validation AUC: {val_auc[-1]:.4f}")

model.save(save_path+'modelTF.h5')

y_pred_prob = []
for vector in X_test:
    input_X = np.array(vector, dtype='float32').reshape(-1, 28, 1)
    #prob_positive = TF_model.predict(input_X)[0][0]
    prob_positive = model.predict(input_X)[0, 1]
    y_pred_prob.append(prob_positive)

fpr, tpr, thresholds = evalTF = roc_curve(y_test, y_pred_prob, pos_label=1)

with open(save_path+"config.txt", "w") as f:
    f.write("\t".join(["fpr", "tpr", "thresholds"])+"\n")

    for eval in evalRF, evalXGB, evalTF:
        fpr, tpr, thresholds = eval
        def obtain_t ():
            fpr_max = max(fpr)
            for f, t, th in zip(fpr, tpr, thresholds):
                if t >= max(tpr) and f<=fpr_max :

                    fpr_max = f

                    f_return = f
                    t_return = t
                    th_return = th
                    
            return [f_return,t_return,th_return]
                
        print("Config file test")
        line = obtain_t()
        print(line)

        f.write("\t".join([str(e) for e in line])+"\n")
