import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, PrecisionRecallDisplay

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator # THIS ONE IS DEPRECATED, BUT I DIDN'T BOTHER TO DO IT THE NEW WAY
from keras.callbacks import ReduceLROnPlateau
import itertools
from keras import backend as K



def plot_model_train_history(model_history, metric="f1_score"):
    '''
    Function to plot the accuracy vs epoch.
    '''
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1,2,figsize=(16,4))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history[metric])+1),model_history['f1_score'])
    axs[0].plot(range(1,len(model_history[f'val_{metric}'])+1),model_history[f'val_{metric}'])
    axs[0].set_title(f'Model {metric.capitalize()}')
    axs[0].set_ylabel(metric.capitalize())
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history[metric])+1))
    axs[0].legend(['Train', 'Valid.'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history['loss'])+1),model_history['loss'])
    axs[1].plot(range(1,len(model_history['val_loss'])+1),model_history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history['loss'])+1))
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()



def EDA_plots(data, variable, kind= "count", hue= None, title: str = "", colors = "Paired", hist_args: dict = {}):

    sns.set_style('whitegrid')
    fig,axes = plt.subplots(figsize=(12,8))

    hist_default = {"kde": False, "discrete": False, "bins": 10, "multiple": "layer"}
    hist_default.update(hist_args)

    if kind == "count":
        ax = sns.countplot(data = data, x = variable, hue=hue, order = data[variable].value_counts().index, palette = colors)
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=45)

    elif kind == "hist":
        ax = sns.histplot(data=data, x=variable, hue=hue, palette=colors, kde=hist_default["kde"], discrete=hist_default["discrete"], bins=hist_default["bins"], multiple=hist_default["multiple"])

    plt.ylabel("Count", weight="bold")
    plt.xlabel(" ".join(variable.split("_")).capitalize(), weight="bold")
    plt.title(title, weight="bold")
    
    plt.show()


def build_CNN(input_shape = (75, 100, 3),  num_classes = 7, metrics=["accuracy"]):

    # Build the CNN model 
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Define the optimizer
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False) 

    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=metrics)


    return model


def evaluate_model(model, X_test, y_test, threshold=None):

    # Predict the values from the validation dataset
    y_pred = model.predict(X_test)
    # Convert predictions classes to one hot vectors
    if threshold == None:
        y_pred_labels = np.argmax(y_pred, axis = 1)
    else:
        y_pred_labels = [1 if prob >= threshold else 0 for prob in np.ravel(y_pred[:,1])]
    # Convert validation observations to one hot vectors
    y_true = np.argmax(y_test, axis = 1) 

    print(f"Accuracy : {accuracy_score(y_true, y_pred_labels):.2f}")
    print(f"Precision : {precision_score(y_true, y_pred_labels):.2f}")
    print(f"Recall : {recall_score(y_true, y_pred_labels):.2f}")

    # Compute the confusion matrix
    CM = confusion_matrix(y_true, y_pred_labels)

    # Create a figure
    sns.set_style("dark")
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(CM)
    disp.plot(cmap="GnBu", ax=ax[0])

    # Plot the Precision - Recall Curve and Confusion Matrix
    PrecisionRecallDisplay.from_predictions(y_true, y_pred[:,1], name=None, plot_chance_level=True, ax=ax[1])
    
    ax[0].set_title("Confusion Matrix", weight="bold")
    ax[1].set_title("Precision-Recall curve", weight="bold")
    
    plt.show()

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


def convert_history(history):
    history_dict = {}
    for key, values in history.items():
        history_dict[key] = [float(value) for value in values]
    return history_dict