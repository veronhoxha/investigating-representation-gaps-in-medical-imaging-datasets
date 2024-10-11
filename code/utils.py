import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras import backend as K
from keras.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator # THIS ONE IS DEPRECATED, BUT I DIDN'T BOTHER TO DO IT THE NEW WAY
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score, PrecisionRecallDisplay

from PIL import Image
from skimage.transform import resize


# color palette for the plots
color_palette = sns.color_palette("deep", 6)


def plot_model_train_history(model_history, metric="f1_score", colors=color_palette):
    '''
    Function to plot the accuracy vs epoch.
    '''
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history[metric]) + 1), model_history[metric], color=colors[0])
    axs[0].plot(range(1, len(model_history[f'val_{metric}']) + 1), model_history[f'val_{metric}'], color=colors[1])
    axs[0].set_title(f'Model {metric.capitalize()}', weight='bold')
    axs[0].set_ylabel(metric.capitalize(), weight='bold')
    axs[0].set_xlabel('Epoch', weight='bold')
    axs[0].set_xticks(np.arange(1, len(model_history[metric]) + 1, 2))
    axs[0].set_xticklabels(np.arange(1, len(model_history[metric]) + 1, 2), rotation=45) 
    axs[0].legend(['Train', 'Validation'], loc='best')
    
    # summarize history for loss
    axs[1].plot(range(1, len(model_history['loss']) + 1), model_history['loss'], color=colors[2])
    axs[1].plot(range(1, len(model_history['val_loss']) + 1), model_history['val_loss'], color=colors[3])
    axs[1].set_title('Model Loss', weight='bold')
    axs[1].set_ylabel('Loss', weight='bold')
    axs[1].set_xlabel('Epoch', weight='bold')
    axs[1].set_xticks(np.arange(1, len(model_history['loss']) + 1, 2))
    axs[1].set_xticklabels(np.arange(1, len(model_history['loss']) + 1, 2), rotation=45) 
    axs[1].legend(['Train', 'Validation'], loc='best')
    
    plt.tight_layout()
    plt.show()


def EDA_plots(data, variable, kind= "count", hue= None, title: str = "", colors = color_palette, hist_args: dict = {}):

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
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, weight_decay=0.0, amsgrad=False) 

    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=metrics)


    return model


def evaluate_model(model, X_test, y_test, threshold=None, title=None):
    '''
    Function to evaluate a model given X and y. Takes an alternative threshold (if None, the threshold is 0.5) and a title for plots.

    Computes metrics of interest and plots Confusion Matrix and Precision-Recall curves.
    '''
    # Predict the values from the validation dataset
    y_pred = model.predict(X_test)
    
    # Convert predictions to class labels
    if threshold is None:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = [1 if prob > threshold else 0 for prob in np.ravel(y_pred[:, 1])]
    
    # Convert true labels to class labels
    y_true = np.argmax(y_test, axis=1)
    
    # Print evaluation metrics
    print(f"<<<<<<<<<< MODEL EVALUATION | Threshold = {threshold} >>>>>>>>>>\n")
    print(classification_report(y_true, y_pred_labels, target_names=["Non-Cancer", "Cancer"]))
    print(f"MCC: {matthews_corrcoef(y_true, y_pred_labels):.2f}")

    # Compute the confusion matrix
    CM = confusion_matrix(y_true, y_pred_labels)
    
    # Create a figure
    sns.set_style("dark")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(CM)
    # disp.plot(cmap="GnBu", ax=ax[0])
    disp.plot(cmap="Blues", ax=ax[0])
    
    # Plot the Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_true, y_pred[:, 1], name=None, plot_chance_level=True, ax=ax[1], color=color_palette[0])
    
    if title:
        fig.suptitle(title, weight="bold", fontsize=16, y=1.05)
    ax[0].set_title("Confusion Matrix", weight="bold")
    ax[1].set_title("Precision-Recall Curve", weight="bold")
    
    plt.subplots_adjust(top=0.85)
    
    plt.show()


def reshape_image(img, shape, mode="empty", padding_constant=0):
    '''
    Reshape an image keeping the ratio intact.
    '''
    # Extract height, width of original image
    h, w, c = img.shape

    # Calculate ratio
    ratio = h/w

    # Extract output (new) height width and ratio
    nh, nw = shape
    nratio = nh/nw

    # If ratios match, just resize
    if ratio == nratio:

        new_img = resize(img, shape, order=None, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)

    # If original ratio is greater than new ratio
    # it means image is vertical
    elif ratio > nratio:

        new_img = resize(img, (shape[0], shape[0] * ratio), order=None, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)
        if mode == "constant":
            new_img = np.pad(new_img, ((0,0), (np.ceil((shape[1] - shape[0] * ratio)/2).astype(int), np.floor((shape[1] - shape[0] * ratio)/2).astype(int)), (0, 0)), mode=mode, constant_values=padding_constant)
        else:
            new_img = np.pad(new_img, ((0,0), (np.ceil((shape[1] - shape[0] * ratio)/2).astype(int), np.floor((shape[1] - shape[0] * ratio)/2).astype(int)), (0, 0)), mode=mode)

    else:

        new_img = resize(img, (shape[1] * ratio, shape[1]), order=None, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)
        if mode == "constant":
            new_img = np.pad(new_img, ((np.ceil((shape[0] - shape[1] * ratio)/2).astype(int), np.floor((shape[0] - shape[1] * ratio)/2).astype(int)), (0,0), (0, 0)), mode='constant', constant_values=padding_constant)
        else:
            new_img = np.pad(new_img, ((0,0), (np.ceil((shape[1] - shape[0] * ratio)/2).astype(int), np.floor((shape[1] - shape[0] * ratio)/2).astype(int)), (0, 0)), mode=mode)
        
    new_img = new_img[:,:,:3]

    # Final reshape for uneven padding
    new_img = resize(new_img, (75, 100), order=None, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)

    return new_img


## ALTERNATIVE
def resize_with_padding(image, target_size, fill_color=0):
    
    # Check if fill_color is a constant or tuple
    if type(fill_color) != tuple:
        try:
            c = int(fill_color)
        except:
            raise TypeError("fill_color must be an integer, or tuple of integers.")
        fill_color = (c, c, c) 

    # Create a new blank image with the desired target size and fill color
    padded_image = Image.new("RGB", target_size, fill_color)
    
    # Calculate the resizing ratio while preserving the aspect ratio
    width_ratio = target_size[0] / image.width
    height_ratio = target_size[1] / image.height
    ratio = min(width_ratio, height_ratio)
    
    # Calculate the new size after resizing
    new_size = (int(image.width * ratio), int(image.height * ratio))
    
    # Resize the original image while maintaining the aspect ratio
    resized_image = image.resize(new_size, Image.LANCZOS)
    
    # Calculate the position to paste the resized image
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    
    # Paste the resized image onto the blank padded image
    padded_image.paste(resized_image, paste_position)
    
    return padded_image



#####################################################################################
#Assignment 1 functions 



class FairnessReport:
    '''
    Custom class to compute a 'Fairness Report'. The Fairness report includes a measure of Statistical Parity, Equalized Odds,
    and Equalized Outcomes from an array of TRUE labels, an array of PREDICTIONS, and the class split. If the probabilities are also 
    passed to the fitter, those are included in the results table.
    '''

    def __init__(self):
            pass
        

    def fit(self, y_true, y_pred, group, pred_prob=None, index=None):
        '''
        Fitter of the class. It basically constructs the table needed for computing the different fairness metrics.

        Arguments:
        - y_true: Array of TRUE labels
        - y_pred: Array of PREDICTED labels
        - group: Partititions for the observations
        - pred_prob: The probabilities from the classifier.
        - index: Index of the observations. If passed, the index from the observations is kept, so easy comparisons can be made.
        '''

        # Create a DF with the each prediction, including group, prediction (selected) and target (true label)
        self.results_table = pd.DataFrame()
        self.results_table["group"] = group
        self.results_table["target"] = y_true
        self.results_table["selected"] = y_pred
        if type(pred_prob) != None:
            self.results_table["pred_prob"] = pred_prob[:,1]
        if type(index) != None:
            try:
                self.results_table.index = index
            except:
                print("Could not include index. Omitted.")
        
        return
        

    def compute(self):
        '''
        Compute the Fairness metrics, and show group-wise Confusion Matrices along with computed Statistical Parity, Equalized Odds and Equalized Outcomes.
        '''

        # Extract TPR and FPR per group
        rates = {x: {} for x in self.results_table.group.unique()}
        
        # print(rates)

        # Colormaps
        colors = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.BuPu]

        # Plot confusion matrices for context
        fig, ax = plt.subplots(len(self.results_table.group.unique()), 2, figsize=(12,12), dpi=300)

        # Populate the subplots
        for ix, group in enumerate(self.results_table.group.unique()):
            # Compute the confusion matrix
            cm = confusion_matrix(self.results_table.loc[self.results_table.group == group, "target"], self.results_table.loc[self.results_table.group == group, "selected"],
                                                    normalize=None)
            # Compute the row-wise normalized matrix
            cm_norm = confusion_matrix(self.results_table.loc[self.results_table.group == group, "target"], self.results_table.loc[self.results_table.group == group, "selected"],
                                                    normalize="true")
            # Extract True Positive Rate (TPR) and False Positive Rate (FPR) from the Confusion Matrix
            rates[group]["TPR"] = cm_norm[1,1]
            rates[group]["FPR"] = cm_norm[0,1]

            # Display both CMs
            disp1 = ConfusionMatrixDisplay(cm)
            disp2 = ConfusionMatrixDisplay(cm_norm)
            cmap = plt.get_cmap(colors[ix])
            disp1.plot(cmap= cmap, ax=ax[ix, 0], values_format='g')
            disp2.plot(cmap= cmap, ax=ax[ix, 1])
            ax[ix,0].set_title(f'Group {group}')
            ax[ix,1].set_title(f'Group {group} - Row Norm.')

        
        for a in ax[:, 0]:
            a.set_xlabel("")
        for a in ax[:, 1]:
            a.set_xlabel("")
        for a in ax[0, 1:]:
            a.set_ylabel("")

        fig.suptitle("Confusion Matrices per Group", fontweight='bold')
        plt.show()

        # Statistical parity: Prob. of being selected given the group
        ## Per each group, we look at the number of TRUE Predictions and divide by the total number of people in that group
        stat_parity = {}
        for group in self.results_table.group.unique():
            group_total = len(self.results_table.loc[self.results_table.group == group])
            if group_total > 0:
                selected = len(self.results_table.loc[(self.results_table.group == group) & (self.results_table.selected)])
                prob_selected = selected / group_total
                stat_parity[group] = prob_selected
            else:
                stat_parity[group] = None

        if all(value is not None for value in stat_parity.values()):
            sp_values = list(stat_parity.values())
            sp_result = all(sp_values[0] == sp_value for sp_value in sp_values)
            sp_message = f"Statistical Parity: {sp_result} -> " + " & ".join([f"P(s=1 | G={group}): {sp_values[ix]:.3f}" for ix, group in enumerate(stat_parity)])
            print(sp_message)
        else:
            print("Cannot compute statistical parity due to missing group data.")

        # Equalized odds: both TPR and FPR are equal for all groups
        if len(rates) > 1:
            unique_groups = list(rates.keys())
            eq_odds = all(rates[unique_groups[0]]["TPR"] == rates[group]["TPR"] and rates[unique_groups[0]]["FPR"] == rates[group]["FPR"] for group in unique_groups[1:])
            eq_message = f'Equalized Odds: {eq_odds} -> ' + " & ".join([f'TPR Group {group}: {rates[group]["TPR"]:.3f} & FPR Group {group}: {rates[group]["FPR"]:.3f}' for group in unique_groups])
            print(eq_message)
        else:
            print("Not enough groups to compare for equalized odds.")

        # Equalized outcomes: Given the prediction, trues are independent of the group
        eq_outcomes = {}
        for group in self.results_table.group.unique():
            group_total = len(self.results_table.loc[self.results_table.group == group])
            if group_total > 0:
                pred_pos = len(self.results_table.loc[(self.results_table.group == group) & (self.results_table.target) & (self.results_table.selected)])
                pred_pos_prob = pred_pos / len(self.results_table.loc[(self.results_table.group == group) & (self.results_table.selected)])
                pred_neg = len(self.results_table.loc[(self.results_table.group == group) & (self.results_table.target) & (~self.results_table.selected)])
                pred_neg_prob = pred_neg / len(self.results_table.loc[(self.results_table.group == group) & (~self.results_table.selected)])
                eq_outcomes[group] = (pred_pos_prob, pred_neg_prob)
            else:
                eq_outcomes[group] = (None, None)

        if all(v[0] is not None and v[1] is not None for v in eq_outcomes.values()):
            eo_values = list(eq_outcomes.values())
            eo_result = all(eo_values[0] == eo_value for eo_value in eo_values)
            eo_message = f"Equalized Outcomes: {eo_result} -> " + " & ".join([f'P(T=1 | G={group}, S=1): {eo_values[ix][0]:.3f} & P(T=1 | G={group}, S=0): {eo_values[ix][1]:.3f}' for ix, group in enumerate(eq_outcomes)])
            print(eo_message)
        else:
            print("Cannot compute equalized outcomes due to missing group data.")

        return


    def get_results_table(self):
        '''
        Return the results table for inspection.
        '''
        return self.results_table


    def plot_roc_curves(self, highlight={}, save=False):
        '''
        Plot the group-wise ROC curves.
        '''

        if "pred_prob" not in self.results_table.columns:
            raise Exception("The predicted probabilities were not passed to the fitter.")
        
        # Plot ROC curve
        plt.figure(figsize=(8,8), dpi=300)
        

        # Color list
        colormap = ["darkblue", "darkred", "darkcyan", "darkmagenta", "darksalmon", "darkkhari"]

        # Predict probabilities for the test set, group-wise
        for ix, group in enumerate(self.results_table["group"].unique()):
            group_subset = self.results_table[self.results_table["group"] == group]
            # Compute ROC curve and ROC area for each group
            fpr, tpr, th = roc_curve(group_subset.target, group_subset.pred_prob)
            roc_auc = roc_auc_score(group_subset.target, group_subset.pred_prob)
            # Plot the ROC curve
            plt.plot(fpr, tpr, color=colormap[ix], lw=2, label=f'ROC curve (Group {group})\n(area = {roc_auc:.2f})')
            #plt.plot(fpr, tpr, color=group_color_map[group], lw=2, label=f'ROC curve (Group {group})\n(area = {roc_auc:.2f})')
            # If a threshold should be highlighted
            if group in highlight:
                # Get the threshold point
                t = highlight[group]
                # Find the closest threshold to the requested point
                idx = (np.abs(th - t)).argmin()
                
                plt.plot(fpr[idx], tpr[idx], marker='X', markersize=10, color=colormap[ix])
                plt.vlines(fpr[idx], 0, tpr[idx], colors=colormap[ix], linestyles='dashed', label='', alpha=0.5)
                plt.text(fpr[idx], tpr[idx]/2, f"{fpr[idx]:.2f}", color="gray", rotation="vertical")
                plt.hlines(tpr[idx], 0, fpr[idx], colors=colormap[ix], linestyles='dashed', label='', alpha=0.5)
                plt.text(fpr[idx]/2, tpr[idx], f"{tpr[idx]:.2f}", color="gray", rotation="horizontal")   
                #plt.plot(fpr[idx], tpr[idx], marker='X', markersize=10, color=group_color_map[group])
                #plt.vlines(fpr[idx], 0, tpr[idx], colors=group_color_map[group], linestyles='dashed', label='', alpha=0.5)
                #plt.text(fpr[idx], tpr[idx] / 2, f"{fpr[idx]:.2f}", color="gray", rotation="vertical")
                #plt.hlines(tpr[idx], 0, fpr[idx], colors=group_color_map[group], linestyles='dashed', label='', alpha=0.5)
                #plt.text(fpr[idx] / 2, tpr[idx], f"{tpr[idx]:.2f}", color="gray", rotation="horizontal")
                
        # Plot the straight line
        plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
        
        # Format the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC)', fontweight='bold', fontsize=16)
        plt.legend(loc="lower right")
        
        if save:
            plt.savefig(save)
        plt.show();

        return