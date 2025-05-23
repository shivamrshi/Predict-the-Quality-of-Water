
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# I have compatibility problems with some packages
# So I will install sklearn to try to fix them
# pip install sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# pip install feature_engine
from feature_engine.outliers import Winsorizer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import RandomSampleImputer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# pip install imblearn
from imblearn.under_sampling import RandomUnderSampler

from scipy.stats import kurtosis, skew

import itertools

from warnings import simplefilter
simplefilter("ignore")


# 1. Read the Data

data = pd.read_csv('water_potability.csv')

data.head()

print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns.')
print(f'The dataset has {data.isna().sum().sum()} null values.')

data_nulls = (data.apply(lambda x: x.isnull().value_counts()).T[True]/len(data)*100).reset_index(name='count')


fig = plt.figure(figsize=(12,6))

fig = sns.barplot(data_nulls, x="index", y="count")
fig.set_title('Null Values in the Data', fontsize=30)
fig.set_xlabel('features', fontsize=12)
fig.set_xticklabels(fig.get_xticklabels(), rotation=45)
fig.set_ylabel('% of null values', fontsize=12)
fig.bar_label(fig.containers[0], fmt='%.1f')

plt.tight_layout()
print(f'There are {data.duplicated().sum()} duplicate rows in the data.')
data.dtypes


# 2. Exploratory Data Analysis
#I am plotting the histograms of the numerical variables. Then, I am also calculating their skewness and kurtosis.

cols = data.columns


for i in range (4):

    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(12,6))

    ax1 = sns.histplot(data[cols[i*3]], ax=ax1)
    ax1.set_title(f"Histogram of '{cols[i*3]}'", size=16)

    if i < 3:
        ax2 = sns.histplot(data[cols[i*3+1]], ax=ax2)
        ax2.set_title(f"Histogram of '{cols[i*3+1]}'", size=16)
        ax3 = sns.histplot(data[cols[i*3+2]], ax=ax3)
        ax3.set_title(f"Histogram of '{cols[i*3+2]}'", size=16)

    plt.tight_layout()

    predictors = cols.copy().tolist()
predictors.remove('Potability')

for col in predictors:
    print(f"Skewness of '{col}: {skew(data[~data[col].isna()][col])}")
    print(f"Kurtosis of '{col}: {kurtosis(data[~data[col].isna()][col])}")
    print()

# As expected from the histograms, some of the distributions (like Turbidity) are very close to perfect Gaussians. This is also reflected in the values of skewness and kurtosis (that are almost zero).
#It is also worth to note that the classes in the target variable, Potability, are inbalanced but their level of imbalance is not dramatic.
#Now, let's have a look at the outliers.
for i in range (3):

    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(12,6))

    ax1 = sns.boxplot(data[cols[i*3]], ax=ax1)
    ax1.set_title(f"Boxplot of '{cols[i*3]}'", size=16)

    ax2 = sns.boxplot(data[cols[i*3+1]], ax=ax2)
    ax2.set_title(f"Boxplot of '{cols[i*3+1]}'", size=16)

    ax3 = sns.boxplot(data[cols[i*3+2]], ax=ax3)
    ax3.set_title(f"Boxplot of '{cols[i*3+2]}'", size=16)

    plt.tight_layout() 

    outliers_perc = []

print('### Percentage of outliers in the columns ###')
print()


def outliers_perc_search(data, cols):
    for k,v in data[cols].items():
        # Column must be of numeric type (not object)
        if data[k].dtype != 'O':
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
            out_tuple = (k,int(perc))
            outliers_perc.append(out_tuple)
            print("Column %s outliers = %.1f%%" % (k,perc))


outliers_perc_search(data, predictors)

### Percentage of outliers in the columns ###

''' Column ph outliers = 1.4%
Column Hardness outliers = 2.5%
Column Solids outliers = 1.4%
Column Chloramines outliers = 1.9%
Column Sulfate outliers = 1.3%
Column Conductivity outliers = 0.3%
Column Organic_carbon outliers = 0.8%
Column Trihalomethanes outliers = 1.0%
Column Turbidity outliers = 0.6%
There is a limited number of outliers in the predictor columns. Capping them is one of the possibilities.
The other is to use classification algorithms that are not affected by outliers (like tree-based models) and leave everything as it is.
Now, let's have a look at the correlation heatmap and pairplot of the variables.
'''
sns.pairplot(data)

'''There is no collinearity between the variables. 
Even though this is an assumption of the linear regression model, collinearity (as well as strong correlations between the variables) may cause the model to overfit.'''

fig = plt.figure(figsize=(12,10))

fig = sns.heatmap(data.corr(), annot=True, cmap='Blues')
fig.set_title('Correlation Heatmap of the Variables', size=30)

plt.tight_layout()

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100)

X = data.drop('Potability', axis=1)
y = data['Potability']

rf.fit(X, y)

# To sort the index in descending order, I multiply 'rf.feature_importances_' by -1
sorted_idx = (-rf.feature_importances_).argsort()

list_of_tuples = list(zip(X.columns[sorted_idx],
                          rf.feature_importances_[sorted_idx]))

feat_importance = pd.DataFrame(list_of_tuples,
                  columns=['feature', 'feature importance'])

##################

fig = plt.figure(figsize=(12,8))

fig = sns.barplot(data=feat_importance, x='feature', y='feature importance')
plt.title('Feature Importance',fontsize=25)
plt.xticks(fontsize=8,rotation=60)

plt.tight_layout()

perm_importance = permutation_importance(rf, X, y)

sorted_idx = (-perm_importance.importances_mean).argsort()

list_of_tuples  = list(zip(X.columns[sorted_idx],
                           perm_importance.importances_mean[sorted_idx]))

perm_importance = pd.DataFrame(list_of_tuples,
                  columns=['feature','permutation importance'])

print(perm_importance.head())

plt.figure(figsize=(12,8))

sns.barplot(perm_importance[perm_importance['permutation importance'] > 0.0005], x='feature', y='permutation importance')

plt.title('Permutation-Based Importances > 0.0005', fontsize=25)
plt.xlabel('feature', fontsize=15)
plt.xticks(fontsize=8, rotation=45)
plt.ylabel('permutation importance', fontsize=15)
    
plt.tight_layout()
#This second plot describes a hierarchy of importance, where ph and Sulfate are at the top.

# 3. Feature Engineering

'''3.1 Imputation of the Null Values
There are several ways to impute the null values. Here, I will show some of them. For more setails, see Ref. 2.

3.1.1 Imputing the Nulls with the Mean'''
X1 = X.copy()

imputer1 = Pipeline([
    ("mean_imputer", MeanMedianImputer(imputation_method="mean"))
])

imputer1.fit(X1)

X1 = imputer1.transform(X1)

X1.head()

null_cols = ['ph','Sulfate','Trihalomethanes']


def plot_nulls(X, X1):

    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(12,6))

    ax1 = sns.kdeplot(X[null_cols[0]], linewidth=4, label='original train data', ax=ax1) 
    ax1 = sns.kdeplot(X1[null_cols[0]], linewidth=4, linestyle='--', label='imputed train data', ax=ax1) 
    ax1.set_title(f"Distribution of '{null_cols[0]}'", size=16)
    ax1.legend()

    ax2 = sns.kdeplot(X[null_cols[1]], linewidth=4, label='original train data', ax=ax2) 
    ax2 = sns.kdeplot(X1[null_cols[1]], linewidth=4, linestyle='--', label='imputed train data', ax=ax2) 
    ax2.set_title(f"Distribution of '{null_cols[1]}'", size=16)
    ax2.legend()

    ax3 = sns.kdeplot(X[null_cols[2]], linewidth=4, label='original train data', ax=ax3) 
    ax3 = sns.kdeplot(X1[null_cols[2]], linewidth=4, linestyle='--', label='imputed train data', ax=ax3) 
    ax3.set_title(f"Distribution of '{null_cols[2]}'", size=16)
    ax3.legend()

    plt.tight_layout()


plot_nulls(X, X1)  
#The mean imputation method substitutes the nulls with new values right on the mean. 
#The result is to increase the kurtosis of the distributions and to make their central peaks higher and narrower.


# 3.1.2 Random Imputation
X2 = X.copy()

imputer2 = ColumnTransformer(
    transformers=[
        ("random_imputer", RandomSampleImputer(random_state=42), predictors)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
).set_output(transform="pandas")

imputer2.fit_transform(X2)
plot_nulls(X, X2) 
# Imputing at random gives rise to distributions that are very similar to the original distributions (those with the nulls).
 
# 3.1.3 Imputing with KNN
X3 = X.copy()

imputer3 = ColumnTransformer(
    transformers=[
        ("KNN_imputer", KNNImputer(n_neighbors=5,weights='distance',metric='nan_euclidean',add_indicator=False), predictors)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
).set_output(transform="pandas")

imputer3.fit_transform(X3)

plot_nulls(X, X3)

# 3.2 Pipelines for Capping the Outliers and Scaling
# I am writing down a pipeline that preprocess the data by imputing the null values, capping the outliers and scaling; for more details, see Ref. 2. 

X4 = X.copy()

preprocessor1 = Pipeline([
    ("mean_imputer", MeanMedianImputer(imputation_method="mean")),
    ('outliers_capping', Winsorizer(variables=predictors, capping_method="iqr", tail="both", fold=1.5)),
    ('scaler', StandardScaler())
    ]).set_output(transform="pandas")

X4 = preprocessor1.fit_transform(X4)

X4.head()

# 4. Classification with a Gradient Boosting Classifier
# In this section I will classify the water potability by using three different classification pipelines. Each pipeline imputes the null values (by means of one of the three methods previously described), it caps the outliers, it scales the data and then it performs the classification. I will compare the accuracies obtained with the three different pipelines.

#4.1 First Pipeline
#First, I will perform train-test split.

X = data.drop('Potability', axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
((2293, 9), (983, 9), (2293,), (983,))
pipe1 = Pipeline([
    ('preprocessor1', preprocessor1),
    ('rf_classifier', GradientBoostingClassifier(random_state=42))
])

pipe1.fit(X_train, y_train)

print(f'Train accuracy: {pipe1.score(X_train, y_train):.3f}')
print(f'Test accuracy: {pipe1.score(X_test, y_test):.3f}')

y_pred1 = pipe1.predict(X_test)

print(classification_report(y_test, y_pred1))


# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, y_pred1)

ax = sns.heatmap(cm, annot=True)
ax.set_title('Confusion Matrix (Mean Imputation)', fontsize=15)
ax.xaxis.set_ticklabels(['Not Potable','Potable']) 
ax.yaxis.set_ticklabels(['Not Potable','Potable']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()

# 4.2 Second Pipeline
preprocessor2 = Pipeline([
    ("random_imputer", RandomSampleImputer(random_state=42)),
    ('outliers_capping', Winsorizer(variables=predictors, capping_method="iqr", tail="both", fold=1.5)),
    ('scaler', StandardScaler())
    ]).set_output(transform="pandas")


pipe2 = Pipeline([
    ('preprocessor2', preprocessor2),
    ('rf_classifier', GradientBoostingClassifier(random_state=42))
])

pipe2.fit(X_train, y_train)

print(f'Train accuracy: {pipe2.score(X_train, y_train):.3f}')
print(f'Test accuracy: {pipe2.score(X_test, y_test):.3f}')

y_pred2 = pipe2.predict(X_test)

print(classification_report(y_test, y_pred2))


# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, y_pred2)

ax = sns.heatmap(cm, annot=True)
ax.set_title('Confusion Matrix (Random Imputation)', fontsize=15)
ax.xaxis.set_ticklabels(['Not Potable','Potable']) 
ax.yaxis.set_ticklabels(['Not Potable','Potable']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()

# 4.3 Third Pipeline¶
preprocessor3 = Pipeline([
    ("KNN_imputer", KNNImputer(n_neighbors=5,weights='distance',metric='nan_euclidean',add_indicator=False)),
    ('outliers_capping', Winsorizer(variables=predictors, capping_method="iqr", tail="both", fold=1.5)),
    ('scaler', StandardScaler())
    ]).set_output(transform="pandas")


pipe3 = Pipeline([
    ('preprocessor3', preprocessor3),
    ('rf_classifier', GradientBoostingClassifier(random_state=42))
])

pipe3.fit(X_train, y_train)

print(f'Train accuracy: {pipe3.score(X_train, y_train):.3f}')
print(f'Test accuracy: {pipe3.score(X_test, y_test):.3f}')

y_pred3 = pipe3.predict(X_test)

print(classification_report(y_test, y_pred3))

# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, y_pred3)

ax = sns.heatmap(cm, annot=True)
ax.set_title('Confusion Matrix (KNN Imputation)', fontsize=15)
ax.xaxis.set_ticklabels(['Not Potable','Potable']) 
ax.yaxis.set_ticklabels(['Not Potable','Potable']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()

T# he second pipeline provides the best results. However, even in this case the results are not the best because the model underperforms on the minority class. The data is imbalanced and, to solve this issue, sampling is needed.

# 4.4 Second Pipeline and Sampling
# I am undersampling the train data. For more details, see Ref. 3.

# Create a RandomUnderSampler object
rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')

# Balancing the data
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
pipe2.fit(X_resampled, y_resampled)

print(f'Train accuracy: {pipe2.score(X_resampled, y_resampled):.3f}')
print(f'Test accuracy: {pipe2.score(X_test, y_test):.3f}')

y_pred2 = pipe2.predict(X_test)

print(classification_report(y_test, y_pred2))


# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, y_pred2)

ax = sns.heatmap(cm, annot=True)
ax.set_title('Confusion Matrix (Random Imputation + Undersampling)', fontsize=12)
ax.xaxis.set_ticklabels(['Not Potable','Potable']) 
ax.yaxis.set_ticklabels(['Not Potable','Potable']) 
ax.set_xlabel("Predicted")
ax.set_ylabel("Target")

plt.tight_layout()

# 5. Classification with PyTorch

'''In this last section, I am trying to classify the data by means of an NN model in Pytorch.

5.1 First Model. Classification with Imbalanced Data
I will start with the imbalanced data case. The first steps are to:

Preprocess the X variables and transform them into tensors.
Creating train and test dataloaders, with given batch size.
I will use the function below to do this.'''

def return_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    
    # Preprocess X test and train    
    X_train = preprocessor2.fit_transform(X_train)
    X_test  = preprocessor2.transform(X_test)

    # Turn the Xs and ys into tensors
    X_train_tens = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tens = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    
    X_test_tens = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tens = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)    

    # Create train and test datasets and dataloaders
    y_train_tens = y_train_tens.unsqueeze(1)
    train_ds = TensorDataset(X_train_tens, y_train_tens)
    train_dl = DataLoader(train_ds, batch_size=batch_size)

    y_test_tens = y_test_tens.unsqueeze(1)
    test_ds = TensorDataset(X_test_tens, y_test_tens)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Calculate the input dimensions of the neural network
    n_input_dim = X_train_tens.shape[1]

    return train_dl, test_dl, n_input_dim


train_dl, test_dl, n_input_dim = return_loaders(X_train, y_train, X_test, y_test)
# Then, I am building my first classification model. It will be a very simple one.

# Layer size
n_hidden = 10  # Number of hidden nodes
n_output =  1  # Number of output nodes

class Model1(nn.Module):
    
    def __init__(self):
        super(Model1, self).__init__()
        self.layer = nn.Linear(n_input_dim, n_hidden) 
        self.layer_out = nn.Linear(n_hidden, n_output) 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.relu(self.layer(inputs))
        x = self.sigmoid(self.layer_out(x))
        
        return x
    

model1 = Model1()
print(model1)
Model1(
  (layer): Linear(in_features=9, out_features=10, bias=True)
  (layer_out): Linear(in_features=10, out_features=1, bias=True)
  (relu): ReLU()
  (sigmoid): Sigmoid()
)
# Then, I am introducing a function to train the model and print the results.

def train_model(model, train_dl, test_dl, learning_rate=0.001, epochs=500):

    #### Loss Function ####
    loss_func = nn.BCELoss()

    #### Optimizer ####
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #### Train the model ####
    loss_hist_train = [0] * epochs
    loss_hist_test  = [0] * epochs

    for epoch in range(epochs):

        train_loss = 0.
        test_loss  = 0.
        
        # Loss on the train set
        # Set model to train mode
        model.train()
        for xb, yb in train_dl:
            y_pred = model(xb).unsqueeze(1)    # Forward Propagation
            train_loss = loss_func(y_pred, yb) # Loss Computation
            optimizer.zero_grad()              # Clearing all previous gradients, setting to zero 
            train_loss.backward()              # Back Propagation
            optimizer.step()                # Updating the parameters 
            loss_hist_train[epoch] += train_loss.item() * yb.size(0)
        loss_hist_train[epoch] /= len(train_dl.dataset)

        # Loss on the test set
        model.eval()
        with torch.no_grad():
            for xbt, ybt in test_dl:
                y_pred_test = model(xbt).unsqueeze(1)
                test_loss = loss_func(y_pred_test, ybt)
                loss_hist_test[epoch] += test_loss.item() * ybt.size(0)
            loss_hist_test[epoch] /= len(test_dl.dataset)

        # Output
        if epoch % 20 == 0:
            print(f'Epoch: {epoch}  Train Loss: {loss_hist_train[epoch]:.3f}  Test Loss: {loss_hist_test[epoch]:.3f}')

    #### Predicting ytest_pred ####
    y_pred_list = []
    model.eval()
    
    # Since we don't need model to back propagate the gradients in test set we use torch.no_grad()
    # It reduces memory usage and speeds up computation
    with torch.no_grad():
        for xb_test, yb_test in test_dl:
            y_test_pred = model(xb_test)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.detach().numpy())
    
    # Takes arrays and makes them list of list for each batch        
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]   
    # Flattens the lists in sequence
    ytest_pred = list(itertools.chain.from_iterable(y_pred_list))

    #### Plotting the train loss and the confusion matrix (test set) ####
    y_true_test = y_test.values.ravel()
    
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6))
    # Plotting the train and test loss 
    ax1 = sns.lineplot(loss_hist_train, label='train loss', ax=ax1)
    ax1 = sns.lineplot(loss_hist_test, label='test loss', ax=ax1)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train and Test Loss', size=20)    
    # Generate array of values for confusion matrix
    conf_matrix = confusion_matrix(y_true_test, ytest_pred)
    ax2 = sns.heatmap(conf_matrix, annot=True, ax=ax2)
    ax2.set_title('Confusion Matrix (Test Data)', fontsize=15)
    #plt.xaxis.set_ticklabels(['Not Potable','Potable']) 
    #plt.yaxis.set_ticklabels(['Not Potable','Potable']) 
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Target")   
    
    plt.tight_layout() 

    #### Calculating the accuracy on the test set ####
    print(f'#### Test accuracy = {accuracy_score(y_true_test, ytest_pred):.3f} ####')
train_model(model1, train_dl, test_dl, epochs=100)

#### Test accuracy = 0.671 ####

'''One can notice that:

The model overfits.
The model does not perform well on the minority target class.
5.2 First Model. Classification with Undersampled Data'''
train_dl, test_dl, n_input_dim = return_loaders(X_resampled, y_resampled, X_test, y_test)

model1 = Model1()

train_model(model1, train_dl, test_dl, epochs=100)

#### Test accuracy = 0.601 ####

# 5.3 Second Model
# Layer size
n_hidden_1 = 10 # Number of hidden nodes (first layer)
n_hidden_2 = 5  # Number of hidden nodes (second layer)
n_output   = 1  # Number of output nodes

class Model2(nn.Module):
    
    def __init__(self):
        super(Model2, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden_1)
        self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer_out = nn.Linear(n_hidden_2, n_output) 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
    

model2 = Model2()
print(model2)
Model2(
  (layer_1): Linear(in_features=9, out_features=10, bias=True)
  (layer_2): Linear(in_features=10, out_features=5, bias=True)
  (layer_out): Linear(in_features=5, out_features=1, bias=True)
  (relu): ReLU()
  (sigmoid): Sigmoid()
  (dropout): Dropout(p=0.1, inplace=False)
)
train_dl, test_dl, n_input_dim = return_loaders(X_train, y_train, X_test, y_test)

model2 = Model2()

train_model(model2, train_dl, test_dl, learning_rate=0.0001, epochs=300)

#### Test accuracy = 0.682 ####

# 5.4 Third Model
# Layer size
n_hidden_1 = 40 # Number of hidden nodes (first layer)
n_hidden_2 = 20 # Number of hidden nodes (second layer)
n_hidden_3 = 10  # Number of hidden nodes (third layer)
n_output   = 1  # Number of output nodes

class Model3(nn.Module):
    
    def __init__(self):
        super(Model3, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden_1)
        self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer_3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer_out = nn.Linear(n_hidden_3, n_output) 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
    

model3 = Model3()
print(model3)
Model3(
  (layer_1): Linear(in_features=9, out_features=40, bias=True)
  (layer_2): Linear(in_features=40, out_features=20, bias=True)
  (layer_3): Linear(in_features=20, out_features=10, bias=True)
  (layer_out): Linear(in_features=10, out_features=1, bias=True)
  (relu): ReLU()
  (sigmoid): Sigmoid()
  (dropout): Dropout(p=0.2, inplace=False)
)
train_dl, test_dl, n_input_dim = return_loaders(X_train, y_train, X_test, y_test)

model3 = Model3()

train_model(model3, train_dl, test_dl, learning_rate=0.0001, epochs=300)

#### Test accuracy = 0.690 ####

