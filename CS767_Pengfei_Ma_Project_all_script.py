#!/usr/bin/env python
# coding: utf-8

# # CS767_Pengfei_Ma_Project

# ## Data preparation
# ### I. Import data set

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


heart1 = pd.read_csv("heart.csv")

heart2 = pd.read_csv('processed_cleveland.data', sep=",", 
                     names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"])

heart3 = pd.read_csv('reprocessed_hungarian.data', sep=" ", 
                     names=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]).abs()


# In[5]:


heart1.head()


# In[6]:


heart2.head()


# In[7]:


heart3.head()


# ### II. Process the data set

# In[8]:


# 1. merge all three data set
heart_total = pd.concat([heart1, heart2, heart3], ignore_index=True)


# In[9]:


heart_total_array = heart_total.values # convert dataframe into numpy array


# In[10]:


X = heart_total_array[:,0:12] # all other columns except target column
Y = heart_total_array[:,13] # target column


# In[11]:


# 2. Scale the X set
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


# In[12]:


# 3. split train and test set in 70/30.
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3)


# ## Data visualization

# In[14]:


import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


# ### I. Ages and sex

# #### 1. Histogram of patient age counts

# In[13]:


female = heart_total.loc[heart_total['sex'] == 0]['age']
male = heart_total.loc[heart_total['sex'] == 1]['age']

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=female,
    histnorm='percent',
    name='Female', 
    marker_color='#EB89B5',
    opacity=0.75
))
fig.add_trace(go.Histogram(
    x=male,
    histnorm='percent',
    name='Male',
    marker_color='#330C73',
    opacity=0.75
))

fig.update_layout(
    title_text='Patient age counts', 
    xaxis_title_text='Age', 
    yaxis_title_text='Count', 
    bargap=0.2, 
    bargroupgap=0.1
)

fig.show()


# #### Histogram of patient age counts with heart attack risk comparsion

# In[14]:


female_all = heart_total.loc[heart_total['sex'] == 0]
female = female_all.loc[female_all['target'] != 0]['age']
male_all = heart_total.loc[heart_total['sex'] == 1]
male = male_all.loc[male_all['target'] != 0]['age']

fig = make_subplots(rows=1, cols=2, subplot_titles=('Female','Male'))

fig.add_trace(
    go.Histogram(
    x=female_all['age'],
    histnorm='percent',
    name='All people', 
    marker_color='#EB89B5',
    opacity=0.75
), row=1, col=1)

fig.add_trace(
    go.Histogram(
    x=male_all['age'],
    histnorm='percent',
    name='All people',
    marker_color='#EB89B5',
    opacity=0.75
), row=1, col=2)

fig.add_trace(
    go.Histogram(
    x=female,
    histnorm='percent',
    name='Female with heart attack risk', 
    marker_color='#330C73',
    opacity=0.75
), row=1, col=1)

fig.add_trace(
    go.Histogram(
    x=male,
    histnorm='percent',
    name='Male with heart attack risk',
    marker_color='#330C73',
    opacity=0.75
), row=1, col=2)

fig.update_layout(
    title_text='Patient age counts with heart attack risk comparsion', 
    xaxis_title_text='Age', 
    yaxis_title_text='Count', 
    bargap=0.2, 
    bargroupgap=0.1
)

fig.show()


# #### 3. Boxplot of ages

# In[15]:


trace0 = go.Box(y=heart_total['age'], name='Age for all people', marker_color = 'blue')
trace1 = go.Box(y=male_all['age'], name='Age for male', marker_color = 'red')
trace2 = go.Box(y=female_all['age'], name='Age for female', marker_color = 'green')

data = [trace0, trace1, trace2]
layout = go.Layout(title='Boxplots for ages and sex', 
                   xaxis_title="Age and sex",yaxis_title="Ages", hovermode='x')

fig = go.Figure(data=data, layout=layout)
fig.update_layout(legend_title_text='Ages and sex label')

fig.show()


# ### II. Correlation Heatmap

# In[16]:


fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
M0 = sns.heatmap(heart_total.corr(), annot = True).set_title('Heart Attack Features correlation')

plt.savefig("Heart Attack Features correlation heatmap.png")


# ## Multi-layers Perceptron

# ### I. Model Design

# In[16]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from functools import partial
from keras import models
from keras import layers
from sklearn.model_selection import cross_val_score


# In[18]:


def create_network(optimizer = 'rmsprop'):
    MaxNormDense = partial(keras.layers.Dense,
                           activation="selu", kernel_initializer="lecun_normal",
                           kernel_constraint=keras.constraints.max_norm(1.)
                           )

    model = models.Sequential()
    model.add(layers.Dense(200, activation='relu', input_shape=(12,)))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

neural_network = KerasClassifier(build_fn=create_network,verbose=0)


# ### II. Grid Search
# #### 1. Optimizers: : adam, nadam, sgd, rmsprop, adamax, adagrad
# #### 2. Epoch: 100, 200, 300

# In[18]:


from sklearn.model_selection import GridSearchCV


# In[21]:


epochs = [100, 200, 300]
optimizers = ['rmsprop', 'nadam', 'adam', 'sgd', 'adamax', 'adagrad']

# Create hyperparameter options
hyperparameters = dict(optimizer = optimizers, epochs=epochs)

# Create grid search
grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters, cv=3) 

# Fit gird search
grid_output = grid.fit(X_test, Y_test)
print(grid_output)
print(grid_output.best_params_)
print(grid_output.best_score_)


# ### III. Best model collection

# In[22]:


model = Sequential([
    Dense(200, activation='relu', input_shape=(12,)),
    Dense(200, activation='relu'),
    Dense(200, activation='relu'),
    Dense(200, activation='relu'),
    Dense(1, activation='sigmoid')
    ])


# In[23]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[24]:


hist = model.fit(X_train, Y_train, 
                 batch_size=32, 
                 epochs=100, 
                 validation_split=0.3)


# In[25]:


fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss','Accuracy'))

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['loss'],
                    mode='lines',marker_color='red',name='Train loss'), row=1, col=1)

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['val_loss'],
                    mode='lines',marker_color='blue',name='Val loss'), row=1, col=1)

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['accuracy'],
                    mode='lines',marker_color='orange',name='Train accuracy'), row=1, col=2)

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['val_accuracy'],
                    mode='lines',marker_color='purple',name='Val_accuracy'), row=1, col=2)

fig.update_layout(
    title_text='Loss and accuracy of the best model', 
    xaxis_title_text='Epoch', 
    bargap=0.2, 
    bargroupgap=0.1,
    hovermode='x'
)

fig.show()


# ### IV. Model evaluation

# In[26]:


model.evaluate(X_test, Y_test)


# ### 2. The best model with 10 neurons each layer

# In[17]:


def create_network(optimizer = 'rmsprop'):
    MaxNormDense = partial(keras.layers.Dense,
                           activation="selu", kernel_initializer="lecun_normal",
                           kernel_constraint=keras.constraints.max_norm(1.)
                           )

    model = models.Sequential()
    model.add(layers.Dense(10, activation='relu', input_shape=(12,)))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

neural_network = KerasClassifier(build_fn=create_network,verbose=0)


# ### II. Grid Search for neuron = 10
# #### 1. Optimizers: : adam, nadam, sgd, rmsprop, adamax, adagrad
# #### 2. Epoch: 100, 200, 300

# In[20]:


epochs = [100, 200, 300]
optimizers = ['rmsprop', 'nadam', 'adam', 'sgd', 'adamax', 'adagrad']

# Create hyperparameter options
hyperparameters = dict(optimizer = optimizers, epochs=epochs)

# Create grid search
grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters, cv=3) 

# Fit gird search
grid_output = grid.fit(X_test, Y_test)
print(grid_output)
print(grid_output.best_params_)
print(grid_output.best_score_)


# ### Best model with 10 neuron each layer

# In[21]:


model = Sequential([
    Dense(10, activation='relu', input_shape=(12,)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
    ])


# In[22]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[23]:


hist = model.fit(X_train, Y_train, 
                 batch_size=32, 
                 epochs=100, 
                 validation_split=0.3)


# In[24]:


fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss','Accuracy'))

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['loss'],
                    mode='lines',marker_color='red',name='Train loss'), row=1, col=1)

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['val_loss'],
                    mode='lines',marker_color='blue',name='Val loss'), row=1, col=1)

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['accuracy'],
                    mode='lines',marker_color='orange',name='Train accuracy'), row=1, col=2)

fig.add_trace(
    go.Scatter(x=np.array(range(101)), y=hist.history['val_accuracy'],
                    mode='lines',marker_color='purple',name='Val_accuracy'), row=1, col=2)

fig.update_layout(
    title_text='Loss and accuracy of the best model', 
    xaxis_title_text='Epoch', 
    bargap=0.2, 
    bargroupgap=0.1,
    hovermode='x'
)

fig.show()


# ### Model Evaluation

# In[25]:


model.evaluate(X_test, Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Classifiers

# ### I. Logistic regression

# In[1]:


from sklearn.linear_model import LogisticRegression


# In[69]:


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,Y_train)

prediction = log_reg_classifier.predict(X_test)
correct_rate = np.mean(prediction == Y_test)


# In[70]:


print("The accuracy of logistic regression is:",correct_rate)


# ### II. Gaussian Naïve Bayes

# In[71]:


from sklearn.naive_bayes import GaussianNB


# In[72]:


NB_classifier = GaussianNB().fit(X_train,Y_train)
prediction = NB_classifier.predict(X_test)
correct_rate = np.mean(prediction== Y_test)


# In[73]:


print("The accuracy of Gaussian Naïve Bayes is:",correct_rate)


# ### III. Gaussian SVM

# In[74]:


from sklearn import svm
from sklearn.preprocessing import StandardScaler


# In[75]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[76]:


svm_classifier = svm.SVC(kernel ='linear')
svm_classifier.fit(X_train,Y_train)

prediction = svm_classifier.predict(X_test)
correct_rate = svm_classifier.score(X_test, Y_test)


# In[77]:


print("The accuracy of Gaussian SVM is:",correct_rate)


# ## Conclusion

#    After 100 epochs, MLP provided 38.38% accuracy. Logistic regression provided 44.81%, Gaussian Naïve Bayes provided 46.67%, and Gaussian SVM provided 51.48% accuracy. It is clear to see that Gaussian SVM provided the highest accuracy and MLP neural network provided the least accuracy. However, it does not mean that neural network is not as reliable as classifiers. 
#    
#    
#    There are two possible reasons why MLP did not perform better than classifiers. First of all, the data is linear. This is quilt important. ANN have the ability to learn and model non-linear and complex relationship and neural networks are good to model with nonlinear data with large number of inputs. On the other hand, regression models and classifiers would perform better on linear data since these models are based on statistical model. 
#    
#    
#    Secondly, the data set is not high-dimensional. Neural networks are best for situations where the data is high-dimensional like images. The data set used is a regular linear data set. So, the shape of the data set is perfectly fit the regression models and classifiers. 
#    
#    
#    In conclusion, back to the research question, two conditions would be the best for neural network. Neural network would perform the best for the high dimensional data or non-linear data.

# In[ ]:




