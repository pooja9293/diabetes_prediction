#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("code.csv")


# In[3]:


data


# In[4]:



import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('code.csv')


X = dataset.drop('Glucose', axis=1)  
y = dataset['BloodPressure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42)  

kmeans.fit(X_train)


y_pred = kmeans.predict(X_test)


cluster_mapping = {cluster_label: 1 if sum(y_train[kmeans.labels_ == cluster_label]) > len(y_train[kmeans.labels_ == cluster_label])/2 else 0 for cluster_label in set(kmeans.labels_)}
y_pred_binary = [cluster_mapping[label] for label in y_pred]


accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


dataset = pd.read_csv('code.csv')

X = dataset.drop('BloodPressure', axis=1)  


kmeans = KMeans(n_clusters=2, random_state=42)  
kmeans.fit(X)


dataset['cluster'] = kmeans.labels_

cluster_counts = dataset['cluster'].value_counts()


plt.bar(cluster_counts.index, cluster_counts.values)
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Distribution of Data Points in Clusters')
plt.show()


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv('code.csv')


features = dataset[['Pregnancies', 'BloodPressure', 'Glucose', 'SkinThickness', 'Insulin']]
labels = dataset['Outcome'] 

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataset = pd.read_csv('code.csv')

features = dataset[['Pregnancies', 'Glucose', 'Glucose', 'SkinThickness', 'Insulin']]
labels = dataset['Outcome']  # Replace 'diabetes_label_column_name' with the actual label column name


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
rscaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

label_counts = labels.value_counts()
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Diabetes Labels')
plt.show()

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

dataset = pd.read_csv('code.csv')


features = dataset[['Pregnancies', 'BloodPressure', 'Glucose', 'SkinThickness', 'Insulin']]
labels = dataset['Outcome']  

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=features.columns, class_names=['0', '1'], filled=True)
plt.show()

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataset = pd.read_csv('code.csv')

features = dataset[['Pregnancies', 'BloodPressure', 'Glucose', 'SkinThickness', 'Insulin']]
labels = dataset['Outcome']  


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


dataset = pd.read_csv('code.csv')


features = dataset[['Pregnancies', 'BloodPressure', 'Glucose', 'SkinThickness', 'Insulin']]
target = dataset['Outcome']  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')


plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True Values vs. Predicted Values')
plt.show()


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


dataset = pd.read_csv('code.csv')


features = dataset[['Pregnancies', 'BloodPressure', 'Glucose', 'SkinThickness', 'Insulin']]
labels = dataset['Outcome']  
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

plt.pie(pd.Series(y_pred).value_counts(), labels=pd.Series(y_pred).value_counts().index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Predicted Values (Classification)')
plt.show()

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv('code.csv')


features = dataset[['BloodPressure', 'SkinThickness']]
labels = dataset['Outcome'] 
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[2]:



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('code.csv')


X = dataset.drop('Outcome', axis=1)  
y = dataset['Outcome']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)


y_pred = random_forest_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[ ]:




