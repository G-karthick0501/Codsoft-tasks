#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[2]:


import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


# Drop Cabin column since it has many missing values
df.drop('Cabin', axis=1, inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


# Fill missing Age values with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the most common port
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


# No missing values now 


# In[10]:


df


# In[11]:


df_encod=df.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_encod['Sex'] = le.fit_transform(df_encod['Sex'])
df_encod['Embarked'] = le.fit_transform(df_encod['Embarked'])


# In[12]:


df_encod


# In[13]:


# Droping these columns as they would not be used for prediction
df_encod.drop('PassengerId', axis=1, inplace=True)
df_encod.drop('Name', axis=1, inplace=True)
df_encod.drop('Ticket', axis=1, inplace=True)


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = df_encod.corr()
print(corr_matrix)
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[15]:


# Intresting to see correlation of 'Age' and 'Fare' contributes less to survival

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist([df[df['Survived'] == 1]['Age'], df[df['Survived'] == 0]['Age']], 
         bins=20, color=['g', 'r'], label=['Survived', 'Not Survived'])
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.title('Survived vs Age')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.hist([df[df['Survived'] == 1]['Fare'], df[df['Survived'] == 0]['Fare']], 
         bins=20, color=['g', 'r'], label=['Survived', 'Not Survived'])
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.title('Survived vs Fare')
plt.legend()
plt.show()


# In[16]:


# Based on heat map lets use features 'Pclass','Sex','Fare','Embarked'

X = df_encod[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = df_encod['Survived']
print(X)
print(y)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[18]:


#Splitting of data into trainning and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Model 1: Based on Polynomial Regression


# In[19]:


poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)


# In[20]:


poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)


# In[22]:


y_poly_pred = poly_model.predict(X_poly_test)
print(y_poly_pred)


# In[23]:


y_poly_pred_class = (y_poly_pred > 0.5).astype(int)
print(y_poly_pred_class)


# In[24]:


poly_accuracy = accuracy_score(y_test, y_poly_pred_class)
poly_classification_rep = classification_report(y_test, y_poly_pred_class)
poly_conf_matrix = confusion_matrix(y_test, y_poly_pred_class)


# In[25]:


print(f'Accuracy: {poly_accuracy}')
print('Classification Report:')
print(poly_classification_rep)
print('Confusion Matrix:')
print(poly_conf_matrix)


# In[28]:


# Print the polynomial equation
coef = poly_model.coef_
intercept = poly_model.intercept_
feature_names = poly.get_feature_names_out(['Pclass', 'Sex', 'Fare', 'Embarked'])
terms = [f"{coef[i]:.3f}*{feature_names[i]}" for i in range(len(coef))]
polynomial_equation = " + ".join(terms) + f" + {intercept:.3f}"
print('Polynomial Equation:')
print(polynomial_equation)


# In[29]:


import numpy as np

# Function to predict survival based on user input
def predict_survival(pclass, sex, fare, embarked):
    # User input
    user_input = np.array([[pclass, sex, fare, embarked]])
    
    # Apply polynomial features transformation
    user_input_poly = poly.transform(user_input)
    
    # Predict survival
    prediction = poly_model.predict(user_input_poly)
    survival = (prediction > 0.5).astype(int)
    
    return survival[0]

# Presenting variables and restrictions
print("Please provide the following details to predict survival on the Titanic:")
print("1. Pclass (Passenger class): 1 = 1st, 2 = 2nd, 3 = 3rd")
print("2. Sex: 0 = female, 1 = male")
print("3. Fare: Ticket fare (a positive float value)")
print("4. Embarked (Port of Embarkation): 0 = Cherbourg, 1 = Queenstown, 2 = Southampton")

# Taking user input
pclass = int(input("Enter Pclass (1/2/3): "))
sex = int(input("Enter Sex (0 for female, 1 for male): "))
fare = float(input("Enter Fare: "))
embarked = int(input("Enter Embarked (0/1/2): "))

# Predict survival
survival = predict_survival(pclass, sex, fare, embarked)

# Display result
if survival == 1:
    print("The passenger would have survived.")
else:
    print("The passenger would not have survived.")


# In[32]:


# Model 2 :Based on Decision Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier


# In[33]:


decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)


# In[34]:


y_pred = decision_tree_model.predict(X_test)


# In[35]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[36]:


print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)
print('Confusion Matrix:')
print(conf_matrix)


# In[41]:


# Visualize the tree
from sklearn.tree import plot_tree

plt.figure(figsize=(30,15))
plot_tree(decision_tree_model, feature_names=['Pclass', 'Sex', 'Fare', 'Embarked'], class_names=['Not Survived', 'Survived'], filled=True,fontsize=8)
plt.show()


# In[42]:


import numpy as np

# Function to predict survival based on user input
def predict_survival_decision_tree(pclass, sex, fare, embarked):
    # User input
    user_input = np.array([[pclass, sex, fare, embarked]])
    
    # Predict survival
    prediction = decision_tree_model.predict(user_input)
    
    return prediction[0]

# Presenting variables and restrictions
print("Please provide the following details to predict survival on the Titanic:")
print("1. Pclass (Passenger class): 1 = 1st, 2 = 2nd, 3 = 3rd")
print("2. Sex: 0 = female, 1 = male")
print("3. Fare: Ticket fare (a positive float value)")
print("4. Embarked (Port of Embarkation): 0 = Cherbourg, 1 = Queenstown, 2 = Southampton")

# Taking user input
pclass = int(input("Enter Pclass (1/2/3): "))
sex = int(input("Enter Sex (0 for female, 1 for male): "))
fare = float(input("Enter Fare: "))
embarked = int(input("Enter Embarked (0/1/2): "))

# Predict survival
survival = predict_survival_decision_tree(pclass, sex, fare, embarked)

# Display result
if survival == 1:
    print("The passenger would have survived.")
else:
    print("The passenger would not have survived.")


# In[ ]:




