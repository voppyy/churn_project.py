# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Importing Scikit-learn modules for data preparation and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# VISUALIZING DATASET INFORMATION FOR DATA CLEANING
df = pd.read_csv("C:/Users/irene/Desktop/magistrale iulm/secondo anno/data mining e text analytics/Churn_Modelling.csv") 
print(df.head()) 
print(df.info())
print(df.isnull().sum())
print(df[df.duplicated()])

# Converting categorical variables into numerical ones
label_encoder = LabelEncoder() 
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df = pd.get_dummies(df,columns=["Geography"], drop_first=True) #one hold encoding
print(df.head())

# Selection of variables of interest
features = ["CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", 
            "EstimatedSalary", "Geography_Germany", "Geography_Spain"] # DO NOT select EXIT as a variable since it will be our dependent variable to predict

# Defining the target variable to predict (Churn)
X = df[features] # Independent variables (features)
y = df["Exited"] # Dependent variable (target)
df = df.rename(columns={'Exited': 'Churn'}) # Renaming the column "Exited" to "Churn" for better readability
print(df.head())


# VISUALIZATION OF THE CHURN VARIABLE

# Counting the number of observations for each class of the target variable "Churn"
churn_counts = df['Churn'].value_counts()
# Calculating the percentages of each class
churn_percentages = churn_counts / len(df) * 100
# Display a bar chart to show the distribution of classes
churn_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Churn variable distribution')
plt.xlabel('Churn (1=Yes, 0=No)')
plt.ylabel('Number of observations')
plt.show()
# Printing the percentages of churn and non-churn
print("Churn percentage:", churn_percentages[1])
print("Non-churn percentage:", churn_percentages[0])



# CLASS BALANCING FOR CHURN WITH OVERSAMPLING OF THE MINORITY CLASS
# Separation of data into majority (0) and minority (1) classes
minority_class = df[df['Churn'] == 1]
majority_class = df[df['Churn'] == 0]
# Oversampling of the minority class
minority_oversampled = minority_class.sample(len(majority_class), replace=True, random_state=42)
# Combining classes into a balanced dataset
df_balanced = pd.concat([majority_class, minority_oversampled])
# Visualising the new distribution of classes
print(df_balanced['Churn'].value_counts())

# VISUALIZATION OF THE BALANCED CHURN
# Counting the number of observations for each class in the balanced dataset
churn_counts_balanced = df_balanced['Churn'].value_counts()
# Calculating the percentage of observations for each class
churn_percentages_balanced = churn_counts_balanced / len(df_balanced) * 100
# Display a bar chart to show the balanced distribution of classes
churn_counts_balanced.plot(kind='bar', color=['green', 'red'])
plt.title('Balanced Churn Variable Distribution')
plt.xlabel('Churn (1=Yes, 0=No)')
plt.ylabel('Number of observations')
plt.xticks(rotation=0)
plt.show()
# Printing the percentages of churn and non-churn in the balanced dataset
print("Balanced churn percentage:", churn_percentages_balanced[1])
print("Balanced non-churn percentage:", churn_percentages_balanced[0])


# CREATING THE TRAINING AND TEST DATASET
# Splitting the balanced dataset into training (80%) and test (20%) data
X_balanced = df_balanced[features]
y_balanced = df_balanced["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)


# STANDARDIZATION OF VARIABLES
# Standardizing the independent variables to remove scale differences
scaler = StandardScaler() # Initializing the Standard Scaler
X_train = scaler.fit_transform(X_train)  # Fitting and transforming the training data
X_test = scaler.transform(X_test) # Transforming the test data using the parameters calculated from the training data
print(X_train[:5], X_test[:5]) # Printing the first 5 records of the standardized dataset


# FIRST MODEL: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

# Creating and training a Random Forest model with 100 trees
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
print(random_forest_model.fit(X_train, y_train))  # Training the model using the training data

# Making predictions on the test data
y_pred = random_forest_model.predict(X_test)

# Calculation and visualization of the confusion matrix, classification report, and accuracy
cof_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
random_forest_accuracy = accuracy_score(y_test, y_pred)  # Use specific metric name
print(cof_matrix, class_report, random_forest_accuracy)

# visualization of variable importance
importances = random_forest_model.feature_importances_
indices = np.argsort(importances) [::-1]
names = [features[i] for i in indices]

# Bar chart of variable importance
plt.figure(figsize=(20, 10))
plt.title("Variable Importance")
plt.barh(range(X.shape[1]), importances[indices])
plt.yticks(range(X.shape[1]), names)
plt.show()




# SECOND MODEL: LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
# Creating and training the logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Making predictions on the test data with the logistic regression model
y_pred_log_reg = log_reg.predict(X_test)

# Calculation and visualization of the confusion matrix, classification report, and accuracy for the logistic regression model
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(conf_matrix_log_reg, class_report_log_reg, accuracy_log_reg)




# THIRD MODEL: SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
# Creating and training the SVM model with a linear kernel
svm_model = SVC(kernel = "linear", random_state=42)
svm_model.fit(X_train, y_train)

# Making predictions on the test data with the SVM model
y_pred_svm = svm_model.predict(X_test)

# Calculation and visualization of the confusion matrix, classification report, and accuracy for the SVM model
conf_matrix_svm = svm_model.predict(X_test)
class_report_svm = classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(conf_matrix_svm, class_report_svm, accuracy_svm)




# FOURTH MODEL: K-NEAREST NEIGHBORS (KNN)
from sklearn.neighbors import KNeighborsClassifier
# Creazione e addestramento del modello K-Nearest Neighbors con 5 vicini
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Making predictions on the test data
y_pred_knn = knn_model.predict(X_test)

# Calculation and visualization of the confusion matrix, classification report, and accuracy 
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(conf_matrix_knn, class_report_knn, accuracy_knn)


# Create a dictionary to store model accuracies
model_accuracies = {
    "Random Forest": random_forest_accuracy,
    "Logistic Regression": accuracy_log_reg,
    "SVM": accuracy_svm,
    "KNN": accuracy_knn
}

# Print the model accuracies in a clear table format
print("\nComparison of Model Accuracies:")
print("{:<25} {:<15}".format('Model', 'Accuracy'))
print("-" * 40)
for model_name, accuracy in model_accuracies.items():
    print("{:<25} {:<15.4f}".format(model_name, accuracy))


# The Random Forest model in this case proved to be the most effective for churn prediction, offering a good balance between precision and recall. 
# ! The best model also depends on business objectives !