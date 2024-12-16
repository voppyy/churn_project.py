# Churn Prediction with Machine Learning Models
This project, created for the university course in Data Mining and Text Analytics, aims to predict customer churn through a step-by-step approach, allowing anyone to use and analyze a dataset even without knowing it in detail.

For academic reasons and with the intention of making the project useful in business contexts as well, we developed flexible code that can be adapted to different realities. The main focus is to provide tools to understand the dataset, process the data in a robust way, and compare the performance of different models according to business objectives.


## 🚀 About us
We are students who are passionate about data science and machine learning. This project is a practical application of the skills we acquired during the Data Mining and Text Analytics course. Our mission is to offer flexible and reproducible tools for data analysis, with a vision geared toward both learning and practical applications in the corporate world.


## 📘 Appendix
This section provides additional details to ensure a step-by-step understanding and adaptability of the code. Below is the structured approach followed in the project:

1. Dataset Exploration and Visualization:
The code begins by displaying the first few rows of the dataset, allowing the user to familiarize themselves with the variables.
The presence of null values, duplicates, and other anomalies is analyzed, forming the foundation for data cleaning.

2. Data Preprocessing and Preparation:
Categorical variables are converted into numerical representations using techniques such as label encoding and one-hot encoding.

The churn variable is balanced to address the common imbalance in these datasets, preventing models from favoring the majority class.

3. Visualizations:
Clear visualizations illustrate the initial and balanced churn distribution, as well as the importance of predictive variables.

4. Standardization:
Numerical variables are standardized to eliminate scaling differences, ensuring that models perform optimally.

5. Implemented Models:
Several models have been implemented to enable performance comparisons:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

6. Evaluation and Analysis:
Metrics such as accuracy, precision, recall, and confusion matrix are computed for each model.

Feature importance is visualized for models that support this analysis.


