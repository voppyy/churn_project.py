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


## ⚙️ Installation

Clone the repository: https://github.com/voppyy/churn_project.py.git

Navigate to the project directory: cd churn_project.py

USAGE
1. Prerequisites:

   Make sure you have the following tools installed:
   - Python (version 3.x or above)
   - A code editor (e.g., Visual Studio Code, PyCharm)


2. Create an enviroment:
   
   A virtual environment isolates your project's dependencies, preventing conflicts with other projects. We used Anaconda but you can also create it from scratch.

   
3. Installing Libraries:
   
   Here’s the list of libraries required for the project:
   - pandas: For data manipulation and analysis
   - numpy: For numerical computations
   - matplotlib: For data visualization
   - seaborn: For statistical visualizations
   - scikit-learn: For machine learning tasks
     
   After installing Python and creating a virtual environment, install the necessary libraries using the following command in your terminal: pip install pandas numpy matplotlib seaborn 
   scikit-learn


## 💡 Acknowledgements

We acknowledge the platform "Kaggle.com" for providing valuable datasets that were essential for this project.

We would also like to credit Karina Data Scientist and her YouTube channel (Karina Data Scientist) for inspiration and guidance in developing parts of our code.

Team Contributions:
The project was a collaborative effort, with each team member contributing equally. The final outcome reflects the continuous teamwork and dedication of Valeria Coppola, Irene Malcangi, and Shereen Zangane.

We also acknowledge the support of generative AI platforms such as ChatGPT and Gemini, which assisted in brainstorming solutions, debugging code, and improving the documentation.

All team members collaborated to upload the final version of the project to GitHub and ensured its seamless presentation.


## 📸 Screenshots

Below are some screenshots showcasing key aspects of the project:

- Churn Distribution: The visualization of the churn variable before balancing the dataset.
- Churn After Balancing: The visualization of the churn variable after applying balancing techniques.
- Feature Importance: The importance of variables as shown by the Random Forest model.

These visuals provide insights into the data analysis, model performance, and key findings of the project.

![descrizione immagine](https://github.com/voppyy/churn_project.py/blob/64bb54cdd1473d86d24d5a30c3227f39c38b6e72/images/churn%20variable%20distribution.jpg)

![descrizione immagine](https://github.com/voppyy/churn_project.py/blob/e725484d392c0a2e4b6a5cd0d5317c40e5c802bc/images/balanced%20churn%20variable%20distribution.jpg)

![descrizione immagine](https://github.com/voppyy/churn_project.py/blob/db6115e6115090da82c954d38952a5073ee19284/images/variable%20importance.jpg).


## 📄 License

This project is licensed under the MIT License.

```
Copyright (c) 2024 Valeria Coppola, Irene Malcangi, Shereen Zangane

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```







