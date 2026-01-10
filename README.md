Mobile Price Classification using Machine Learning

This project builds a machine learning system that predicts the price range of a mobile phone based on its technical specifications such as RAM, internal storage, battery power, camera quality, and connectivity features.

Instead of predicting an exact price, the model classifies phones into four price categories:

Class	Price Range
0	Low cost
1	Medium cost
2	High cost
3	Very high cost

This type of classification system is commonly used by e-commerce platforms, mobile manufacturers, and market analysts to segment products and guide pricing strategies.

Dataset Description

The dataset contains multiple technical features of smartphones, including:
Battery Power
RAM
Internal Memory
Clock Speed
Number of Cores
Front & Primary Camera
Screen Height & Width
Connectivity (3G, 4G, WiFi)
Touch Screen Support

The target variable is:
price_range → Class label (0–3)

Project Workflow
This project follows a complete machine learning pipeline:
1) Data Loading
2) Exploratory Data Analysis (EDA)
3) Data Preprocessing
4) Feature Scaling
5) Model Training
6) Model Evaluation
7) Prediction on New Data

Exploratory Data Analysis (EDA)
EDA was performed to:
Understand feature distributions
Detect correlations between RAM, battery, and price
Identify important predictors of phone cost

Key insights:
RAM and battery power are strongly correlated with price.
Phones with 4G, WiFi, and higher cores tend to fall into higher price categories.

Machine Learning Models Used
Several classification models were tested:
Logistic Regression
K-Nearest Neighbors (KNN)
Random Forest Classifier
Decision Tree
Support Vector Machine (SVM)

The best performing model was selected based on:
Accuracy
Precision
Recall
F1-score

Final Model Performance
The final trained model achieved high accuracy on unseen test data, indicating strong generalization.

How to Run the Project
Clone the repository
git clone https://github.com/mansityagi0606-lab/Mobile-price-classification.git
cd Mobile-price-classification

Install required libraries
pip install -r requirements.txt

Run the notebook or training script
jupyter notebook
or
python train.py

Example Prediction
Given a mobile phone with:
RAM = 6GB
Battery = 4000 mAh
4G Support = Yes
Camera = 16MP

The model predicts:
Price Range → High (Class 2)

Real-World Applications
This project can be used in:
Mobile price recommendation systems
E-commerce product filtering
Market analysis & competitor pricing
Inventory and demand forecasting

Technologies Used
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-Learn
Jupyter Notebook
