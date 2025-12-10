# 🏠 House Price Prediction using EDA

This project predicts Ames housing prices using Exploratory Data Analysis (EDA) and a Linear Regression model.
It includes a clean, dark-themed Streamlit web app where users can input property details and instantly get a predicted price, along with visual insights from EDA.

# 🔍 Overview

The goal of this project is to analyze how various housing features — such as overall quality, living area, neighborhood, and year built — influence property prices.
Using the Ames Housing Dataset, this app performs data preprocessing, visualization, and prediction through an interactive interface.

# ⚙️ Features

Interactive Streamlit App to predict housing prices

Machine Learning Model (Linear Regression) trained on cleaned data

EDA Visualizations: missing values, correlation heatmap, and feature importance

Dark Spotify-inspired UI theme

PDF Export Option for EDA charts

Dynamic Input Fields (sliders, dropdowns, and numeric inputs)

# 🧠 Tech Stack
| Component          | Technology Used                                 |
| ------------------ | ----------------------------------------------- |
| Frontend/UI        | Streamlit                                       |
| Data Analysis      | Pandas, NumPy                                   |
| Data Visualization | Matplotlib, Seaborn                             |
| Machine Learning   | Scikit-learn (LinearRegression, StandardScaler) |
| Report Generation  | FPDF                                            |
| Dataset            | Ames Housing Dataset                            |


# 📂 Project Structure
📦 House-Price-Prediction
 ┣ 📜 app.py                     # Streamlit app
 ┣ 📜 model_train.py             # Model training and preprocessing
 ┣ 📜 train.csv                  # Dataset
 ┣ 📜 model_input_template.csv   # Template for input features
 ┣ 📜 linear_model.pkl           # Trained model
 ┣ 📜 scaler.pkl                 # Feature scaler
 ┣ 📜 EDA_Report.pdf             # Exported EDA report
 ┣ 📂 assets/                    # Graph images used in README
 ┗ 📜 README.md                  # Project documentation

# 📊 Exploratory Data Analysis

Below are some of the insights from EDA visualizations:
1️⃣ Top Features with Missing Values
Shows which columns had the most missing data before cleaning.

2️⃣ Correlation Heatmap
Displays the relationships between numeric features like GrLivArea, GarageCars, and SalePrice.

3️⃣ Top 10 Important Features
Highlights the most influential features contributing to the predicted sale price.

# 🚀 How to Run the Project

Clone the repository:
git clone https://github.com/<your-username>/Vidyarthi-Saathi.git
cd House-Price-Prediction

Install dependencies:
pip install -r requirements.txt

#  Run the app:
streamlit run app.py

Open the provided localhost URL in your browser.

# 🧩 Model Insights

GrLivArea, OverallQual, and YearBuilt are strong predictors of sale price.

Higher OverallQual (material & finish quality) leads to higher prices.

Neighborhoods have a noticeable impact on housing value distribution.

# 🖤 UI Highlights

Minimalist Spotify-style dark theme

Animated buttons and EDA toggles

Clean separation of Home and Prediction pages

📈 Sample Output
| Feature     | Input Value | Predicted Price |
| ----------- | ----------- | --------------- |
| OverallQual | 7           | $210,000        |
| GrLivArea   | 1650        | $208,450        |
| GarageCars  | 2           | $215,900        |
| YearBuilt   | 2003        | $221,700        |

# Graphs
## 📊 Exploratory Data Analysis  

### 1️⃣ Top Features with Missing Values  
![Missing Values](https://raw.githubusercontent.com/Aryanambre77/House-Price-Prediction-System-Using-EDA/main/assets/missing_values.png)

### 2️⃣ Correlation Heatmap  
![Correlation Heatmap](https://raw.githubusercontent.com/Aryanambre77/House-Price-Prediction-System-Using-EDA/main/assets/correlation_heatmap.png)

### 3️⃣ Top 10 Important Features  
![Feature Importance](https://raw.githubusercontent.com/Aryanambre77/House-Price-Prediction-System-Using-EDA/main/assets/feature_importance.png)
