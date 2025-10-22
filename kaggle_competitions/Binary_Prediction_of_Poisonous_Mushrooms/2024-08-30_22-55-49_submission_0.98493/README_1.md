# 🍄 Mushroom Edibility Classification (Kaggle)
This project focuses on building, tuning, and comparing two Machine Learning models,  
including **CatBoost**,**CatBoost tuning** , and a **Deep Neural Network (DNN)**,  
to accurately classify mushrooms as either **edible ('e')** or **poisonous ('p')** based on 22 categorical features.
🔍 Project Overview
- Domain: Binary Classification (Edible vs. Poisonous)
- Goal: Achieve maximum prediction accuracy, measured by the Matthews Correlation Coefficient (MCC).
- Tech Stack: Python, CatBoost, Scikit-learn, TensorFlow/Keras, Pandas, NumPy.
- Key Features:
Robust Preprocessing: 
Scaling, 
One-Hot Encoding (OHE), and handling of missing values.
CatBoost Utilization: Effective handling of raw categorical data without OHE.
Hyperparameter Tuning: Grid Search for optimal CatBoost performance.
Model Comparison: Evaluation of Tree-based, Linear (LogReg), and Deep Learning models.
Stratified K-Fold (Folding): Advanced ensembling technique for stable final predictions.
📁 Repository Structuremushroom_edibility_classification/
├── README.md
├── mushroom_classification_main.ipynb (Main EDA, Preprocessing, and Modeling script)
├── mushroom_classification_model_ver-3.h5 (Saved Keras DNN model)
├── data/│   └── agaricus-lepiota.csv (Source dataset)
├── images/
│   ├── feature_importances_catboost.png
│   ├── dnn_loss_accuracy_plot.png
│   └── model_comparison_table.png

📊 Model Comparison: Highest MCC Scores
The final performance comparison table illustrates the effectiveness of different modeling 
approaches on the validation set. CatBoost models, which inherently handle categorical features 
well, achieved the best scores.

Model NameMCCLoglossTimeLearning RateDepthCatBoost_Tuned0.99950.00180:02:450.0212
CatBoost_Baseline0.99890.00310:00:15defaultdefault
DNN_Keras_Baseline0.99720.01250:01:100.00057
Logistic_Regression_OHE0.97540.02510:00:03N/AN/A

Feature Importance
The most significant features identified by the top-performing CatBoost model were related to the mushroom's Odor and Gill Size.
🧪 Environment Setup
The project requires standard data science libraries and specific GPU-enabled packages for 
CatBoost and TensorFlow performance.

To recreate the environment:

Bash# 
Assuming you use Anaconda/Miniconda
conda create -n mushroom_env python=3.9
conda activate mushroom_env
# Install core libraries
pip install pandas numpy scikit-learn matplotlib
# Install CatBoost and TensorFlow for GPU acceleration (optional but recommended)
pip install catboost tensorflow-gpu

🚀 How to RunClone the repository:
Bash
git clone https://github.com/YourUsername/YourRepoName.git
Navigate to the project folder:
Bashcd mushroom_edibility_classification
Open the mushroom_classification_main.ipynb notebook in Jupyter (or your preferred environment).
Run all cells sequentially to perform EDA, Preprocessing, Model Training, 
Hyperparameter Tuning, and generate the Final Submission File.

