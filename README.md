# 3rd Place Model for Microsoft Data Science Competition (Predicting Student Loan Repayment Rates)
For the Summer 2017 term for Microsoft's Data Science Capstone, students were tasked with predicting student loan repayment rates for institutions. [Follow this link for more information about the project](https://www.datasciencecapstone.org/competitions/1/student-loans/page/1/).

### Files Descriptions

#### Final Model
- **[DAT102X_6.31.py](/DAT102X_6.31.py)**: A python program containing the final predictive model. This is a complete pipeline for feature engineering, transformations, and Gradient Boosted Tree implemention. This file was written using Spyder and should be used as such, it is not designed to be run completely from start to finish. Contains cells which are useful (and necessary) for getting results and comparing scores. Contains both xgboost and lightgbm packages. Also has a cell for plotting feature importance from xgboost.
- **[DAT102X_Report_Foster.pdf](/DAT102X_Report_Foster.pdf)**: Final report describing the data, correlations, conclusions, and reccomendations.

#### Supporting Files
- **[DAT102X_1_Exploratory.ipynb](/DAT102X_1_Exploratory.ipynb)**: Notebook exploring the data and variables with histograms, regression scatter plots, boxplots/violin plots. Later, this notebook was expanded to create visualizations for the final report.
- **[DAT102X_2_Model_Comparison.ipynb](/DAT102X_2_Model_Comparison.ipynb)**: This was a regression task, so this notebook compares the performance of various regression models from the sci-kit learn package (Random Forest, Linear, KNeighbors, etc.).
- **[DAT102X_3_Feature_Selection.ipynb](/DAT102X_3_Feature_Selection.ipynb)**: Notebook for some feature engineering and correlation calculations.
- **[features_corr.csv](/features_corr.csv)**: A spreadsheet containing correlations between features and repayment rates, created with a Microsoft Azure ML module.
- **[permutation_feature_importance.csv](/permutation_feature_importance.csv)**: A spreadsheet with feature importance scores, created with a Microsoft Azure ML module.
- **[train_values.csv](/train_values.csv)**: Training Data
- **[train_labels.csv](/train_labels.csv)**: Labels for training data (repayment rates)
- **[xgb_features.png](/xgb_features.png)**: A graph of feature importances according to xgboost.
- **[sklearn_features.png](/sklearn_features.png)**: A graph of feature importances according to scikit-learn Gradient Boosted Tree.
- **[final_results.png](/final_results.png)**: A screenshot of my model's submission results.
- **[final_results2.png](/final_results2.png)**: A screenshot of the final leaderboard.

## Author
[Hiram Foster](https://linkedin.com/in/hiramf)
I would love to hear from you concerning suggestions, questions, or career opportunities!
