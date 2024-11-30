This project is designed to provide a comprehensive analysis and optimization of ensemble learning models specifically targeted at churn prediction. By leveraging the strengths of different ensemble methods, including Gradient Boosting and AdaBoost, the project aims to refine predictive performance through systematic training, evaluation, and hyperparameter tuning.

Tools and Technologies
Python 3.10.13: The chosen programming language, ideal for data manipulation and machine learning tasks.
Scikit-learn: Provides robust tools for data mining and data analysis, including pre-built models and tuning techniques.
XGBoost & AdaBoost: Advanced ensemble techniques utilized for their strength in handling complex datasets and improving prediction outcomes.
Conda Environment: Ensures a controlled and consistent setup across all development and production phases, managing dependencies effectively.

Methodology
The process begins with preprocessing the input data, ensuring that it is clean and formatted correctly. This data is then used to train several models. Each model is first evaluated using standard metrics such as confusion matrices and classification reports to establish a baseline performance. Following this, a detailed hyperparameter tuning session is conducted using GridSearchCV, aiming to find the optimal settings for each model that enhance their prediction accuracy.

# Model Inferencing for daily level usage


Creating a Python Environment

  * conda create --name churn_prediction python=3.10.13
  * conda activate churn_prediction
  * conda install pip
  * pip install -r requirements.txt

Call inference.py for live predictions and pass data locations 


# Model Training and Evalution


Summary of best model gradient boosting classifier - top_6_features

Classification Report:
              precision    recall  f1-score   support
              0.0       0.92      0.97      0.94       545
              1.0       0.96      0.92      0.94       543

   accuracy                           0.94      1088
   macro avg       0.94      0.94      0.94      1088
weighted avg       0.94      0.94      0.94      1088


The classification report details robust performance metrics for a model that differentiates between churn (class 1.0) and non-churn (class 0.0) customers. The model achieved a precision of 0.92 for non-churn and 0.96 for churn customers, paired with recall rates of 0.97 for non-churn and 0.92 for churn. Both classes received an F1-score of 0.94, reflecting highly effective predictive accuracy. Overall, the model's accuracy is 94% across 1088 instances, highlighting its strong capability in accurately predicting and distinguishing between customers who churn and those who do not. This demonstrates the model’s balanced performance in handling both potential outcomes.


Summary of best model gradient boosting classifier - top_8_features

Classification Report:
              precision    recall  f1-score   support

  0.0       0.93      0.97      0.95       545
  1.0       0.97      0.93      0.95       543

   accuracy                           0.95      1088
   macro avg       0.95      0.95      0.95      1088
weighted avg       0.95      0.95      0.95      1088

The classification report highlights the strong performance of a model designed to predict customer churn. The model distinguishes between two classes: non-churn (class 0.0) and churn (class 1.0). For non-churn customers, the model achieved a precision of 0.93, indicating that 93% of predicted non-churn cases were correct. The recall for non-churn is an impressive 0.97, showing that 97% of actual non-churn cases were correctly identified. This balance of precision and recall results in an F1-score of 0.95 for non-churn, reflecting the model’s high reliability in predicting this category.


Summary of best model gradient boosting classifier - top_10_features

Classification Report:
              precision    recall  f1-score   support

   0.0       0.92      0.96      0.94       545
   1.0       0.96      0.92      0.94       543

   accuracy                           0.94      1088
   macro avg       0.94      0.94      0.94      1088
weighted avg       0.94      0.94      0.94      1088



