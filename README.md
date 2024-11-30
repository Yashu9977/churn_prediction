This project is designed to provide a comprehensive analysis and optimization of ensemble learning models specifically targeted at churn prediction. By leveraging the strengths of different ensemble methods, including Gradient Boosting and AdaBoost, the project aims to refine predictive performance through systematic training, evaluation, and hyperparameter tuning.

Tools and Technologies
Python 3.10.13: The chosen programming language, ideal for data manipulation and machine learning tasks.
Scikit-learn: Provides robust tools for data mining and data analysis, including pre-built models and tuning techniques.
XGBoost & AdaBoost: Advanced ensemble techniques utilized for their strength in handling complex datasets and improving prediction outcomes.
Conda Environment: Ensures a controlled and consistent setup across all development and production phases, managing dependencies effectively.

Methodology
The process begins with preprocessing the input data, ensuring that it is clean and formatted correctly. This data is then used to train several models. Each model is first evaluated using standard metrics such as confusion matrices and classification reports to establish a baseline performance. Following this, a detailed hyperparameter tuning session is conducted using GridSearchCV, aiming to find the optimal settings for each model that enhance their prediction accuracy.

Creating a Python Environment

  * conda create --name churn_prediction python=3.10.13
  * conda activate churn_prediction
  * conda install pip
  * pip install -r requirements.txt
