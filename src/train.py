import os 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

top_6_features = ["Tenure", "Complain", "CashbackAmount", "SatisfactionScore", "DaySinceLastOrder", 
                              "WarehouseToHome","Churn"]

top_8_features = ["Tenure", "Complain", "CashbackAmount", "SatisfactionScore", "DaySinceLastOrder", 
                              "WarehouseToHome", "OrderAmountHikeFromlastYear", "NumberOfAddress","Churn"]

top_10_features = ["Tenure", "Complain", "CashbackAmount", "SatisfactionScore", "DaySinceLastOrder", 
                   "WarehouseToHome", "OrderAmountHikeFromlastYear", "NumberOfAddress", "CouponUsed", "HourSpendOnApp","Churn"]


train_data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/references/model_data_train.csv')[top_6_features]

X = train_data.drop('Churn',axis=1)
y = train_data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_evaluate_tune_save_models():
    models = {
        'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
        'AdaBoost Classifier': AdaBoostClassifier(random_state=42)
    }
    
    param_grids = {
        'Gradient Boosting Classifier': {
            'n_estimators': [3,5,7,10,20,30,40,50,60,70,80,90,100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5,7,9,11,13,15,17,20],
            'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0]
        },
        'AdaBoost Classifier': {
            'n_estimators': [3,5,7,10,20,30,40,50,60,70,80,90,100],
            'learning_rate': [0.01, 0.1],
            'base_estimator__max_depth': [1, 2,3,4,5,6,7,8,9,10,],
        'algorithm': ['SAMME', 'SAMME.R']
        }
    }
    
    best_model = None
    highest_score = 0

    
    for name, model in models.items():
        print(f"\nTraining and Evaluating: {name}")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print("\nBefore Tuning:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model_grid = grid_search.best_estimator_
        best_score = grid_search.best_score_
        
        y_pred_tuned = best_model_grid.predict(X_test)
        
        print("\nAfter Tuning:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_tuned))
        print("Classification Report:")
        print(classification_report(y_test, y_pred_tuned))
        
        if best_score > highest_score:
            best_model = best_model_grid
            highest_score = best_score
    
    if best_model:
        with open('best_model.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        print(f"\nBest model saved: {best_model}")



