# Project Overview

This project is designed to provide a comprehensive analysis and optimization of ensemble learning models specifically targeted at churn prediction. By leveraging the strengths of different ensemble methods, including Gradient Boosting and AdaBoost, the project aims to refine predictive performance through systematic training, evaluation, and hyperparameter tuning.

Tools and Technologies
Python 3.10.13: The chosen programming language, ideal for data manipulation and machine learning tasks.
Scikit-learn: Provides robust tools for data mining and data analysis, including pre-built models and tuning techniques.
XGBoost & AdaBoost: Advanced ensemble techniques utilized for their strength in handling complex datasets and improving prediction outcomes.
Conda Environment: Ensures a controlled and consistent setup across all development and production phases, managing dependencies effectively.

Methodology
The process begins with preprocessing the input data, ensuring that it is clean and formatted correctly. This data is then used to train several models. Each model is first evaluated using standard metrics such as confusion matrices and classification reports to establish a baseline performance. Following this, a detailed hyperparameter tuning session is conducted using GridSearchCV, aiming to find the optimal settings for each model that enhance their prediction accuracy.

# Insights

### Findings from Univariate Analysis

* The tenure data shows a mean of 10.23 years and a mode of 1 year, indicating high tenure within the first year but some long-term retention
* The data indicates a strong preference for mobile devices ('Mobile Phone' and 'Phone') over computers for login, suggesting users are more mobile-oriented
*  The majority of users are from Tier 1 cities, indicating a potential concentration of user base in more developed urban areas.
*  The Warehouse To Home distance is predominantly short, with a mean at 15 and mode of 9 kilometers, suggesting most customers are concentrated near the warehouse, but a long tail to 127 kilometers indicates some far-off deliveries.
*  A significant majority of users spend 2 to 3 hours on the mobile application or website, indicating a core engagement window, with very few users spending very little (0-1 hours) or much more time (5 hours)
*  Most users have multiple devices registered on their accounts, with 3 to 4 devices being the most common, suggesting a high level of engagement and potential dependence on the service
*  The highest preference among users is for 'Laptop & Accessory' and 'Mobile Phone' categories, indicating a strong consumer focus on technology-related products over categories like fashion and groceries
*  Customer satisfaction scores are widely distributed, with a notable concentration at the mid-point (score 3.0), suggesting a significant portion of customers feel neutral about the service, while extremes (very satisfied or dissatisfied) are less common but still significant
*  A significant number of customers have multiple addresses linked to their accounts, with 2 to 3 addresses being the most common, suggesting flexibility in delivery options
*  The majority of customers have not registered a complaint (0.0), indicating overall satisfaction or fewer issues with the service, while a smaller yet significant portion have lodged a complaint (1.0), signaling areas for improvement.
*  The percentage increase in order value compared to last year has a mean of 15.74%, with most values clustering tightly around the median of 15%, suggesting a moderate but consistent growth across orders. The mode at 13% indicates the most frequent rate of increase experienced by customers.
* The distribution of coupons used by customers shows that a single coupon (1.0) is the most common, followed by two coupons (2.0), with usage tapering off significantly as the number of coupons increases.
* Most customers placed either 1 or 2 orders last month, with a significant drop in frequency as the number of orders increases, indicating that while a core group of customers orders frequently, most limit their purchases
* Large number of customers ordering very recently (within a few days), with a peak at 3 days since the last order. However, there are noticeable counts for customers whose last order was significantly longer ago, up to 31 and 46 days.
* The average cashback received last month is around 177.36, with a median of 163.38 and a mode of 123.42, indicating that most customers receive moderate cashback amounts
* The churn rate is relatively low, with 20% of customers (1.0) having churned and 80% (0.0) retained, indicating overall stability but highlighting a segment that requires attention to prevent further churn.
  
### Findings from Bivariate Analysis

* Customers who churn tend to have a significantly shorter tenure (average of 3.9 months) compared to those who don't (average of 11.5 months). Strategies to engage customers early in their lifecycle could be crucial in reducing churn and enhancing long-term customer retention
* Customers who prefer logging in via Mobile Phone have the highest retention rate, with the lowest proportion of churn compared to other devices. Conversely, customers using a Phone have the highest churn rate relative to their count.
* Customers in CityTier 1 have the highest absolute number of churns but a relatively lower churn rate compared to CityTier 3, which has a significantly higher churn rate despite fewer customers. Targeted retention strategies in CityTier 3 could be beneficial in reducing overall churn rates
* Customers who churn tend to have a slightly higher average (17.11) and median (15) distance from the warehouse to their homes compared to non-churners (mean of 15.42, median of 13). This suggests that longer delivery times may contribute to churn
* Customers using Debit Card and E-wallet have higher counts of churn (296 and 115 respectively) compared to other payment methods. This suggests that there might be issues or dissatisfactions associated with these payment modes
* Significant increase in churn rates for users spending 3 hours on the app, with 383 out of 2236 churning, indicating potential issues or dissatisfaction at this usage level. Lower usage (0, 1, and 5 hours) shows negligible to no churn, suggesting that either very light or very heavy users are more likely to stay. Optimizing app engagement for users in the 2-4 hours range might help reduce churn rates
* Churn rates increase significantly with the number of devices registered. Users with 4 devices registered show the highest numbers of churn (315 out of 1968), suggesting potential issues with multi-device management or user experience. Interestingly, as the number of devices increases to 6, although the base is smaller
* Combining the categories "Mobile" and "Mobile Phone" shows a total of 458 users churning out of 2408, making it the category with the highest absolute number of churns. This combined category significantly indicates dissatisfaction or service issues in mobile-related purchases.
* Notable trend where higher churn rates are observed with lower satisfaction scores. Customers with the highest satisfaction score (5) have a considerable churn number (200 out of 841), suggesting that even satisfied customers might have reasons to leave, possibly due to factors not captured by the satisfaction score alone. However, the highest churn is observed among those with a satisfaction score of 3 (214 out of 1260), indicating specific dissatisfaction at this level
* Customers with 2 and 3 addresses also exhibit relatively high churn rates—185 out of 1099 for 2 addresses, and 178 out of 1023 for 3 addresses. This pattern suggests a trend where having multiple addresses might be associated with higher churn
* The data highlights a significant difference in churn based on complaints: 408 out of 1269 customers who have registered complaints churn, which is a much higher rate compared to those who have not complained (350 out of 3290). This suggests that addressing the issues leading to complaints more effectively could substantially reduce churn
* The data reveals a pattern where churn generally decreases as the percentage increase in order amount from the previous year rises. For instance, at a lower hike of 11%, the churn rate is relatively higher (65 out of 302), whereas at a 26% hike, the churn is significantly lower (2 out of 27). This suggests that customers who experience modest increases in order amounts are more likely to remain loyal, possibly viewing these increases as reasonable or reflective of added value.
* Trend where churn rates generally decrease as the number of coupons used increases. For customers using no coupons, the churn rate is higher (154 out of 868), whereas those using more coupons, such as 14 and above, show no churn at all. This pattern indicates that engaging customers with more coupon offers could significantly enhance retention
* Customers who place fewer orders (1 or 2 per month) have relatively higher churn rates (260 out of 1442 for 1 order, and 294 out of 1693 for 2 orders) compared to those who place more frequent orders. As the number of orders increases beyond 2, the churn rate generally decreases, indicating stronger loyalty among more active customers.
* Churn rates are highest among customers with recent interactions (0 to 2 days since last order), suggesting potential dissatisfaction or unresolved issues shortly after their last transaction. The churn significantly decreases as the days since the last order increase, indicating that customers who do not engage for longer periods are more likely to stay, possibly due to resolved issues or satisfaction with their last interaction
* The data indicates that customers who do not churn receive a higher average and median cashback amount (180.70 and 166.05, respectively) compared to those who churn (160.59 and 149.62, respectively). This suggests that higher cashback amounts are associated with better customer retention. Enhancing cashback offers could potentially be a strategic approach to reducing churn, as it appears to positively influence customer loyalty and satisfaction.

### Findings from Multivariate Analysis

* Customers with higher Satisfaction Scores and longer Tenures tend to churn less. Combined "Mobile" users show the highest retention (mean tenure: 12.54 months, churn score: 3.33), while "Phone" users experience shorter tenures (8.75 months) and higher churn. Focusing on improving satisfaction and engagement for "Phone" users can significantly enhance retention.
* Cashback amounts correlate with lower churn, as non-churning customers receive an average cashback of 180.70 compared to 160.59 for churners. Interestingly, similar order amount hikes (15.75 vs. 15.71) show no significant impact on churn. However, customers farther from the warehouse exhibit higher churn (17.11 vs. 15.42), indicating that while cashback helps retention, it may not fully mitigate the dissatisfaction caused by longer delivery distances.
* Customers who complain more (average complaints: 0.54 for churners vs. 0.23 for non-churners) and have a slightly higher number of addresses (4.60 vs. 4.19) are more likely to churn. Additionally, churners have significantly shorter tenures (3.89 vs. 11.47 months), suggesting that complaints and address management issues accelerate churn, particularly for newer customers



# Model Inferencing for daily level usage


Creating a Python Environment

  * conda create --name churn_prediction python=3.10.13
  * conda activate churn_prediction
  * conda install pip
  * pip install -r requirements.txt

Call inference.py for live predictions and pass data locations 

if __name__ == "__main__":
    data_path = 'live_data.csv' 
    encoders_path = 'models/encoders.pkl'
    model_path = 'models/gmm.pkl'
    predictor = churn_predictor(data_path, encoders_path, model_path)
    predictions = predictor.predict()
    print(predictions)


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


The classification report showcases a model with strong predictive performance in distinguishing between non-churn (class 0.0) and churn (class 1.0) customers. The model achieved a precision of 0.92 for non-churn and 0.96 for churn, with recall rates of 0.96 and 0.92, respectively. Both classes have an F1-score of 0.94, indicating a balanced performance. The overall accuracy of the model is 94% across 1088 instances, with macro and weighted averages for precision, recall, and F1-score all at 0.94. This demonstrates the model’s reliability and effectiveness in predicting customer churn and non-churn outcomes.


