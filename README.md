## Abstract
The project aimed to find a release date/window of an animation movie that maximizes its revenue using machine learning algorithms. 
Animation movies have a separate demographic and the number of audiences who are watching animation movies is increasing by year. 
Selecting an appropriate release date is important to ensure the film does not compete directly with other major releases, as simultaneous premieres often dilute 
box office earnings for both movies involved. To avoid this, we trained models that can predict revenue and created an algorithm to  identify the best release date for
a movie that has a high chance of maximizing the revenue. We trained different regression algorithms like linear regression, support vector regressor, decision tree regressor,
and KNN regressor to fit the data. We studied which model was able to study the underlying pattern in this data. We also check seasonality in the release dates which gave us a
better understanding of which season has maximum revenue. Further we also applied ensemble techniques to improve our accuracy. By trying out all these models we wanted to 
identify which model works best. We checked how well our system works using the test set to see if the model is solving the problem defined.

## Dataset
The dataset we are using for this project is a kaggle dataset named “Animation Movies”. 51945 rows and 23 columns. 
### Data Pre-processing:
The pre-processing began with initial data cleaning by removing zero revenue values (missing information) and dropping irrelevant metadata columns. 
The release_date, one of the crucial variable for analysis, was divided into temporal features like release_date, release_month, release_year. 
Using these three feature, a new feature release_season is created through feature engineering. 

## Methodologies
model best fit the data and had the best results.
### Linear Models:
Simple regression, Lasso regression, and Ridge regression were employed to evaluate their performance on the dataset. The effectiveness of each model was measured using standard performance metrics such as R-squared, Mean Absolute Error (MAE), and Mean Squared Error (MSE). This evaluation helped in comparing their accuracy and predictive capabilities.
### Non-linear Models:
Models such as Decision Trees, Support Vector Machines (SVM), and K-Nearest Neighbors (KNN) were also tested. Extensive hyperparameter tuning was conducted through GridSearchCV, which systematically explored a wide range of hyperparameters to select the best combination based on cross-validation results, ensuring optimal model configuration tailored to the dataset.
### Ensemble Methods:
Ensemble methods, including Random Forest and Elastic Net, were utilized to assess their performance. These methods often enhance accuracy by aggregating the 
### Artificial Neural Network:
Artificial Neural Networks (ANN) were included in our analysis to model complex non-linear relationships inherent in the data. 
### Optimal Date Prediction & Time Series Analysis:
For our set of test data, the model predicts the optimal release date for each movie sample in the test dataset, considering the revenue predictions across different dates within the specified year.  

## Testing Results

### Linear Regression
Adjusted R2 : 43.99%
MSE : 0.555
### Support Vector Regressor
Adjusted R2 : 72.58%
MSE : 0.274
### Decision Tree Regressor
Adjusted R2 : 100%
MSE : 0.000
### K nearest Neighbors Regressor
Adjusted R2 : 100%
MSE : 0.000
### Random Forest Regressor
Adjusted R2 : 96.30%
MSE : 0.095
### Ridge Regression
Adjusted R2 : 41.89%
MSE : 0.581
### Lasso Regression 
Adjusted R2 : 0.000%
MSE : 1.000
### Elastic Net
Adjusted R2 : 41.15%
MSE : 0.405
### ANN
Adjusted R2 : 71.95%
MSE : 0.3654

## Conclusion 
The Random forest model proved to be most accurate in capturing the relationship between different variables and revenue of animation movie in the test set. Due to its high R squared we proceeded to use this model to predict on the test set. We found the R-squared on testing data to be 82.94%, MAE: 0.220557 and MSE: 0.194233. The model still performs well, but this indicates some overfitting. Because of this, we chose to predict with a second model, ANN, with the second best R squared. We found the R-squared on testing data to be 71.95%, MAE: 0.2924 and MSE: 0.3897. This model did not show signs of overfitting, Further study can include predicting dates using ANN which is complex.. The seasonality and feature importance was not significantly different between the models.The general overfitting is most likely due to the large amount of observations we needed to remove due to missing revenue values. We suggest in the future, additional methods should be applied to fill in this missing data to reduce overfitting. The seasonality analysis revealed significant differences in revenue across seasons, highlighting the importance of considering release timing in maximizing the revenue. 

