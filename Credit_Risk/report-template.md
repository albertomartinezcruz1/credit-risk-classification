# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge.

This analysis has the purpose of categorizing the credit risk based on the given data.The data was on a credit risk clasification table, and we need to predict whether the credit will be healthy or it has a high risk of defaulting, the we will analyze the precision, accuracy, and recall.

Out of all the data 75036 loans were healthy and 2500 were on high risk of defaulting, the information tables has info on: loan_size, interest_rate,	borrower_income,	debt_to_income,	num_of_accounts,	derogatory_marks and	total_debt, the information has two labels (0 healthy credit 1 high risk of defaulting)

The steps we follow to solve this are:
-Split the Data into Training and Testing Sets using the train_test_split.
-Create a Logistic Regression Model with the Original Data and then use Random over sampling data with Logistic Regression.

To solve this problem we used two methods:
-Logistic regression: This is a helpful tool when you want to predict binary results such as this case, basically this method is based on a threshold of probability so above or below this threshold the machine decides whether it will be a good crredit or a posible default.
-Random over sampling and logistic regression: This ones is used to increase the number of minority class this should improve accuracy, precicion and recall.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Logistic regression:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  The balanced accuracy score is 0.95, wich means how many times the model was correct out of all the observations.
  The precision was 0.92 wich tell us that there are very little false positive, in this case the false positive is high risk of defaulting, the formula for precision is:
  TP/(TP+FP)
  Recall was 0.95 this tell us about a low false negative rate the formula is:
  TP/(TP+FN)


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  The balanced accuracy score is 0.9945026387334079, this is almost perfect accuracy
  The precision was 0.99 this means that there were almost none false positives (false high risk of defaulting)
  Recall was  0.99 this means that there is almost none false negatives either (False healthy credit)
## Summary

-Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

*The best method by far is Logistic regression using resampled data, we can see that if performs better in accuracy, precision and recall, this method is almost perfect so i strongly recommend to use it with this sort of data.

-Does performance depend on the problem we are trying to solve? (For example, is it more important 
to predict the `1`'s, or predict the `0`'s? )

* Yes, performance depends on the problem for this case it is better to focus on predicting 1 correctly because we want to know the possible default credit correctly beforehand.


If you do not recommend any of the models, please justify your reasoning.
