# _Scores and Accuracy of the different model_

1. Linear Regression using sklearn
	* Single variable
		- Sqrt(MSE)   = 2.72725e5
		- Score       = 0.5032

2. Logistic Regression using sklearn
	* Binary Logistic Reg.
		- Accuracy     = 0.8125
		- Precision    = 0.7391
		- Recall       = 0.6538
		- AUC          = 0.8475
	* Multivariate Logistic Reg.
    		- Accuracy     = 0.9774
		- AUC (macro)  = 0.9645

3. Ridge and Lasso Regression
	* Ridge Reg.
		-test score (low alpha)   = 0.7146
		-test score (high alpha)  = 0.6805
	* Lasso Reg.
		-test score (low alpha)   =  0.6641
		-test score (high alpha)  =  0.7318

4. K-Nearest Neighbors Model
	- k = 10
	- Score = 0.9667

5. Random Forest Classifier
	- Score = 1.0

6. Decision Tree Classifier
	- Score = 1.0

7. XGboost Model 
	- Accuracy = 0.7402 

8. Support Vector Machines Model
	- Accuracy = 1.0
