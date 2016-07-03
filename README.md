Here are the steps I took in order to prepare my best submission for the competition per https://inclass.kaggle.com/c/can-we-predict-voting-outcomes
- I removed outliers in the training set (by simple eliminating records where YOB > max(test$YOB) or YOB < min(test$YOB)
- I imputed demographic variables (but not questions) using mice, doing 3 imputation cycles for training and test data sets separately
- I introduced a new feature (AgeGroup, a factor with 6 categories) based on the value of YOB. I then deleted the original YOB variable from the data frame to be used in model training
- I limited the set of question predictor variables to only those that I found significant (both statistically and based on my subjective limited understanding of realities of the US political system). This translated to the set of questions below to participate as predictors in the classification models
- I applied a GBM prediction model (with a set of tuned hyper-parameters and AdaBoost exponential distribution of losses) to achieve the results. Hyper-parameters were selected with cross-validation, using standard cross-validation instruments provided by GBM package

*_Notes_*: 
- This was a binary classification problem
- Since the training data contained balanced observations (in terms of ratio of the class labels of the predicting variable), the best-fitting single-classifier model worked better than ensembles of various compositions

Alternatives that I also tried (with lower prediction accuracy) were as follows
- Applied other predictive models with limited success vs. the best submission above. Among the options were GBM with Bernoulli losses distribution, XGBOOST, glment, CART decision trees, random forests (both randomForest and Ranger implementations), old-generation neural networks (nnet) and modern deep-learning based neural network algorithms (several options within deep learning features of H2O were tried)
- Ensembles of models – I tried to apply a bunch of ensembles where GBM models were blended with a combination of other models mentioned above (except for H2O-driven models). The performance of ensembles appeared to be worse than one of GBM Adaboost (see notes above)
- Building AgeGroup as 5-level factor (this did not change model performance vs. 6-level factor, for the major prediction algorithms and model ensembles mentioned above)
- Imputing question answers (replacing “”) in addition to demographic variables – this always led to dropped prediction performance by 1-2% (depending on a model)
- Post-calibration of prediction results (using Platt technique) was implemented and applied. Post-calibration improved prediction performance of the majority of non-GBM predictions (especially for xgboost) yet the calibrated predictions still underperformed vs GBM submission with Adaboost. Calibration of GBM predictions did not add any edge

The interesting findings/lessons I learned are as follows
- Caret package is excellent tool to apply multiple models in a rapid and efficient way
- However, if you would like to engage really advanced options of a particular algorithm, it is worth trying direct use of the underlying package (for instance, GBM with Adaboost exponential distribution of losses was only available, if used GBM via gbm package)
- Random forests demonstrate better performance if they are provided with a smaller set of really significant independent variables
- The data set seems to be too small for xgboost – therefore xgboost overfitted to the training set (even in small levels of learning depth) and generally underperformed vs. gbm
