# This module contains the script to train the ensemble of
# prediction models for Kaggle competion to
# predict the US President Election voting results 
# among the users of Show of Hands application
# (see https://inclass.kaggle.com/c/can-we-predict-voting-outcomes)
#
# Author: George Vyshnya
# Date: May 26 - Jun 16, 2016
#
# Summary: 
# - submission of equal-weight ensemble prediction results
# - prediction results of individual models callibrated by Platt method
#   before composing the ensemble submission


library(caret)
library(caTools)
library(plyr)
library(dplyr)
library(car)
library(Matrix)
library(e1071)
library(vcd)
library(xgboost)
library(nnet)
library(gbm)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(ranger)
library(glmnet)
library(ggplot2)
library(pROC)

# this function applies Platt calibration
# to the original class probability prediction of 
# a classification predictor model
# Ref. and ideas: http://danielnee.com/tag/platt-scaling/
#    - bin.results.vector - vector  of 0-1 values
#          describing the actual value of a var on the
#          calibration data set
#    - pred - the prediction object obtained from predict(...)
#             on the calibration data set
#    - pred.2.calibrate - the prediction obtained on the
#             test data set (it will be callibrated as a result)
calibrate.prediction <- function (bin.results.vector, pred, pred.2.calibrate) 
{
	calib.data.frame <- data.frame(cbind(bin.results.vector, pred))
	colnames(calib.data.frame) <- c("y", "x")
	calib.model <- glm(y ~ x, calib.data.frame, family=binomial)
	calib.data.frame <- data.frame(pred.2.calibrate)
	colnames(calib.data.frame) <- c("x")
	pred.calibrated <- predict(calib.model, newdata=calib.data.frame, 
                type="response")
	pred.calibrated
}

detect.liberals <- function (df) {
	result.df <- df

	for(i in 1:nrow(result.df)) {
    		row <- result.df[i,]
    		# do stuff with row
		if (row$Q109244 == "Yes")
		{
			row$IsLiberal <- "Yes"
		}
		else if (row$Q115611 == "Yes")
		{
			row$IsLiberal <- "No"
		}
		else if (row$Q113181 == "Yes")
		{
			row$IsLiberal <- "No"
		}
		else if (row$Q113181 == "No" & row$Q115611 == "No" & row$Q109244 == "No")
		{
			row$IsLiberal <- "Yes"
		}
		result.df[i,] <- row
	}

	result.df
}

is.democrat <- function (x) {
	result <- 0
	if (x == "Democrat") {result <- 1}
	result
}

YoB2AgeGroup <- function (x) {
	result <- 1 # under 20
	current.year <- 2016
	years.diff <- abs(current.year - x)
	if (years.diff < 20) {result <- 1}
	if (years.diff >= 20 & years.diff < 30) {result <- 2}
	if (years.diff >= 30 & years.diff < 40) {result <- 3}
	if (years.diff >= 40 & years.diff < 50) {result <- 4}
	if (years.diff >= 50 & years.diff < 60) {result <- 5}
	if (years.diff >= 60 ) {result <- 6}
	result
}
# submission output file name
output.csv.file <- "Submission_ensemble1.csv"

# read data
train <- read.csv("train2016.csv", na.strings=c("NA","NaN", " ", ""))
test <- read.csv("test2016.csv", na.strings=c("NA","NaN", " ", ""))
str(train)
str(test)

# make the copies of the original train and test datasets
trainOriginal <- train
testOriginal <- test

# baseline accuracy
table(train$Party)
# baseline is predicting Democrate
accuracy.baseline <- 2951 / (2951 + 2617)
accuracy.baseline # 0.5299928

set.seed(123)

# prediction threshold
threshold <- 0.5

# impute missing data - training set
library(mice)
tempData <- mice(train,m=3,maxit=2,meth='pmm',seed=500)

# complete the final data inputation into train
train <- complete(tempData, 3)

# impute missing data - test set
tempData <- mice(test,m=3,maxit=2,meth='pmm',seed=500)
test <- complete(tempData, 3)

# remove outliers
yob.min <- min(test$YOB)
yob.max <- max(test$YOB)
train <- subset (train, YOB >= yob.min & YOB <= yob.max)

# further data preparations
party <- train$Party
end_trn <- nrow(train)

trainLimited <- select(train, -Party)

# combine test and training data into one data set for ease of manipulation
all <- rbind(trainLimited,test)
end <- nrow(all)

# add a new factor column for YOB : -20, 20-30, 30-40, 40-50, 50-60, 60+
all$AgeGroup <- sapply(all$YOB, function (x) YoB2AgeGroup(x) )

# add a new factor column to indicate possible liberal values of a respondent
all$IsLiberal <- factor(levels = c("Yes", "No", "Unclear"), 
      x=rep(c("Unclear"), times = nrow(all)))
all <- detect.liberals (all)

# delete USER_ID and YOB as they won't help much in the forecasts
all <- select (all, -USER_ID)
all <- select (all, -YOB)

# split the original training set into subsets for true training and 
# out-of-sample validation
library(caTools)

#a special training data set to fit caret train interface as well as 
# formula interfaces of some other major packages
training.df.xgboost <- all[1:end_trn,]
training.df.xgboost$Party <- party   #partyBinary

# create a callibration set out of the original training set
set.seed(825)
spl <- sample.split(train$Party, SplitRatio = 0.9)

train.tr <- subset(training.df.xgboost, spl == TRUE)
callibrate.tr <- subset(training.df.xgboost, spl == FALSE)

# set a nominal name to the test set to predict against
validate.tr <- all[(end_trn+1):end,]

# reference to the limited test set in all: all[(end_trn+1):end,]
# reference to the original training set in all: all[1:end_trn,]

# convert party data into 0-1 sequence vector to apply for gradient boost 
# algorithms
partyBinary <- sapply(party, function(x) is.democrat(x))
# turn calibrate's party values into 0-1 vector
partyCalibBinary <- sapply(callibrate.tr$Party, function(x) is.democrat(x))


#######################################################
# Modelling and predictions: Ensemble
#######################################################

#---------------------------------------
# 1. GBM model #1
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", classProbs = TRUE, 
       number = 4, repeats = 1)
formula <- as.formula(Party ~.)

gbmGrid <-  expand.grid(n.trees=5000, interaction.depth=4, 
          shrinkage=0.001, n.minobsinnode=20)
tr <- train(formula, data = train.tr, method = "gbm", trControl = fitControl, 
          tuneGrid = gbmGrid,
          metric = "ROC", preProc = c("center", "scale"))

p.gbm1.train <- predict(tr, newdata=train.tr, type="prob")
p.gbm1.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.gbm1.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.gbm1.callibrate[,1]
pred.being.calibrated <- p.gbm1.test[,1]

# do Platt calibration
p.gbm1.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
gbm1.test.pred <- p.gbm1.test.calibrated 
gbm1.train.pred <- p.gbm1.train[,1]

#---------------------------------------
# 2. GBM model #2
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", classProbs = TRUE, 
       number = 4, repeats = 1)

formula <- as.formula(Party ~ Q98197+Q109244+Q112512+Q113181+Q115611+
Q99480+HouseholdStatus+ Q101163+ Q118232)

gbmGrid <-  expand.grid(n.trees=5000, interaction.depth=4, 
          shrinkage=0.001, n.minobsinnode=20)
tr <- train(formula, data = train.tr, method = "gbm", trControl = fitControl, 
          tuneGrid = gbmGrid,
          metric = "ROC", preProc = c("center", "scale"))

p.gbm2.train <- predict(tr, newdata=train.tr, type="prob")
p.gbm2.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.gbm2.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.gbm2.callibrate[,1]
pred.being.calibrated <- p.gbm2.test[,1]

# do Platt calibration
p.gbm2.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
gbm2.test.pred <- p.gbm2.test.calibrated 
gbm2.train.pred <- p.gbm2.train[,1]

#---------------------------------------
# 3. XGBOOST model #1
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 15, repeats = 3)
formula <- as.formula(Party ~ .)
#IsLiberal+Q98197)

gbmGrid <-  expand.grid(nrounds = 4, max_depth = 4, eta=0.1, gamma=0, 
         colsample_bytree=0.9, min_child_weight=0)
set.seed(825)
tr <- train(formula, data = train.tr, method = "xgbTree", 
trControl = fitControl
, tuneGrid = gbmGrid
)

p.xgboost1.train <- predict(tr, newdata=train.tr, type="prob")
p.xgboost1.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.xgboost1.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.xgboost1.callibrate[,1]
pred.being.calibrated <- p.xgboost1.test[,1]

# do Platt calibration
p.xgboost1.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
xgboost1.test.pred <- p.xgboost1.test.calibrated 
xgboost1.train.pred <- p.xgboost1.train[,1]

#---------------------------------------
# 4. XGBOOST model #2
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 15, repeats = 3)
formula <- as.formula(Party ~ IsLiberal+Q98197)

gbmGrid <-  expand.grid(nrounds = 4, max_depth = 4, eta=0.1, gamma=0, 
         colsample_bytree=0.9, min_child_weight=0)
set.seed(825)
tr <- train(formula, data = train.tr, method = "xgbTree", 
trControl = fitControl
, tuneGrid = gbmGrid
)

p.xgboost2.train <- predict(tr, newdata=train.tr, type="prob")
p.xgboost2.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.xgboost2.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.xgboost2.callibrate[,1]
pred.being.calibrated <- p.xgboost2.test[,1]

# do Platt calibration
p.xgboost2.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
xgboost2.test.pred <- p.xgboost2.test.calibrated 
xgboost2.train.pred <- p.xgboost2.train[,1]

#---------------------------------------
# 5. NNET model #1
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats=1, 
         returnResamp = "all",classProbs = TRUE, 
         summaryFunction = twoClassSummary)

formula <- as.formula(Party ~ .)

set.seed(825)
tr <- train(formula, data = train.tr, 
                 method = "nnet", 
                 preProcess = "range", 
                 tuneLength = 4, trControl = fitControl,
                 trace = FALSE,
                 maxit = 10)

p.nnet1.train <- predict(tr, newdata=train.tr, type="prob")
p.nnet1.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.nnet1.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.nnet1.callibrate[,1]
pred.being.calibrated <- p.nnet1.test[,1]

# do Platt calibration
p.nnet1.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
nnet1.test.pred <- p.nnet1.test.calibrated 
nnet1.train.pred <- p.nnet1.train[,1]

#---------------------------------------
# 5. NNET model #1
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats=1, 
         returnResamp = "all",classProbs = TRUE, 
         summaryFunction = twoClassSummary)

formula <- as.formula(Party ~ .)

set.seed(825)
tr <- train(formula, data = train.tr, 
                 method = "nnet", 
                 preProcess = "range", 
                 tuneLength = 4, trControl = fitControl,
                 trace = FALSE,
                 maxit = 10)

p.nnet1.train <- predict(tr, newdata=train.tr, type="prob")
p.nnet1.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.nnet1.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.nnet1.callibrate[,1]
pred.being.calibrated <- p.nnet1.test[,1]

# do Platt calibration
p.nnet1.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
nnet1.test.pred <- p.nnet1.test.calibrated 
nnet1.train.pred <- p.nnet1.train[,1]

#---------------------------------------
# 5. NNET model #1
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats=1, 
         returnResamp = "all",classProbs = TRUE, 
         summaryFunction = twoClassSummary)

formula <- as.formula(Party ~ .)

set.seed(825)
tr <- train(formula, data = train.tr, 
                 method = "nnet", 
                 preProcess = "range", 
                 tuneLength = 4, trControl = fitControl,
                 trace = FALSE,
                 maxit = 10)

p.nnet1.train <- predict(tr, newdata=train.tr, type="prob")
p.nnet1.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.nnet1.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.nnet1.callibrate[,1]
pred.being.calibrated <- p.nnet1.test[,1]

# do Platt calibration
p.nnet1.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
nnet1.test.pred <- p.nnet1.test.calibrated 
nnet1.train.pred <- p.nnet1.train[,1]

#---------------------------------------
# 6. NNET model #2
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats=1, 
         returnResamp = "all",classProbs = TRUE, 
         summaryFunction = twoClassSummary)

formula <- as.formula(Party ~ .)

set.seed(825)
tr <- train(formula, data = train.tr, 
                 method = "nnet", 
                 preProcess = "range", 
                 tuneLength = 4, trControl = fitControl,
                 trace = FALSE,
                 maxit = 10)

p.nnet2.train <- predict(tr, newdata=train.tr, type="prob")
p.nnet2.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.nnet2.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.nnet1.callibrate[,1]
pred.being.calibrated <- p.nnet1.test[,1]

# do Platt calibration
p.nnet2.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
nnet2.test.pred <- p.nnet2.test.calibrated 
nnet2.train.pred <- p.nnet2.train[,1]

#---------------------------------------
# 7. GLMNET model #1
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 3, repeats=1, 
         returnResamp = "all",classProbs = TRUE, 
summaryFunction = twoClassSummary)

formula <- as.formula(Party ~ .)

gbmGrid <- expand.grid(alpha = seq(.05, 1, length = 15),
                       lambda = c((1:5)/10))

set.seed(825)
tr <- train(formula, data = train.tr, method = "glmnet", 
          trControl = fitControl, tuneGrid = gbmGrid,
          preProc = c("center", "scale"))

p.glmnet1.train <- predict(tr, newdata=train.tr, type="prob")
p.glmnet1.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.glmnet1.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.glmnet1.callibrate[,1]
pred.being.calibrated <- p.glmnet1.test[,1]

# do Platt calibration
p.glmnet1.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
glmnet1.test.pred <- p.glmnet1.test.calibrated 
glmnet1.train.pred <- p.glmnet1.train[,1]

#---------------------------------------
# 8. GLMNET model #2
#---------------------------------------
set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", number = 3, repeats=1, 
         returnResamp = "all",classProbs = TRUE, 
summaryFunction = twoClassSummary)

formula <- as.formula(Party ~ Q98197+Q109244+Q112512+Q113181+Q115611+
Q99480+HouseholdStatus+ Q101163+ Q118232)

gbmGrid <- expand.grid(alpha = seq(.05, 1, length = 15),
                       lambda = c((1:5)/10))

set.seed(825)
tr <- train(formula, data = train.tr, method = "glmnet", 
          trControl = fitControl, tuneGrid = gbmGrid,
          preProc = c("center", "scale"))

p.glmnet2.train <- predict(tr, newdata=train.tr, type="prob")
p.glmnet2.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.glmnet2.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.glmnet2.callibrate[,1]
pred.being.calibrated <- p.glmnet2.test[,1]

# do Platt calibration
p.glmnet2.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
glmnet2.test.pred <- p.glmnet2.test.calibrated 
glmnet2.train.pred <- p.glmnet2.train[,1]

#---------------------------------------
# 9. RF model #1
#---------------------------------------
set.seed(825)

formula <- as.formula(Party ~ IsLiberal+Q116881+Q109244+Q98197)

set.seed(825)
tr <- randomForest(formula, data=train.tr)

p.rf1.train <- predict(tr, newdata=train.tr, type="prob")
p.rf1.callibrate <- predict(tr, newdata=callibrate.tr, type="prob")
p.rf1.test <- predict(tr, newdata=validate.tr, type="prob")

pred.to.calibrate <- p.rf1.callibrate[,1]
pred.being.calibrated <- p.rf1.test[,1]

# do Platt calibration
p.rf1.test.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, pred.being.calibrated)

# Validate 
rf1.test.pred <- p.rf1.test.calibrated 
rf1.train.pred <- p.rf1.train[,1]



#######################################################
# make the ensemble prediction
#######################################################
output.csv.file <- "Submission_ensemble.csv"
n.models <- 9
# 
TestPredictions <- ( 
rf1.test.pred +
glmnet2.test.pred +
glmnet1.test.pred +
nnet2.test.pred +
nnet1.test.pred +
xgboost1.test.pred +
xgboost2.test.pred +
gbm2.test.pred +
gbm1.test.pred
) / n.models

#   
TrainPredictions <- ( 
rf1.train.pred +
glmnet2.train.pred +
glmnet1.train.pred +
nnet2.train.pred +
nnet1.train.pred +
xgboost1.train.pred +
xgboost2.train.pred +
gbm2.train.pred +
gbm1.train.pred
) / n.models

### End of modelling and prediction ######

#### Movel verification and output #######

# round the predictions to zero and one
v.testp <- as.numeric(as.vector(TestPredictions))
v.trainp <- as.numeric(as.vector(TrainPredictions))
TestPredictions.rounded <- round(v.testp)
TrainPredictions.rounded <- round(v.trainp)

# compare training predictions and actual values fit
head(TrainPredictions.rounded, n = 20)
head(partyBinary, n = 20)

# in-sample classification (prediction) accuracy
countOfMatchingResults <- length(which(TrainPredictions.rounded == partyBinary))
countOfMatchingResults 
insample.arrucary <- countOfMatchingResults/length(TrainPredictions)
insample.arrucary

# Output submission for the simple logistic regression model
PredTestLabels <- as.factor(ifelse(v.testp < threshold, "Republican", "Democrat"))
MySubmission <- data.frame(USER_ID = testOriginal$USER_ID, Predictions = PredTestLabels)
write.csv(MySubmission, output.csv.file, row.names=FALSE)
# That's all!
########################################
