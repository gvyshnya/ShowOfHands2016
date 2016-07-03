# This module contains the script to train the GLMNET
# prediction model for Kaggle competion to
# predict the US President Election voting results 
# among the users of Show of Hands application
# (see https://inclass.kaggle.com/c/can-we-predict-voting-outcomes)
#
# Author: George Vyshnya
# Date: May 26 - Jun 16, 2016
#
# Summary: 
# - submission of prediction results of GLMNET model with tuned parameters
# - GLMNET model invoked via caret interface
# - post-processing of prediction results with Platt calibration applied

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
output.csv.file <- "Submission_glmnet.csv"

# read data
train <- read.csv("train2016.csv", na.strings=c("NA","NaN", " "))
test <- read.csv("test2016.csv", na.strings=c("NA","NaN", " "))
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


# remove outliers
yob.min <- 1900
yob.max <- 2003
train <- subset (train, YOB >= yob.min & YOB <= yob.max)

# further data preparations
party <- train$Party
end_trn <- nrow(train)

#start imputation
train.imputed <- train
test.imputed <- test

# impute missing data - test set
tempData <- mice(test.imputed,m=3,maxit=2,meth='pmm',seed=500)
test.imputed<- complete(tempData, 3)

train <- train.imputed
test <- test.imputed
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
spl <- sample.split(train$Party, SplitRatio = 0.7)

train_callibrate.tr <- subset(training.df.xgboost, spl == TRUE)
validate.tr <- subset(training.df.xgboost, spl == FALSE)

# create a callibration set
set.seed(825)
spl2 <- sample.split(train_callibrate.tr$Party, SplitRatio = 0.9)
train.tr <- subset(train_callibrate.tr, spl2 == TRUE)
callibrate.tr <- subset(train_callibrate.tr, spl2 == FALSE)

# set a nominal name to the test set to predict against
test.test <- all[(end_trn+1):end,]

# reference to the limited test set in all: all[(end_trn+1):end,]
# reference to the original training set in all: all[1:end_trn,]

# convert party data into 0-1 sequence vector to apply for gradient boost 
# algorithms
partyBinary <- sapply(party, function(x) is.democrat(x))
# turn calibrate's party values into 0-1 vector
partyCalibBinary <- sapply(callibrate.tr$Party, function(x) is.democrat(x))


#######################################################
# Modelling and predictions: GLMNET
#######################################################
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

TestPredictions.full <- glmnet1.test.pred #calibrated prediction
p.test <- p.glmnet1.test #non-calibrated prediction
#  
TrainPredictions.full <- glmnet1.train.pred

# confusion matrix in percentale form - full set, callibrated
prop.table(table(TestPredictions.full >threshold,validate.tr$Party))

# confusion matrix in percentale form - full set, non-callibrated
prop.table(table(p.test[,1] >threshold,validate.tr$Party)) #p.test [,1]


# Train set prediction
# confusion matrix in percentale form - training set
prop.table(table(TrainPredictions.full >threshold,train.tr$Party))

# Test set AUC - non-callibrated
ROCRpred <- prediction(p.test[,1], validate.tr$Party)
as.numeric(performance(ROCRpred, "auc")@y.values)

# Test set AUC - callibrated
ROCRpred.calib <- prediction(glmnet1.test.pred, validate.tr$Party)
as.numeric(performance(ROCRpred.calib, "auc")@y.values)

#######################################################
# make predictions
#######################################################
p.glmnet1.final <- predict(tr, newdata=test.test, type="prob")

# do Platt calibration
p.final.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, p.glmnet1.final)

TestPredictions <- p.final.calibrated
# Output submission for the simple logistic regression model
PredTestLabels <- as.factor(ifelse(TestPredictions < threshold, "Republican", "Democrat"))
MySubmission <- data.frame(USER_ID = testOriginal$USER_ID, Predictions = PredTestLabels)
write.csv(MySubmission, output.csv.file, row.names=FALSE)
# That's all!
########################################
