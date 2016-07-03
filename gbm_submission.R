# This module contains the script to train the GBM
# prediction model for Kaggle competion to
# predict the US President Election voting results 
# among the users of Show of Hands application
# (see https://inclass.kaggle.com/c/can-we-predict-voting-outcomes)
#
# Author: George Vyshnya
# Date: May 26 - Jun 16, 2016
#
# Summary: 
# - submission of prediction results of GBM model with tuned parameters
# - GBM model invoked via caret interface
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
	result <- 1 # under 25
	current.year <- 2016
	years.diff <- abs(current.year - x)
	if (years.diff < 25) {result <- 1}
	if (years.diff >= 25 & years.diff < 36) {result <- 2}
	if (years.diff >= 36 & years.diff < 49) {result <- 3}
	if (years.diff >= 49 & years.diff < 64) {result <- 4}
	if (years.diff >= 64 ) {result <- 5}
	result
}
# submission output file name
output.csv.file <- "Submission_gbm.csv"

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

set.seed(825)

# prediction threshold
threshold <- 0.5

nrow(train)

# remove outliers
yob.min <- 1900
yob.max <- 2003
train <- subset (train, YOB >= yob.min & YOB <= yob.max)
# size after removing outliers
nrow(train)

# further data preparations
party <- train$Party
end_trn <- nrow(train)
length(party)
end_trn

# impute missing data
library(mice)
#start imputation by reading data with different options, for inputation reasons
train.imputed <- read.csv("train2016.csv", na.strings=c("NA","NaN", " ", ""))
test.imputed <- read.csv("test2016.csv", na.strings=c("NA","NaN", " ", ""))

#remove outliers from this replica of training data, too
train.imputed  <- subset (train.imputed, YOB >= yob.min & YOB <= yob.max)
# size after removing outliers
nrow(train.imputed)


# limit the set of fields to impute
train.imputed <- select(train.imputed, USER_ID, Party, YOB, Income, Gender, 
             HouseholdStatus, EducationLevel)
test.imputed <- select(test.imputed, USER_ID, YOB, Income, Gender, 
             HouseholdStatus, EducationLevel)

nrow(train.imputed)
nrow(test.imputed)

# impute missing data - train set
tempData <- mice(train.imputed,m=3,maxit=2,meth='pmm',seed=825)
train.imputed <- complete(tempData, 3)

# impute missing data - test set
tempData <- mice(test.imputed,m=3,maxit=2,meth='pmm',seed=825)
test.imputed<- complete(tempData, 3)

# re-assign imputed values back to the original data sets
train$YOB <- train.imputed$YOB
train$Income <- train.imputed$Income
train$HouseholdStatus <- train.imputed$HouseholdStatus
train$Gender <- train.imputed$Gender
train$EducationLevel <- train.imputed$EducationLevel

test$YOB <- test.imputed$YOB
test$Income <- test.imputed$Income
test$HouseholdStatus <- test.imputed$HouseholdStatus
test$Gender <- test.imputed$Gender
test$EducationLevel <- test.imputed$EducationLevel

# prepare subsets with essential variables to do imputation
train <- select(train,
     USER_ID, Party, YOB, Income, HouseholdStatus, Gender, EducationLevel,
     Q96024, Q98197, Q98078, Q98869, Q99480, Q99716,
     Q100680, Q100689, Q101163, Q102089, Q105840, Q106042,
     Q106272, Q108342, Q108617, Q109244, Q110740, Q111220,
     Q113181, Q115899, Q115611, Q116881, Q118232, Q119851,
     Q120014, Q120472, Q120650, Q120379, Q121700, Q123464,
     Q110740 )

test <- select(test,
     USER_ID, YOB, Income, HouseholdStatus, Gender, EducationLevel,
     Q96024, Q98197, Q98078, Q98869, Q99480, Q99716,
     Q100680, Q100689, Q101163, Q102089, Q105840, Q106042,
     Q106272, Q108342, Q108617, Q109244, Q110740, Q111220,
     Q113181, Q115899, Q115611, Q116881, Q118232, Q119851,
     Q120014, Q120472, Q120650, Q120379, Q121700, Q123464,
     Q110740 
)

trainLimited <- select(train, -Party)

# combine test and training data into one data set for ease of manipulation
all <- rbind(trainLimited,test)
end_trn <- nrow(trainLimited)
end <- nrow(all)
end_trn
end


# add a new factor column for YOB : -20, 20-30, 30-40, 40-50, 50-60, 60+
all$AgeGroup <- as.factor(sapply(all$YOB, function (x) YoB2AgeGroup(x) ))

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
# Modelling and predictions: GBM
#######################################################

set.seed(825)
# Number of folds
fitControl <- trainControl(method = "repeatedcv", classProbs = TRUE, 
       number = 4, repeats = 1)

formula <- as.formula(Party ~ .)

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

TestPredictions.full <- gbm2.test.pred #calibrated prediction
p.test <- p.gbm2.test #non-calibrated prediction
#  
TrainPredictions.full <- gbm2.train.pred

threshold <- 0.5
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
ROCRpred.calib <- prediction(gbm2.test.pred, validate.tr$Party)
as.numeric(performance(ROCRpred.calib, "auc")@y.values)

# Performance function
ROCRperf <- performance(ROCRpred.calib, "tpr", "fpr")

# Plot ROC curve with colors
plot(ROCRperf, colorize=TRUE)

# Add threshold labels 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))



# AUC analysis, training data
# prediction object
ROCRpred.train <- prediction(TrainPredictions.full, train.tr$Party)

# Performance function
ROCRperf <- performance(ROCRpred.train, "tpr", "fpr")

# Plot ROC curve with colors
plot(ROCRperf, colorize=TRUE)

# Add threshold labels 
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

#######################################################
# make predictions
#######################################################
p.gbm2.final <- predict(tr, newdata=test.test, type="prob")

# do Platt calibration
p.final.calibrated <- calibrate.prediction (partyCalibBinary,
       pred.to.calibrate, p.gbm2.final)

TestPredictions <- p.final.calibrated
# Output submission for the simple logistic regression model
PredTestLabels <- as.factor(ifelse(TestPredictions < threshold, "Republican", "Democrat"))
MySubmission <- data.frame(USER_ID = testOriginal$USER_ID, Predictions = PredTestLabels)
write.csv(MySubmission, output.csv.file, row.names=FALSE)
# That's all!
########################################
