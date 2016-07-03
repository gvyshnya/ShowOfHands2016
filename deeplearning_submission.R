# This module contains the script to train the H2O deep learning
# prediction model for Kaggle competion to
# predict the US President Election voting results 
# among the users of Show of Hands application
# (see https://inclass.kaggle.com/c/can-we-predict-voting-outcomes)
#
# Author: George Vyshnya
# Date: May 26 - Jun 16, 2016
#
# Summary: 
# - submission of prediction results of H2O DL model with tuned parameters


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
library(h2o)

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
train.imputed <- train        #read.csv("train2016.csv", na.strings=c("NA","NaN", " ", ""))
test.imputed <- test          #read.csv("test2016.csv", na.strings=c("NA","NaN", " ", ""))

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

# combine train and test data set into a single one to produce
# feature engineering and prediction of missing values on the
# essential questions

party <- train$Party

trainLimited <- select(train, -Party)
testLimited <- test
trainLimited$FromTrain = TRUE
testLimited$FromTrain = FALSE 


# combine test and training data into one data set for ease of manipulation
all <- rbind(trainLimited,testLimited )
end_trn <- nrow(trainLimited)
end <- nrow(all)
end_trn
end


# add a new factor column for YOB 
all$AgeGroup <- as.factor(sapply(all$YOB, function (x) YoB2AgeGroup(x) ))

# add a new factor column to indicate possible liberal values of a respondent
#all$IsLiberal <- factor(levels = c("Yes", "No", "Unclear"), 
#      x=rep(c("Unclear"), times = nrow(all)))
#all <- detect.liberals (all)

#delete YOB
all <- select (all, -YOB)

# predict missing values for Q109244
model.pmv.df <- select(all, USER_ID, AgeGroup, Income, HouseholdStatus, Gender, EducationLevel, Q109244)
model.pmv.df.train <- subset(model.pmv.df, is.na(Q109244) == FALSE)
model.pmv.df.test <- subset(model.pmv.df, is.na(Q109244) == TRUE)

set.seed(825)
# Number of folds
tr.control = trainControl(method = "cv", number = 10)
cp.grid = expand.grid( .cp = (0:10)*0.001)
tr <- train(Q109244 ~.-USER_ID, data = model.pmv.df.train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
best.tree <- tr$finalModel
prp(best.tree)

pred.missing <- predict(tr, newdata=model.pmv.df.test)
model.pmv.df.test$Q109244 <- pred.missing

model.pmv.df <- rbind(model.pmv.df.train, model.pmv.df.test)

model.pmv.df[order(model.pmv.df$USER_ID),]
all[order(all$USER_ID),]
all$Q109244 <- model.pmv.df$Q109244 

# predict missing values for Q98197
model.pmv.df <- select(all, USER_ID, AgeGroup, Income, HouseholdStatus, Gender, EducationLevel, Q109244, Q98197)
model.pmv.df.train <- subset(model.pmv.df, is.na(Q98197) == FALSE)
model.pmv.df.test <- subset(model.pmv.df, is.na(Q98197) == TRUE)

set.seed(825)
# Number of folds
tr.control = trainControl(method = "cv", number = 10)
cp.grid = expand.grid( .cp = (0:10)*0.001)
tr <- train(Q98197 ~.-USER_ID, data = model.pmv.df.train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
best.tree <- tr$finalModel
prp(best.tree)

pred.missing <- predict(tr, newdata=model.pmv.df.test)
model.pmv.df.test$Q98197 <- pred.missing

model.pmv.df <- rbind(model.pmv.df.train, model.pmv.df.test)

model.pmv.df[order(model.pmv.df$USER_ID),]
all[order(all$USER_ID),]
all$Q98197<- model.pmv.df$Q98197

# prepare subsets with essential variables to do imputation
train <- select(train,
     USER_ID, Party, YOB, Income, HouseholdStatus, Gender, EducationLevel,
     Q98197, Q98869, Q99480, Q109244, Q113181, Q115611, Q116881
)

test <- select(test,
     USER_ID, YOB, Income, HouseholdStatus, Gender, EducationLevel,
     Q98197, Q98869, Q99480, Q109244, Q113181, Q115611, Q116881
)

train.imputed <- subset(all, FromTrain == TRUE)
test.imputed <- subset(all, FromTrain == FALSE)

train.imputed[order(train.imputed$USER_ID),]
train[order(train$USER_ID),]
test.imputed[order(test.imputed$USER_ID),]
test[order(test$USER_ID),]

train$Q98197 <- train.imputed$Q98197
train$Q109244 <- train.imputed$Q109244
test$Q98197 <- test.imputed$Q98197
test$Q109244 <- test.imputed$Q109244

#plot(train$Q109244, train$Party)
#plot(train$Q98197, train$Party)
#plot(train$Q98869, train$Party)
#plot(train$Q99480, train$Party)
#plot(train$Q113181, train$Party)
#plot(train$Q115611, train$Party)
#plot(train$Q116881, train$Party)

# Inspecting:

#-plot(train$Q96024, train$Party)
#-plot(train$Q98078, train$Party)
#-plot(train$Q100680, train$Party)
#-plot(train$Q106042, train$Party)
#-plot(train$Q108342, train$Party)
#-plot(train$Q108617, train$Party)
#-plot(train$Q110740, train$Party)
#-plot(train$Q111220, train$Party)
#-plot(train$Q120014, train$Party)
#-plot(train$Q120650, train$Party)
#-plot(train$Q121700, train$Party)
#-plot(train$Q123464, train$Party)

#-+plot(train$Q110740, train$Party)
#-+plot(train$Q120379, train$Party)
#-+plot(train$Q120472, train$Party)
#-+plot(train$Q119851, train$Party)
#-+plot(train$Q118232, train$Party)
#-+plot(train$Q115899, train$Party)
#-+plot(train$Q106272, train$Party)
#-+plot(train$Q105840, train$Party)
#-+plot(train$Q102089, train$Party)
#-+plot(train$Q101163, train$Party)
#-+plot(train$Q100689, train$Party)
#-+plot(train$Q99716, train$Party)

#######################################
#plot(train$Q109244, train$Party)
#plot(train$Q98197, train$Party)
#plot(train$Q98869, train$Party)
#plot(train$Q99480, train$Party)
#plot(train$Q113181, train$Party)
#plot(train$Q115611, train$Party)
#plot(train$Q116881, train$Party)
#prop.table(table(test$Q109244))
#prop.table(table(train$Q109244))

#df.noQ109244 <- subset(test, test$Q109244 == "")
#nrow(df.noQ109244)
#plot(df.noQ109244$Q98197)

trainLimited <- select(train, -Party)
testLimited <- test

# combine test and training data into one data set for ease of manipulation
all <- rbind(trainLimited,testLimited )
end_trn <- nrow(trainLimited)
end <- nrow(all)
end_trn
end


# delete USER_ID and YOB as they won't help much in the forecasts
all <- select (all, -USER_ID)


#a special training data set to fit model interfaces
training.df.xgboost <- all[1:end_trn,]
training.df.xgboost$Party <- party   #partyBinary

# set a nominal name to the test set to predict against
test.test <- all[(end_trn+1):end,]

# serialize pre-processed data sets to disk to prepare inputs to h2o
train.csv.file <- "train_processed.csv"
test.csv.file <- "test_processed.csv"
write.csv(training.df.xgboost, train.csv.file, row.names=TRUE)
write.csv(test.test, test.csv.file, row.names=TRUE)

nrow(training.df.xgboost)
nrow(test.test)

##########################################################
# Deap Learning Model Run
##########################################################

#### Start H2O
#Start up a 1-node H2O server on your local machine, and allow it to use all CPU cores and up to 2GB of memory:
#
h2o.init(nthreads=-1, max_mem_size="2G")
h2o.removeAll() ## clean slate - just in case the cluster was already running

df.train <- h2o.importFile(path = normalizePath(train.csv.file))
dim(df.train)
nrow(as.data.frame(df.train))

df.test <- h2o.importFile(path = normalizePath(test.csv.file))
dim(df.test)
nrow(as.data.frame(df.test))

par(mfrow=c(1,1)) # reset canvas
plot(h2o.tabulate(df.train, "Q109244", "Party"))
set.seed(825)

#Split the original train data set: 
# - 60% for training, 
# - 20% for validation (hyper parameter tuning) 
# - 20% for final testing of the model
#
splits <- h2o.splitFrame(df.train, c(0.6,0.2), seed=825)
train.tr  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid.tr  <- h2o.assign(splits[[2]], "valid.hex") # 20%
test.tr   <- h2o.assign(splits[[3]], "test.hex")  # 20%

response <- "Party" # prediction target
predictors <- setdiff(names(df.train), response)

  
#The simplest hyperparameter search method is a brute-force scan of the full Cartesian product of all 
# combinations specified by a grid search:
#
hyper_params <- list(
  hidden=list(c(256,512)),
  input_dropout_ratio=c(0.4),
  rate=c(0.004,0.005),
  rate_annealing=c(1e-8,1e-7,1e-6)
)
hyper_params
grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id="dl_grid", 
  training_frame=train.tr,
  validation_frame=valid.tr, 
  x=predictors, 
  y=response,
  epochs=10,
  stopping_metric="misclassification",
  stopping_tolerance=1e-2,        ## stop when misclassification does not improve by >=1% for 2 scoring events
  stopping_rounds=10,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## if F, then manually tuned learning rate
  momentum_start=0.5,             ## manually tuned momentum
  momentum_stable=0.9, 
  momentum_ramp=1e7, 
  l1=1e-5,
  l2=1e-5,
  activation=c("Tanh"),  #Rectifier
  max_w2=5,                      ## can help improve stability for Rectifier
  hyper_params=hyper_params
)
grid
#                                
#Let's see which model had the lowest validation error:
#
grid <- h2o.getGrid("dl_grid",sort_by="err",decreasing=FALSE)
grid

## To see what other "sort_by" criteria are allowed
#grid <- h2o.getGrid("dl_grid",sort_by="wrong_thing",decreasing=FALSE)

## Sort by min logloss, not validation error (err) or AUR
h2o.getGrid("dl_grid",sort_by="err",decreasing=FALSE)

## Find the best model and its full set of parameters
grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]])
best_model

print(best_model@allparameters)
print(h2o.performance(best_model, valid=T))
print(h2o.logloss(best_model, valid=T))

m3 <- best_model 
plot(h2o.performance(m3)) 

#
#Let's compare the training error with the validation and test set errors
#
h2o.performance(m3, train=T)          ## sampled training data (from model building)
h2o.performance(m3, valid=T)          ## sampled validation data (from model building)
h2o.performance(m3, newdata=train.tr)    ## full training data
h2o.performance(m3, newdata=valid.tr)    ## full validation data
h2o.performance(m3, newdata=test.tr)     ## full test data

# make predictions
df.valid.tr <- as.data.frame(valid.tr)
df.test.tr <- as.data.frame(test.tr)
pred.calib <- h2o.predict(m3, valid.tr)
pred <- h2o.predict(m3, test.tr)
pred
test.tr$Accuracy <- as.numeric(pred$predict == test.tr$Party)
pred.accuracy <- mean(test.tr$Accuracy)
pred.accuracy

df.valid.tr$PartyBin <- sapply(df.valid.tr$Party, function(x) is.democrat(x))
CalibProbability <- calibrate.prediction (df.valid.tr$PartyBin, as.vector(pred.calib[,2]), as.vector(pred[,2])) 

CalibDemocrate <- sapply(CalibProbability, function(x) if (x > threshold) {"Democrat"} else {"Republican"})
df.test.tr$CalibAccuracy <- as.numeric(CalibDemocrate == df.test.tr$Party)
pred.calib.accuracy <- mean(df.test.tr$CalibAccuracy)
pred.calib.accuracy

# make the final predictions 
pred.final <- h2o.predict(m3, df.test)
pred.final

CalibProbability <- calibrate.prediction (df.valid.tr$PartyBin, as.vector(pred.calib[,2]), as.vector(pred.final[,2])) 

#######################################################
# make the submission
#######################################################
threshold <- 0.5
output.csv.file <- "Submission_dl.csv"

TestPredictions <- pred.final
df <- as.data.frame(TestPredictions)
nrow(df)
# Output submission for the simple logistic regression model
# PredTestLabels <- as.factor(ifelse(df[,2] < threshold, "Republican", "Democrat"))
PredTestLabels <- as.factor(ifelse(CalibProbability < threshold, "Republican", "Democrat"))

MySubmission <- data.frame(USER_ID = testOriginal$USER_ID)
MySubmission$USER_ID = testOriginal$USER_ID

#MySubmission$Predictions = df$predict
MySubmission$Predictions = PredTestLabels
write.csv(MySubmission, output.csv.file, row.names=FALSE)

#### All done, shutdown H2O
# h2o.shutdown(prompt=FALSE)

########################################
# That's all!
########################################
