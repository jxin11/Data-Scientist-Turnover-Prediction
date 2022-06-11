library(dplyr)
library(DataExplorer)
library(mice)
library(VIM)
library(missForest)
library(caret)
library(devtools)
# install_github("cran/DMwR")
library(DMwR)
library(caTools)
library(e1071)
library(randomForest)
library(ROCR)
library(mlr)
library(xgboost)

## Read in train & test data

# Original Train
df_train <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train2_train.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
dfImputation_train <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_meanMode_train.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mice_train <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_mice_train.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mF_train <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_mF_train.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)

df_train$target = as.factor(df_train$target)
dfImputation_train$target = as.factor(dfImputation_train$target)
df_mice_train$target = as.factor(df_mice_train$target)
df_mF_train$target = as.factor(df_mF_train$target)

# Upsampling Train
df_trainbal <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train2_trainbal.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
dfImputation_trainbal <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_meanMode_trainbal.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mice_trainbal <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_mice_trainbal.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mF_trainbal <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_mF_trainbal.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)

df_trainbal$target = as.factor(df_trainbal$target)
dfImputation_trainbal$target = as.factor(dfImputation_trainbal$target)
df_mice_trainbal$target = as.factor(df_mice_trainbal$target)
df_mF_trainbal$target = as.factor(df_mF_trainbal$target)

# Test
df_test <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train2_test.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
dfImputation_test <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_meanMode_test.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mice_test <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_mice_test.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)
df_mF_test <- read.csv('C:/Users/GL63/OneDrive - Asia Pacific University/6. Applied Machine Learning/Assignment/aug_train_mF_test.csv', header = T, na.strings=c("NA","NaN", ""), stringsAsFactors=T)

dfImputation_test$target = as.factor(dfImputation_test$target)
df_mice_test$target = as.factor(df_mice_test$target)
df_mF_test$target = as.factor(df_mF_test$target)


## Function
rocxgboost <- function(model, test, targetCol, title){
  dtest <- xgb.DMatrix(data = as.matrix(test[ ,-which(names(test) %in% c("target"))]),
                       label=as.matrix(test[['target']]))
  dlabel <- as.matrix(test[['target']])
  
  Predict_ROC <- predict (model,dtest)
  # Predict_ROC <- ifelse (Predict_ROC > 0.5,Predict_ROC,1-Predict_ROC)
  
  pred = prediction(Predict_ROC, test[[targetCol]])
  perf = ROCR::performance(pred, "tpr", "fpr")
  #plot(perf, colorize = T)
  plot(perf,
       main = sprintf("ROC curve - %s",title),
       ylab = "Sensitivity",
       xlab = "1-Specificity")
  abline(a=0, b= 1, lty=2)
  
  # Area Under Curve
  auc = as.numeric(ROCR::performance(pred, "auc")@y.values)
  auc = round(auc, 4)
  print(paste("AUC: ", auc))
}




## Data Preparation for XGBoost

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, 
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)


#hyperparam tuning - GridSearch
xgb_grid = expand.grid(eta = c(0.3,0.1,0.05,0.01),
                       max_depth = c(2, 4, 6, 8, 10),
                       #gamma = c(0,1), when using default param the value are in sync
                       min_child_weight = c(0:10),
                       subsample = c(0.6,0.8,1),
                       colsample_bytree=c(0.6,0.8,1))

#A1 ================================================
#preparing matrix
dtrainA1 <- xgb.DMatrix(data = as.matrix(dfImputation_train[ ,-which(names(dfImputation_train) %in% c("target"))]),
                      label = as.matrix(dfImputation_train[['target']])) 
dtestA1 <- xgb.DMatrix(data = as.matrix(dfImputation_test[ ,-which(names(dfImputation_test) %in% c("target"))]),
                     label=as.matrix(dfImputation_test[['target']]))

# calculate the best nround for this model
set.seed(123)
xgbcvA1 <- xgb.cv(params = params, data = dtrainA1, nrounds = 100, nfold = 10, metrics = 'error', 
                  showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

# Analyse Model
xgbcvA1$best_iteration
xgbcv_logsA1 <- data.frame(xgbcvA1$evaluation_log)
# Train & Test Error Plot
plot(xgbcv_logsA1$iter, xgbcv_logsA1$train_error_mean,col='blue',ylim=c(0.17,0.23))
lines(xgbcv_logsA1$iter, xgbcv_logsA1$test_error_mean, col='red')
sprintf("Minimum Error in Test Set: %f at iter %f", 
        round(min(xgbcv_logsA1$test_error_mean),4), xgbcv_logsA1[xgbcv_logsA1$test_error_mean==min(xgbcv_logsA1$test_error_mean),'iter'])

# Construct model with best nround
set.seed(123)
xgbA1 <- xgb.train (params = params, data = dtrainA1, nrounds = xgbcvA1$best_iteration, watchlist = list(val=dtestA1,train=dtrainA1), 
                    print_every_n = 10, early_stopping_rounds = 20, maximize = F , eval_metric = "error")

# Prediction
xgbpredA1 <- predict(xgbA1,dtestA1)
xgbpredA1 <- ifelse (xgbpredA1 > 0.5,1,0)
confusionMatrix (as.factor(xgbpredA1), dfImputation_test[['target']])
rocxgboost(xgbA1, dfImputation_test, "target", "XGBoost A1")

# Hyperparam Tuning===============

# Create empty lists
lowest_error_listA1 = list()

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(xgb_grid)){
  set.seed(123)
  mdcv <- xgb.train(data=dtrainA1,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = xgb_grid$max_depth[row],
                    eta = xgb_grid$eta[row],
                    subsample = xgb_grid$subsample[row],
                    colsample_bytree = xgb_grid$colsample_bytree[row],
                    min_child_weight =xgb_grid$min_child_weight[row],
                    nrounds= 100,
                    eval_metric = "error",
                    early_stopping_rounds= 20,
                    print_every_n = 100,
                    watchlist = list(train= dtrainA1, val= dtestA1)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_listA1[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_dfA1 = do.call(rbind, lowest_error_listA1)

# Bind columns of accuracy values and random hyperparameter values
randomsearchA1 = cbind(lowest_error_dfA1, xgb_grid)

# Quickly display highest accuracy
max(randomsearchA1$`1 - min(mdcv$evaluation_log$val_error)`)

randomsearchA1 <- as.data.frame(randomsearchA1) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_error)`) %>%
  arrange(-val_acc)
randomsearchA1

set.seed(123)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearchA1[1,]$max_depth,
               eta = randomsearchA1[1,]$eta,
               subsample = randomsearchA1[1,]$subsample,
               colsample_bytree = randomsearchA1[1,]$colsample_bytree,
               min_child_weight = randomsearchA1[1,]$min_child_weight)
params
set.seed(123)
xgbcv_tunedA1 <- xgb.cv(params = params,
                        data = dtrainA1,
                        nrounds =100,
                        print_every_n = 10, nfold=10,
                        eval_metric = "error",
                        early_stopping_rounds = 20, gamma = 0.1,
                        watchlist = list(train= dtrainA1, val= dtestA1))
set.seed(123)
xgb_tunedA1 <- xgb.train(params = params,
                         data = dtrainA1,
                         nrounds =xgbcv_tunedA1$best_iteration,
                         print_every_n = 10,
                         eval_metric = "error",
                         #early_stopping_rounds = 20, 
                         gamma = 0.1,
                         watchlist = list(train= dtrainA1, val= dtestA1))
# Prediction
xgb_tunedpredA1 <- predict(xgb_tunedA1,dtestA1)
xgb_tunedpredA1 <- ifelse (xgb_tunedpredA1 > 0.5,1,0)
confusionMatrix (as.factor(xgb_tunedpredA1), dfImputation_test[['target']])
rocxgboost(xgb_tunedA1, dfImputation_test, "target", "XGBoost A1 (Tuned)")

saveRDS(xgb_tunedA1, "xgb_tunedA1.rds")
#==================================================

#A2 ================================================
#preparing matrix
dtrainA2 <- xgb.DMatrix(data = as.matrix(dfImputation_trainbal[ ,-which(names(dfImputation_trainbal) %in% c("target"))]),
                        label = as.matrix(dfImputation_trainbal[['target']])) 
dtestA2 <- xgb.DMatrix(data = as.matrix(dfImputation_test[ ,-which(names(dfImputation_test) %in% c("target"))]),
                       label=as.matrix(dfImputation_test[['target']]))

# calculate the best nround for this model
set.seed(123)
xgbcvA2 <- xgb.cv(params = params, data = dtrainA2, nrounds = 1000, nfold = 10, metrics = 'error', 
                  showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

# Analyse Model
xgbcvA2$best_iteration
xgbcv_logsA2 <- data.frame(xgbcvA2$evaluation_log)
# Train & Test Error Plot
plot(xgbcv_logsA2$iter, xgbcv_logsA2$train_error_mean,col='blue')#,ylim=c(0.17,0.10))
lines(xgbcv_logsA2$iter, xgbcv_logsA2$test_error_mean, col='red')
sprintf("Minimum Error in Test Set: %f at iter %f", 
        round(min(xgbcv_logsA2$test_error_mean),4), xgbcv_logsA2[xgbcv_logsA2$test_error_mean==min(xgbcv_logsA2$test_error_mean),'iter'])

# Construct model with best nround
set.seed(123)
xgbA2 <- xgb.train (params = params, data = dtrainA2, nrounds = xgbcvA2$best_iteration, watchlist = list(val=dtestA2,train=dtrainA2), 
                    print_every_n = 10, early_stopping_rounds = 20, maximize = F , eval_metric = "error")

# Prediction
xgbpredA2 <- predict(xgbA2,dtestA2)
xgbpredA2 <- ifelse (xgbpredA2 > 0.5,1,0)
confusionMatrix (as.factor(xgbpredA2), dfImputation_test[['target']])
rocxgboost(xgbA2, dfImputation_test, "target", "XGBoost A2")

# Hyperparam Tuning===============

# Create empty lists
lowest_error_listA2 = list()

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(xgb_grid)){
  set.seed(123)
  mdcv <- xgb.train(data=dtrainA2,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = xgb_grid$max_depth[row],
                    eta = xgb_grid$eta[row],
                    subsample = xgb_grid$subsample[row],
                    colsample_bytree = xgb_grid$colsample_bytree[row],
                    min_child_weight =xgb_grid$min_child_weight[row],
                    nrounds= 1000,
                    eval_metric = "error",
                    early_stopping_rounds= 20,
                    print_every_n = 100,
                    watchlist = list(train= dtrainA2, val= dtestA2)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_listA2[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_dfA2 = do.call(rbind, lowest_error_listA2)

# Bind columns of accuracy values and random hyperparameter values
randomsearchA2 = cbind(lowest_error_dfA2, xgb_grid)

# Quickly display highest accuracy
max(randomsearchA2$`1 - min(mdcv$evaluation_log$val_error)`)

randomsearchA2 <- as.data.frame(randomsearchA2) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_error)`) %>%
  arrange(-val_acc)
randomsearchA2

set.seed(123)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearchA2[1,]$max_depth,
               eta = randomsearchA2[1,]$eta,
               subsample = randomsearchA2[1,]$subsample,
               colsample_bytree = randomsearchA2[1,]$colsample_bytree,
               min_child_weight = randomsearchA2[1,]$min_child_weight)
params
set.seed(123)
xgbcv_tunedA2 <- xgb.cv(params = params,
                        data = dtrainA2,
                        nrounds =1000,
                        print_every_n = 100, nfold=10,
                        eval_metric = "error",
                        early_stopping_rounds = 20, gamma=0,
                        watchlist = list(train= dtrainA2, val= dtestA2))
set.seed(123)
xgb_tunedA2 <- xgb.train(params = params,
                         data = dtrainA2,
                         nrounds =xgbcv_tunedA2$best_iteration,
                         print_every_n = 100,
                         eval_metric = "error",
                         #early_stopping_rounds = 20, 
                         gamma = 0,
                         watchlist = list(train= dtrainA2, val= dtestA2))
# Prediction
xgb_tunedpredA2 <- predict(xgb_tunedA2,dtestA2)
xgb_tunedpredA2 <- ifelse (xgb_tunedpredA2 > 0.5,1,0)
confusionMatrix (as.factor(xgb_tunedpredA2), dfImputation_test[['target']])
rocxgboost(xgb_tunedA2, dfImputation_test, "target", "XGBoost A2 (Tuned)")

saveRDS(xgb_tunedA2, "xgb_tunedA2.rds")
#==================================================

#B1 ================================================
#preparing matrix
dtrainB1 <- xgb.DMatrix(data = as.matrix(df_mice_train[ ,-which(names(df_mice_train) %in% c("target"))]),
                        label = as.matrix(df_mice_train[['target']])) 
dtestB1 <- xgb.DMatrix(data = as.matrix(df_mice_test[ ,-which(names(df_mice_test) %in% c("target"))]),
                       label=as.matrix(df_mice_test[['target']]))

# calculate the best nround for this model
set.seed(123)
xgbcvB1 <- xgb.cv(params = params, data = dtrainB1, nrounds = 100, nfold = 10, metrics = 'error', 
                  showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

# Analyse Model
xgbcvB1$best_iteration
xgbcv_logsB1 <- data.frame(xgbcvB1$evaluation_log)
# Train & Test Error Plot
plot(xgbcv_logsB1$iter, xgbcv_logsB1$train_error_mean,col='blue',ylim=c(0.17,0.23))
lines(xgbcv_logsB1$iter, xgbcv_logsB1$test_error_mean, col='red')
sprintf("Minimum Error in Test Set: %f at iter %f", 
        round(min(xgbcv_logsB1$test_error_mean),4), xgbcv_logsB1[xgbcv_logsB1$test_error_mean==min(xgbcv_logsB1$test_error_mean),'iter'])

# Construct model with best nround
set.seed(123)
xgbB1 <- xgb.train (params = params, data = dtrainB1, nrounds = xgbcvB1$best_iteration, watchlist = list(val=dtestB1,train=dtrainB1), 
                    print_every_n = 10, early_stopping_rounds = 20, maximize = F , eval_metric = "error")

# Prediction
xgbpredB1 <- predict(xgbB1,dtestB1)
xgbpredB1 <- ifelse (xgbpredB1 > 0.5,1,0)
confusionMatrix (as.factor(xgbpredB1), df_mice_test[['target']])
rocxgboost(xgbB1, df_mice_test, "target", "XGBoost B1")

# Hyperparam Tuning===============

# Create empty lists
lowest_error_listB1 = list()

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(xgb_grid)){
  set.seed(123)
  mdcv <- xgb.train(data=dtrainB1,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = xgb_grid$max_depth[row],
                    eta = xgb_grid$eta[row],
                    subsample = xgb_grid$subsample[row],
                    colsample_bytree = xgb_grid$colsample_bytree[row],
                    min_child_weight =xgb_grid$min_child_weight[row],
                    nrounds= 100,
                    eval_metric = "error",
                    early_stopping_rounds= 20,
                    print_every_n = 100,
                    watchlist = list(train= dtrainB1, val= dtestB1)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_listB1[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_dfB1 = do.call(rbind, lowest_error_listB1)

# Bind columns of accuracy values and random hyperparameter values
randomsearchB1 = cbind(lowest_error_dfB1, xgb_grid)

# Quickly display highest accuracy
max(randomsearchB1$`1 - min(mdcv$evaluation_log$val_error)`)

randomsearchB1 <- as.data.frame(randomsearchB1) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_error)`) %>%
  arrange(-val_acc)
randomsearchB1

set.seed(123)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearchB1[1,]$max_depth,
               eta = randomsearchB1[1,]$eta,
               subsample = randomsearchB1[1,]$subsample,
               colsample_bytree = randomsearchB1[1,]$colsample_bytree,
               min_child_weight = randomsearchB1[1,]$min_child_weight)
params
set.seed(123)
xgbcv_tunedB1 <- xgb.cv(params = params,
                        data = dtrainB1,
                        nrounds =100,
                        print_every_n = 10, nfold=10,
                        eval_metric = "error",
                        early_stopping_rounds = 20, gamma = 0.4,
                        watchlist = list(train= dtrainB1, val= dtestB1))
set.seed(123)
xgb_tunedB1 <- xgb.train(params = params,
                         data = dtrainB1,
                         nrounds =xgbcv_tunedB1$best_iteration,
                         print_every_n = 10,
                         eval_metric = "error",
                         early_stopping_rounds = 20, 
                         gamma = 0.4,
                         watchlist = list(train= dtrainB1, val= dtestB1))
# Prediction
xgb_tunedpredB1 <- predict(xgb_tunedB1,dtestB1)
xgb_tunedpredB1 <- ifelse (xgb_tunedpredB1 > 0.5,1,0)
confusionMatrix (as.factor(xgb_tunedpredB1), df_mice_test[['target']])
rocxgboost(xgb_tunedB1, df_mice_test, "target", "XGBoost B1 (Tuned)")

saveRDS(xgb_tunedB1, "xgb_tunedB1.rds")
#==================================================

#B2 ================================================
#preparing matrix
dtrainB2 <- xgb.DMatrix(data = as.matrix(df_mice_trainbal[ ,-which(names(df_mice_trainbal) %in% c("target"))]),
                        label = as.matrix(df_mice_trainbal[['target']])) 
dtestB2 <- xgb.DMatrix(data = as.matrix(df_mice_test[ ,-which(names(df_mice_test) %in% c("target"))]),
                       label=as.matrix(df_mice_test[['target']]))

# calculate the best nround for this model
set.seed(123)
xgbcvB2 <- xgb.cv(params = params, data = dtrainB2, nrounds = 1000, nfold = 10, metrics = 'error', 
                  showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

# Analyse Model
xgbcvB2$best_iteration
xgbcv_logsB2 <- data.frame(xgbcvB2$evaluation_log)
# Train & Test Error Plot
plot(xgbcv_logsB2$iter, xgbcv_logsB2$train_error_mean,col='blue')#,ylim=c(0.17,0.23))
lines(xgbcv_logsB2$iter, xgbcv_logsB2$test_error_mean, col='red')
sprintf("Minimum Error in Test Set: %f at iter %f", 
        round(min(xgbcv_logsB2$test_error_mean),4), xgbcv_logsB2[xgbcv_logsB2$test_error_mean==min(xgbcv_logsB2$test_error_mean),'iter'])

# Construct model with best nround
set.seed(123)
xgbB2 <- xgb.train (params = params, data = dtrainB2, nrounds = xgbcvB2$best_iteration, watchlist = list(val=dtestB2,train=dtrainB2), 
                    print_every_n = 10, early_stopping_rounds = 20, maximize = F , eval_metric = "error")

# Prediction
xgbpredB2 <- predict(xgbB2,dtestB2)
xgbpredB2 <- ifelse (xgbpredB2 > 0.5,1,0)
confusionMatrix (as.factor(xgbpredB2), df_mice_test[['target']])
rocxgboost(xgbB2, df_mice_test, "target", "XGBoost B2")

# Hyperparam Tuning===============

# Create empty lists
lowest_error_listB2 = list()

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(xgb_grid)){
  set.seed(123)
  mdcv <- xgb.train(data=dtrainB2,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = xgb_grid$max_depth[row],
                    eta = xgb_grid$eta[row],
                    subsample = xgb_grid$subsample[row],
                    colsample_bytree = xgb_grid$colsample_bytree[row],
                    min_child_weight =xgb_grid$min_child_weight[row],
                    nrounds= 1000,
                    eval_metric = "error",
                    early_stopping_rounds= 20,
                    print_every_n = 100,
                    watchlist = list(train= dtrainB2, val= dtestB2)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_listB2[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_dfB2 = do.call(rbind, lowest_error_listB2)

# Bind columns of accuracy values and random hyperparameter values
randomsearchB2 = cbind(lowest_error_dfB2, xgb_grid)

# Quickly display highest accuracy
max(randomsearchB2$`1 - min(mdcv$evaluation_log$val_error)`)

randomsearchB2 <- as.data.frame(randomsearchB2) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_error)`) %>%
  arrange(-val_acc)
randomsearchB2

set.seed(123)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearchB2[1,]$max_depth,
               eta = randomsearchB2[1,]$eta,
               subsample = randomsearchB2[1,]$subsample,
               colsample_bytree = randomsearchB2[1,]$colsample_bytree,
               min_child_weight = randomsearchB2[1,]$min_child_weight)
params
set.seed(123)
xgbcv_tunedB2 <- xgb.cv(params = params,
                        data = dtrainB2,
                        nrounds =1000,
                        print_every_n = 100, nfold=10,
                        eval_metric = "error",
                        early_stopping_rounds = 20, gamma = 0.6,
                        watchlist = list(train= dtrainB2, val= dtestB2))
set.seed(123)
xgb_tunedB2 <- xgb.train(params = params,
                         data = dtrainB2,
                         nrounds =xgbcv_tunedB2$best_iteration,
                         print_every_n = 10,
                         eval_metric = "error",
                         early_stopping_rounds = 20, 
                         gamma = 0.6,
                         watchlist = list(train= dtrainB2, val= dtestB2))
# Prediction
xgb_tunedpredB2 <- predict(xgb_tunedB2,dtestB2)
xgb_tunedpredB2 <- ifelse (xgb_tunedpredB2 > 0.5,1,0)
confusionMatrix (as.factor(xgb_tunedpredB2), df_mice_test[['target']])
rocxgboost(xgb_tunedB2, df_mice_test, "target", "XGBoost B2 (Tuned)")

saveRDS(xgb_tunedB2, "xgb_tunedB2.rds")
#==================================================

#C1 ================================================
#preparing matrix
dtrainC1 <- xgb.DMatrix(data = as.matrix(df_mF_train[ ,-which(names(df_mF_train) %in% c("target"))]),
                        label = as.matrix(df_mF_train[['target']])) 
dtestC1 <- xgb.DMatrix(data = as.matrix(df_mF_test[ ,-which(names(df_mF_test) %in% c("target"))]),
                       label=as.matrix(df_mF_test[['target']]))

# calculate the best nround for this model
set.seed(123)
xgbcvC1 <- xgb.cv(params = params, data = dtrainC1, nrounds = 100, nfold = 10, metrics = 'error', 
                  showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

# Analyse Model
xgbcvC1$best_iteration
xgbcv_logsC1 <- data.frame(xgbcvC1$evaluation_log)
# Train & Test Error Plot
plot(xgbcv_logsC1$iter, xgbcv_logsC1$train_error_mean,col='blue',ylim=c(0.14,0.22))
lines(xgbcv_logsC1$iter, xgbcv_logsC1$test_error_mean, col='red')
sprintf("Minimum Error in Test Set: %f at iter %f", 
        round(min(xgbcv_logsC1$test_error_mean),4), xgbcv_logsC1[xgbcv_logsC1$test_error_mean==min(xgbcv_logsC1$test_error_mean),'iter'])

# Construct model with best nround
set.seed(123)
xgbC1 <- xgb.train (params = params, data = dtrainC1, nrounds = xgbcvC1$best_iteration, watchlist = list(val=dtestC1,train=dtrainC1), 
                    print_every_n = 10, early_stopping_rounds = 20, maximize = F , eval_metric = "error")

# Prediction
xgbpredC1 <- predict(xgbC1,dtestC1)
xgbpredC1 <- ifelse (xgbpredC1 > 0.5,1,0)
confusionMatrix (as.factor(xgbpredC1), df_mF_test[['target']])
rocxgboost(xgbC1, df_mF_test, "target", "XGBoost C1")

# Hyperparam Tuning===============

# Create empty lists
lowest_error_listC1 = list()

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(xgb_grid)){
  set.seed(123)
  mdcv <- xgb.train(data=dtrainC1,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = xgb_grid$max_depth[row],
                    eta = xgb_grid$eta[row],
                    subsample = xgb_grid$subsample[row],
                    colsample_bytree = xgb_grid$colsample_bytree[row],
                    min_child_weight =xgb_grid$min_child_weight[row],
                    nrounds= 100,
                    eval_metric = "error",
                    early_stopping_rounds= 20,
                    print_every_n = 100,
                    watchlist = list(train= dtrainC1, val= dtestC1)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_listC1[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_dfC1 = do.call(rbind, lowest_error_listC1)

# Bind columns of accuracy values and random hyperparameter values
randomsearchC1 = cbind(lowest_error_dfC1, xgb_grid)

# Quickly display highest accuracy
max(randomsearchC1$`1 - min(mdcv$evaluation_log$val_error)`)

randomsearchC1 <- as.data.frame(randomsearchC1) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_error)`) %>%
  arrange(-val_acc)
randomsearchC1

set.seed(123)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearchC1[1,]$max_depth,
               eta = randomsearchC1[1,]$eta,
               subsample = randomsearchC1[1,]$subsample,
               colsample_bytree = randomsearchC1[1,]$colsample_bytree,
               min_child_weight = randomsearchC1[1,]$min_child_weight)
params
set.seed(123)
xgbcv_tunedC1 <- xgb.cv(params = params,
                        data = dtrainC1,
                        nrounds =100,
                        print_every_n = 10, nfold=10,
                        eval_metric = "error",
                        early_stopping_rounds = 20, gamma = 0.6,
                        watchlist = list(train= dtrainC1, val= dtestC1))
set.seed(123)
xgb_tunedC1 <- xgb.train(params = params,
                         data = dtrainC1,
                         nrounds =xgbcv_tunedC1$best_iteration,
                         print_every_n = 10,
                         eval_metric = "error",
                         early_stopping_rounds = 20, 
                         gamma = 0.6,
                         watchlist = list(train= dtrainC1, val= dtestC1))
# Prediction
xgb_tunedpredC1 <- predict(xgb_tunedC1,dtestC1)
xgb_tunedpredC1 <- ifelse (xgb_tunedpredC1 > 0.5,1,0)
confusionMatrix (as.factor(xgb_tunedpredC1), df_mF_test[['target']])
rocxgboost(xgb_tunedC1, df_mF_test, "target", "XGBoost C1 (Tuned)")

saveRDS(xgb_tunedC1, "xgb_tunedC1.rds")
#==================================================

#C2 ================================================
#preparing matrix
dtrainC2 <- xgb.DMatrix(data = as.matrix(df_mF_trainbal[ ,-which(names(df_mF_trainbal) %in% c("target"))]),
                        label = as.matrix(df_mF_trainbal[['target']])) 
dtestC2 <- xgb.DMatrix(data = as.matrix(df_mF_test[ ,-which(names(df_mF_test) %in% c("target"))]),
                       label=as.matrix(df_mF_test[['target']]))

# calculate the best nround for this model
set.seed(123)
xgbcvC2 <- xgb.cv(params = params, data = dtrainC2, nrounds = 1000, nfold = 10, metrics = 'error', 
                  showsd = T, stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)

# Analyse Model
xgbcvC2$best_iteration
xgbcv_logsC2 <- data.frame(xgbcvC2$evaluation_log)
# Train & Test Error Plot
plot(xgbcv_logsC2$iter, xgbcv_logsC2$train_error_mean,col='blue')#,ylim=c(0.14,0.22))
lines(xgbcv_logsC2$iter, xgbcv_logsC2$test_error_mean, col='red')
sprintf("Minimum Error in Test Set: %f at iter %f", 
        round(min(xgbcv_logsC2$test_error_mean),4), xgbcv_logsC2[xgbcv_logsC2$test_error_mean==min(xgbcv_logsC2$test_error_mean),'iter'])

# Construct model with best nround
set.seed(123)
xgbC2 <- xgb.train (params = params, data = dtrainC2, nrounds = xgbcvC2$best_iteration, watchlist = list(val=dtestC2,train=dtrainC2), 
                    print_every_n = 10, early_stopping_rounds = 20, maximize = F , eval_metric = "error")

# Prediction
xgbpredC2 <- predict(xgbC2,dtestC2)
xgbpredC2 <- ifelse (xgbpredC2 > 0.5,1,0)
confusionMatrix (as.factor(xgbpredC2), df_mF_test[['target']])
rocxgboost(xgbC2, df_mF_test, "target", "XGBoost C2")

# Hyperparam Tuning===============

# Create empty lists
lowest_error_listC2 = list()

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(xgb_grid)){
  set.seed(123)
  mdcv <- xgb.train(data=dtrainC2,
                    booster = "gbtree",
                    objective = "binary:logistic",
                    max_depth = xgb_grid$max_depth[row],
                    eta = xgb_grid$eta[row],
                    subsample = xgb_grid$subsample[row],
                    colsample_bytree = xgb_grid$colsample_bytree[row],
                    min_child_weight =xgb_grid$min_child_weight[row],
                    nrounds= 1000,
                    eval_metric = "error",
                    early_stopping_rounds= 20,
                    print_every_n = 100,
                    watchlist = list(train= dtrainC2, val= dtestC2)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_error))
  lowest_error_listC2[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_dfC2 = do.call(rbind, lowest_error_listC2)

# Bind columns of accuracy values and random hyperparameter values
randomsearchC2 = cbind(lowest_error_dfC2, xgb_grid)

# Quickly display highest accuracy
max(randomsearchC2$`1 - min(mdcv$evaluation_log$val_error)`)

randomsearchC2 <- as.data.frame(randomsearchC2) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_error)`) %>%
  arrange(-val_acc)
randomsearchC2

set.seed(123)
params <- list(booster = "gbtree", 
               objective = "binary:logistic",
               max_depth = randomsearchC2[1,]$max_depth,
               eta = randomsearchC2[1,]$eta,
               subsample = randomsearchC2[1,]$subsample,
               colsample_bytree = randomsearchC2[1,]$colsample_bytree,
               min_child_weight = randomsearchC2[1,]$min_child_weight)
params
set.seed(123)
xgbcv_tunedC2 <- xgb.cv(params = params,
                        data = dtrainC2,
                        nrounds =1000,
                        print_every_n = 100, nfold=10,
                        eval_metric = "error",
                        early_stopping_rounds = 20, #gamma = 0.1,
                        watchlist = list(train= dtrainC2, val= dtestC2))
set.seed(123)
xgb_tunedC2 <- xgb.train(params = params,
                         data = dtrainC2,
                         nrounds =xgbcv_tunedC2$best_iteration,
                         print_every_n = 100,
                         eval_metric = "error",
                         #early_stopping_rounds = 20, 
                         #gamma = 0.1,
                         watchlist = list(train= dtrainC2, val= dtestC2))
# Prediction
xgb_tunedpredC2 <- predict(xgb_tunedC2,dtestC2)
xgb_tunedpredC2 <- ifelse (xgb_tunedpredC2 > 0.5,1,0)
confusionMatrix (as.factor(xgb_tunedpredC2), df_mF_test[['target']])
rocxgboost(xgb_tunedC2, df_mF_test, "target", "XGBoost C2 (Tuned)")

saveRDS(xgb_tunedC2, "xgb_tunedC2.rds")
#==================================================


# load the model
xgb_tunedA1 <- readRDS("xgb_tunedA1.rds")
xgb_tunedA2 <- readRDS("xgb_tunedA2.rds")
xgb_tunedB1 <- readRDS("xgb_tunedB1.rds")
xgb_tunedB2 <- readRDS("xgb_tunedB2.rds")
xgb_tunedC1 <- readRDS("xgb_tunedC1.rds")
xgb_tunedC2 <- readRDS("xgb_tunedC2.rds")

imp <- xgb.importance(colnames(dtrainC1),model=xgb_tunedC1)
xgb.plot.importance(imp,)


