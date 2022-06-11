###############################################
#             LOGISTIC REGRESSION             #
###############################################

# install_github("cran/DMwR")
library(dplyr)
library(DataExplorer)
library(mice)
library(VIM)
library(missForest)
library(caret)
library(devtools)
library(DMwR)
library(caTools)
library(e1071)
library(MLmetrics)
library(glmnet)


## Function ##########################################

# Function - plot ROC & calculate AUC
roc <- function(model, test, targetCol, title){
  Predict_ROC = predict(model, test[ ,-which(names(test) %in% c(targetCol))], type = "response")
  #print((Predict_ROC))
  #print(Predict_ROC[,2])
  
  pred = ROCR::prediction(Predict_ROC, test[[targetCol]])
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

rocEN <- function(model, test, targetCol, title){
  Predict_ROC = predict(model, as.matrix(test[ ,-which(names(test) %in% c(targetCol))], type = "response"))
  #print((Predict_ROC))
  #print(Predict_ROC[,2])
  
  pred = ROCR::prediction(Predict_ROC, test[[targetCol]])
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

######################################################


## Read in train & test data #########################

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

#####################################################


## LR A1 ############################################

# Step 1a: Build Default Model --------------------------------
set.seed(123)
lrA1 = glm(target ~.,dfImputation_train,family = binomial)
summary(lrA1)

# Step 1b: Predict on Training Set (CM) --------------------------------
pred_prob_train_lrA1 <- predict(lrA1, type = 'response', 
                                newdata= dfImputation_train[ ,-which(names(dfImputation_train) %in% c("target"))])
pred_class_train_lrA1 = ifelse(pred_prob_train_lrA1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_lrA1), dfImputation_train[['target']])
# Accuracy: 0.7689
LogLoss(pred_prob_train_lrA1, pred_class_train_lrA1)
# LogLoss: 0.2725

# Step 1c: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_lrA1 <- predict(lrA1, type = 'response', 
                               newdata= dfImputation_test[ ,-which(names(dfImputation_test) %in% c("target"))])
pred_class_test_lrA1 = ifelse(pred_prob_test_lrA1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_lrA1), dfImputation_test[['target']])
# Accuracy: 0.7653
LogLoss(pred_prob_test_lrA1, pred_class_test_lrA1)
# LogLoss: 0.2735

roc(lrA1,dfImputation_test,"target","Logistic Regression A1")
# 0.7281


# Step 2a: Elastic Net Regression with Cross Validation --------------------------------
customA1 <- trainControl(method="repeatedcv",number=10,repeats=5,verboseIter = T)
en_A1 = caret::train(target~., dfImputation_train,
                     method='glmnet', family='binomial',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=5)),
                     trControl=customA1)

# Step 2b: Plot Results --------------------------------
plot(en_A1)
plot(en_A1$finalModel, xvar='lambda', label=T)
plot(en_A1$finalModel, xvar='dev', label=T)
plot(varImp(en_A1))

# Step 2c: Get Best Model --------------------------------
en_A1$bestTune
en_A1$bestTune$lambda
set.seed(123)
en_A1_final <- glmnet(data.matrix(dfImputation_train[ ,-which(names(dfImputation_train) %in% c("target"))]), 
                      dfImputation_train[['target']], 
                      alpha = en_A1$bestTune$alpha,
                      lambda = en_A1$bestTune$lambda,
                      family = binomial)

# Step 2d: Predict on Train Set (CM) --------------------------------
pred_prob_train_enA1 <- predict(en_A1_final,  type = 'response', 
                                newx= as.matrix(dfImputation_train[ ,-which(names(dfImputation_train) %in% c("target"))]))
pred_class_train_enA1 = ifelse(pred_prob_train_enA1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_enA1), dfImputation_train[['target']])
# Accuracy: 0.7690
LogLoss(pred_prob_train_enA1, pred_class_train_enA1)
# LogLoss: 0.2725

# Step 2e: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_enA1 <- predict(en_A1_final, type = 'response', 
                               newx= as.matrix(dfImputation_test[ ,-which(names(dfImputation_test) %in% c("target"))]))
pred_class_test_enA1 = ifelse(pred_prob_test_enA1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_enA1), dfImputation_test[['target']])
# Accuracy: 0.7653
LogLoss(pred_prob_test_enA1, pred_class_test_enA1)
# LogLoss: 0.2736
rocEN(en_A1_final,dfImputation_test,"target","Elastic Net Regression A1")
# 0.7281

saveRDS(lrA1, "lrA1.rds")
#####################################################


## LR A2 ############################################

# Step 1a: Build Default Model --------------------------------
set.seed(123)
lrA2 = glm(target ~.,dfImputation_trainbal,family = binomial)
summary(lrA2)

# Step 1b: Predict on Training Set (CM) --------------------------------
pred_prob_train_lrA2 <- predict(lrA2, type = 'response', 
                                newdata= dfImputation_trainbal[ ,-which(names(dfImputation_trainbal) %in% c("target"))])
pred_class_train_lrA2 = ifelse(pred_prob_train_lrA2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_lrA2), dfImputation_trainbal[['target']])
# Accuracy: 0.6952
LogLoss(pred_prob_train_lrA2, pred_class_train_lrA2)
# LogLoss: 0.3783

# Step 1c: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_lrA2 <- predict(lrA2, type = 'response', 
                               newdata= dfImputation_test[ ,-which(names(dfImputation_test) %in% c("target"))])
pred_class_test_lrA2 = ifelse(pred_prob_test_lrA2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_lrA2), dfImputation_test[['target']])
# Accuracy: 0.7382
LogLoss(pred_prob_test_lrA2, pred_class_test_lrA2)
# LogLoss: 0.3730

roc(lrA2,dfImputation_test,"target","Logistic Regression A2")
# 0.7290


# Step 2a: Elastic Net Regression with Cross Validation --------------------------------
customA2 <- trainControl(method="repeatedcv",number=10,repeats=5,verboseIter = T)
en_A2 = caret::train(target~., dfImputation_trainbal,
                     method='glmnet', family='binomial',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=10)),
                     trControl=customA2)

# Step 2b: Plot Results --------------------------------
plot(en_A2)
plot(en_A2$finalModel, xvar='lambda', label=T)
plot(en_A2$finalModel, xvar='dev', label=T)
plot(varImp(en_A2))

# Step 2c: Get Best Model --------------------------------
en_A2$bestTune
en_A2$bestTune$lambda
set.seed(123)
en_A2_final <- glmnet(data.matrix(dfImputation_trainbal[ ,-which(names(dfImputation_trainbal) %in% c("target"))]), 
                      dfImputation_trainbal[['target']], 
                      alpha = en_A2$bestTune$alpha,
                      lambda = en_A2$bestTune$lambda,
                      family = binomial)

# Step 2d: Predict on Train Set (CM) --------------------------------
pred_prob_train_enA2 <- predict(en_A2_final,  type = 'response', 
                                newx= as.matrix(dfImputation_trainbal[ ,-which(names(dfImputation_trainbal) %in% c("target"))]))
pred_class_train_enA2 = ifelse(pred_prob_train_enA2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_enA2), dfImputation_trainbal[['target']])
# Accuracy: 0.6996
LogLoss(pred_prob_train_enA2, pred_class_train_enA2)
# LogLoss: 0.5099

# Step 2e: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_enA2 <- predict(en_A2_final, type = 'response', 
                               newx= as.matrix(dfImputation_test[ ,-which(names(dfImputation_test) %in% c("target"))]))
pred_class_test_enA2 = ifelse(pred_prob_test_enA2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_enA2), dfImputation_test[['target']])
# Accuracy: 0.7666
LogLoss(pred_prob_test_enA2, pred_class_test_enA2)
# LogLoss: 0.5002
rocEN(en_A2_final,dfImputation_test,"target","Elastic Net Regression A2")
# 0.7024

saveRDS(lrA2, "lrA2.rds")
#####################################################


## LR B1 ############################################

# Step 1a: Build Default Model --------------------------------
set.seed(123)
lrB1 = glm(target ~.,df_mice_train,family = binomial)
summary(lrB1)

# Step 1b: Predict on Training Set (CM) --------------------------------
pred_prob_train_lrB1 <- predict(lrB1, type = 'response', 
                                newdata= df_mice_train[ ,-which(names(df_mice_train) %in% c("target"))])
pred_class_train_lrB1 = ifelse(pred_prob_train_lrB1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_lrB1), df_mice_train[['target']])
# Accuracy: 0.7671
LogLoss(pred_prob_train_lrB1, pred_class_train_lrB1)
# LogLoss: 0.2752

# Step 1c: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_lrB1 <- predict(lrB1, type = 'response', 
                               newdata= df_mice_test[ ,-which(names(df_mice_test) %in% c("target"))])
pred_class_test_lrB1 = ifelse(pred_prob_test_lrB1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_lrB1), df_mice_test[['target']])
# Accuracy: 0.7640
LogLoss(pred_prob_test_lrB1, pred_class_test_lrB1)
# LogLoss: 0.2794

roc(lrB1,df_mice_test,"target","Logistic Regression B1")
# 0.7281


# Step 2a: Elastic Net Regression with Cross Validation --------------------------------
customB1 <- trainControl(method="repeatedcv",number=10,repeats=5,verboseIter = T)
en_B1 = caret::train(target~., df_mice_train,
                     method='glmnet', family='binomial',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=5)),
                     trControl=customB1)

# Step 2b: Plot Results --------------------------------
plot(en_B1)
plot(en_B1$finalModel, xvar='lambda', label=T)
plot(en_B1$finalModel, xvar='dev', label=T)
plot(varImp(en_B1))

# Step 2c: Get Best Model --------------------------------
en_B1$bestTune
en_B1$bestTune$lambda
set.seed(123)
en_B1_final <- glmnet(data.matrix(df_mice_train[ ,-which(names(df_mice_train) %in% c("target"))]), 
                      df_mice_train[['target']], 
                      alpha = en_B1$bestTune$alpha,
                      lambda = en_B1$bestTune$lambda,
                      family = binomial)

# Step 2d: Predict on Train Set (CM) --------------------------------
pred_prob_train_enB1 <- predict(en_B1_final,  type = 'response', 
                                newx= as.matrix(df_mice_train[ ,-which(names(df_mice_train) %in% c("target"))]))
pred_class_train_enB1 = ifelse(pred_prob_train_enB1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_enB1), df_mice_train[['target']])
# Accuracy: 0.7671
LogLoss(pred_prob_train_enB1, pred_class_train_enB1)
# LogLoss: 0.2753

# Step 2e: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_enB1 <- predict(en_B1_final, type = 'response', 
                               newx= as.matrix(df_mice_test[ ,-which(names(df_mice_test) %in% c("target"))]))
pred_class_test_enB1 = ifelse(pred_prob_test_enB1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_enB1), df_mice_test[['target']])
# Accuracy: 0.7638
LogLoss(pred_prob_test_enB1, pred_class_test_enB1)
# LogLoss: 0.2795
rocEN(en_B1_final,df_mice_test,"target","Elastic Net Regression B1")
# 0.7249

saveRDS(lrB1, "lrB1.rds")
#####################################################


## LR B2 ############################################

# Step 1a: Build Default Model --------------------------------
set.seed(123)
lrB2 = glm(target ~.,df_mice_trainbal,family = binomial)
summary(lrB2)

# Step 1b: Predict on Training Set (CM) --------------------------------
pred_prob_train_lrB2 <- predict(lrB2, type = 'response', 
                                newdata= df_mice_trainbal[ ,-which(names(df_mice_trainbal) %in% c("target"))])
pred_class_train_lrB2 = ifelse(pred_prob_train_lrB2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_lrB2), df_mice_trainbal[['target']])
# Accuracy: 0.7046
LogLoss(pred_prob_train_lrB2, pred_class_train_lrB2)
# LogLoss: 0.3733

# Step 1c: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_lrB2 <- predict(lrB2, type = 'response', 
                               newdata= df_mice_test[ ,-which(names(df_mice_test) %in% c("target"))])
pred_class_test_lrB2 = ifelse(pred_prob_test_lrB2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_lrB2), df_mice_test[['target']])
# Accuracy: 0.7398
LogLoss(pred_prob_test_lrB2, pred_class_test_lrB2)
# LogLoss: 0.3663

roc(lrB2,df_mice_test,"target","Logistic Regression B2")
# 0.7248


# Step 2a: Elastic Net Regression with Cross Validation --------------------------------
customB2 <- trainControl(method="repeatedcv",number=10,repeats=5,verboseIter = T)
en_B2 = caret::train(target~., df_mice_trainbal,
                     method='glmnet', family='binomial',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=5)),
                     trControl=customB2)

# Step 2b: Plot Results --------------------------------
plot(en_B2)
plot(en_B2$finalModel, xvar='lambda', label=T)
plot(en_B2$finalModel, xvar='dev', label=T)
plot(varImp(en_B2))

# Step 2c: Get Best Model --------------------------------
en_B2$bestTune
en_B2$bestTune$lambda
set.seed(123)
en_B2_final <- glmnet(data.matrix(df_mice_trainbal[ ,-which(names(df_mice_trainbal) %in% c("target"))]), 
                      df_mice_trainbal[['target']], 
                      alpha = en_B2$bestTune$alpha,
                      lambda = en_B2$bestTune$lambda,
                      family = binomial)

# Step 2d: Predict on Train Set (CM) --------------------------------
pred_prob_train_enB2 <- predict(en_B2_final,  type = 'response', 
                                newx= as.matrix(df_mice_trainbal[ ,-which(names(df_mice_trainbal) %in% c("target"))]))
pred_class_train_enB2 = ifelse(pred_prob_train_enB2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_enB2), df_mice_trainbal[['target']])
# Accuracy: 0.7120
LogLoss(pred_prob_train_enB2, pred_class_train_enB2)
# LogLoss: 0.4388

# Step 2e: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_enB2 <- predict(en_B2_final, type = 'response', 
                               newx= as.matrix(df_mice_test[ ,-which(names(df_mice_test) %in% c("target"))]))
pred_class_test_enB2 = ifelse(pred_prob_test_enB2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_enB2), df_mice_test[['target']])
# Accuracy: 0.7476
LogLoss(pred_prob_test_enB2, pred_class_test_enB2)
# LogLoss: 0.4303
rocEN(en_B2_final,df_mice_test,"target","Elastic Net Regression B2")
# 0.7208

saveRDS(lrB2, "lrB2.rds")
#####################################################


## LR C1 ############################################

# Step 1a: Build Default Model --------------------------------
set.seed(123)
lrC1 = glm(target ~.,df_mF_train,family = binomial) 
summary(lrC1)

# Step 1b: Predict on Training Set (CM) --------------------------------
pred_prob_train_lrC1 <- predict(lrC1, type = 'response', 
                                newdata= df_mF_train[ ,-which(names(df_mF_train) %in% c("target"))])
pred_class_train_lrC1 = ifelse(pred_prob_train_lrC1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_lrC1), df_mF_train[['target']])
# Accuracy: 0.7681
LogLoss(pred_prob_train_lrC1, pred_class_train_lrC1)
# LogLoss: 0.2766

# Step 1c: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_lrC1 <- predict(lrC1, type = 'response', 
                               newdata= df_mF_test[ ,-which(names(df_mF_test) %in% c("target"))])
pred_class_test_lrC1 = ifelse(pred_prob_test_lrC1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_lrC1), df_mF_test[['target']])
# Accuracy: 0.7690
LogLoss(pred_prob_test_lrC1, pred_class_test_lrC1)
# LogLoss: 0.2724

roc(lrC1,df_mF_test,"target","Logistic Regression C1")
# 0.7491


# Step 2a: Elastic Net Regression with Cross Validation --------------------------------
customC1 <- trainControl(method="repeatedcv",number=10,repeats=5,verboseIter = T)
en_C1 = caret::train(target~., df_mF_train,
                     method='glmnet', family='binomial',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=5)),
                     trControl=customC1)

# Step 2b: Plot Results --------------------------------
plot(en_C1)
plot(en_C1$finalModel, xvar='lambda', label=T)
plot(en_C1$finalModel, xvar='dev', label=T)
plot(varImp(en_C1))

# Step 2c: Get Best Model --------------------------------
en_C1$bestTune
en_C1$bestTune$lambda
set.seed(123)
en_C1_final <- glmnet(data.matrix(df_mF_train[ ,-which(names(df_mF_train) %in% c("target"))]), 
                      df_mF_train[['target']], 
                      alpha = en_C1$bestTune$alpha,
                      lambda = en_C1$bestTune$lambda,
                      family = binomial)

# Step 2d: Predict on Train Set (CM) --------------------------------
pred_prob_train_enC1 <- predict(en_C1_final,  type = 'response', 
                                newx= as.matrix(df_mF_train[ ,-which(names(df_mF_train) %in% c("target"))]))
pred_class_train_enC1 = ifelse(pred_prob_train_enC1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_enC1), df_mF_train[['target']])
# Accuracy: 0.7679
LogLoss(pred_prob_train_enC1, pred_class_train_enC1)
# LogLoss: 0.2768

# Step 2e: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_enC1 <- predict(en_C1_final, type = 'response', 
                               newx= as.matrix(df_mF_test[ ,-which(names(df_mF_test) %in% c("target"))]))
pred_class_test_enC1 = ifelse(pred_prob_test_enC1 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_enC1), df_mF_test[['target']])
# Accuracy: 0.7695
LogLoss(pred_prob_test_enC1, pred_class_test_enC1)
# LogLoss: 0.2725
rocEN(en_C1_final,df_mF_test,"target","Elastic Net Regression C1")
# 0.7491

saveRDS(lrC1, "lrC1.rds")
#####################################################


## LR C2 ############################################

# Step 1a: Build Default Model --------------------------------
set.seed(123)
lrC2 = glm(target ~.,df_mF_trainbal,family = binomial) 
summary(lrC2)

# Step 1b: Predict on Training Set (CM) --------------------------------
pred_prob_train_lrC2 <- predict(lrC2, type = 'response', 
                                newdata= df_mF_trainbal[ ,-which(names(df_mF_trainbal) %in% c("target"))])
pred_class_train_lrC2 = ifelse(pred_prob_train_lrC2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_lrC2), df_mF_trainbal[['target']])
# Accuracy: 0.7172
LogLoss(pred_prob_train_lrC2, pred_class_train_lrC2)
# LogLoss: 0.3634

# Step 1c: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_lrC2 <- predict(lrC2, type = 'response', 
                               newdata= df_mF_test[ ,-which(names(df_mF_test) %in% c("target"))])
pred_class_test_lrC2 = ifelse(pred_prob_test_lrC2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_lrC2), df_mF_test[['target']])
# Accuracy: 0.7653
LogLoss(pred_prob_test_lrC2, pred_class_test_lrC2)
# LogLoss: 0.2735

roc(lrC2,df_mF_test,"target","Logistic Regression C2")
# 0.7495


# Step 2a: Elastic Net Regression with Cross Validation --------------------------------
customC2 <- trainControl(method="repeatedcv",number=10,repeats=5,verboseIter = T)
en_C2 = caret::train(target~., df_mF_trainbal,
                     method='glmnet', family='binomial',
                     tuneGrid=expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=5)),
                     trControl=customC2)

# Step 2b: Plot Results --------------------------------
plot(en_C2)
plot(en_C2$finalModel, xvar='lambda', label=T)
plot(en_C2$finalModel, xvar='dev', label=T)
plot(varImp(en_C2))

# Step 2c: Get Best Model --------------------------------
en_C2$bestTune
en_C2$bestTune$lambda
set.seed(123)
en_C2_final <- glmnet(data.matrix(df_mF_trainbal[ ,-which(names(df_mF_trainbal) %in% c("target"))]), 
                      df_mF_trainbal[['target']], 
                      alpha = en_C2$bestTune$alpha,
                      lambda = en_C2$bestTune$lambda,
                      family = binomial)

# Step 2d: Predict on Train Set (CM) --------------------------------
pred_prob_train_enC2 <- predict(en_C2_final,  type = 'response', 
                                newx= as.matrix(df_mF_trainbal[ ,-which(names(df_mF_trainbal) %in% c("target"))]))
pred_class_train_enC2 = ifelse(pred_prob_train_enC2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_train_enC2), df_mF_trainbal[['target']])
# Accuracy: 0.7690
LogLoss(pred_prob_train_enC2, pred_class_train_enC2)
# LogLoss: 0.2725

# Step 2e: Predict on Testing Set (CM & ROC) --------------------------------
pred_prob_test_enC2 <- predict(en_C2_final, type = 'response', 
                               newx= as.matrix(df_mF_test[ ,-which(names(df_mF_test) %in% c("target"))]))
pred_class_test_enC2 = ifelse(pred_prob_test_enC2 > 0.5, 1, 0)
confusionMatrix (as.factor(pred_class_test_enC2), df_mF_test[['target']])
# Accuracy: 0.7653
LogLoss(pred_prob_test_enC2, pred_class_test_enC2)
# LogLoss: 0.2736
rocEN(en_C2_final,df_mF_test,"target","Elastic Net Regression C2")
# 0.7281

saveRDS(lrC2, "lrC2.rds")
#####################################################