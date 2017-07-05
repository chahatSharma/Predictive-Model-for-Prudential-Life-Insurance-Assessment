############installing required packages###########################################################
#install.packages("xgboost")
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("xgboost", repos="http://dmlc.ml/drat/", type="source")
#install.packages("magrittr")
library(caret)
require(xgboost)
require(Matrix)
library(magrittr)
library(dplyr)

#############Loading train data######################################################################
trainData <- read.csv("C:/Users/Angad/Dropbox (Personal)/Chahat/ADS/midTermProject/train.csv/train.csv")

############# Separating the different types of independent variables ##################################
categorical_features <- c("Product_Info_1", "Product_Info_2", "Product_Info_3", 
                          "Product_Info_5", "Product_Info_6", "Product_Info_7", 
                          "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", 
                          "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", 
                          "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
                          "Insurance_History_1", "Insurance_History_2", "Insurance_Hi
                          story_3", 
                          "Insurance_History_4", "Insurance_History_7", "Insurance_History_8", 
                          "Insurance_History_9", "Family_Hist_1", "Medical_History_2", 
                          "Medical_History_3", "Medical_History_4", "Medical_History_5", 
                          "Medical_History_6", "Medical_History_7", "Medical_History_8", 
                          "Medical_History_9", "Medical_History_11", "Medical_History_12", 
                          "Medical_History_13", "Medical_History_14", "Medical_History_16", 
                          "Medical_History_17", "Medical_History_18", "Medical_History_19", 
                          "Medical_History_20", "Medical_History_21", "Medical_History_22", 
                          "Medical_History_23", "Medical_History_25", "Medical_History_26", 
                          "Medical_History_27", "Medical_History_28", "Medical_History_29", 
                          "Medical_History_30", "Medical_History_31", "Medical_History_33", 
                          "Medical_History_34", "Medical_History_35", "Medical_History_36", 
                          "Medical_History_37", "Medical_History_38", "Medical_History_39", 
                          "Medical_History_40", "Medical_History_41")
continuous_features <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", 
                         "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", 
                         "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5")
discrete_features <- c("Medical_History_1", "Medical_History_10", "Medical_History_15", 
                       "Medical_History_24", "Medical_History_32")

temp <- trainData %>% select(contains("Medical_Keyword"))
trainData$Medical_Keyword_Sum <- temp %>% apply(1, sum, na.rm = T)


##########################################Tried  converting all factors to numbers #########################
# for (feature in categorical_features) {
#   trainData[, feature] <- as.factor(trainData[, feature])
# }
#for (feature in categorical_features) {
  #allLevels <- levels(unlist(factor(trainData[,"Product_Info_2"])))
  #contrasts(trainData[,"Product_Info_2"]) <- contr.treatment(allLevels)

#}

#creating dummyvar model for categorical features
dmy <- dummyVars("~Product_Info_2",data=trainData)

#converting the categorical variables into dummy variables in the data frame
et<-as.data.frame(predict(dmy, newdata = trainData))

#appending dummy vars to train data set
trainData <- with(trainData,data.frame(trainData[,1:128],et))


#dmy <- dummyVars("~ Product_Info_2",data=trainData)

#################################################further realised it was not necessary###################################################

################################################# Fillinf missing values ######################################################3

##using median to fill NAs for continuous_features
for (feature in continuous_features) {
  trainData[is.na(trainData[, feature]), feature] <- median(trainData[, feature], na.rm = TRUE)
  trainData[, feature] <- as.numeric(trainData[, feature])
}
## using model to fill NAs for discrere_features
Mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
for (feature in discrete_features) {
  trainData[is.na(trainData[, feature]), feature] <- Mode(trainData[, feature])
  trainData[, feature] <- as.numeric(trainData[, feature])
}
trainDataNew <- trainData

#################################################Plots##################################
plot(density(trainData$Response))
table(trainData$Response)

for(feature in continuous_features)
{
  #plot(trainData[,feature],trainData$Response, main="Scatterplot" , ylab =  "Response" , xlab = as.character(feature))
  r <- cor.test(trainData$Response,trainData[,feature],method = 'pearson')
  #corrplot.mixed(cor(trainData[,"Response"],trainData[,feature]))
  hist(trainData[,feature],freq = T,xlab = as.character(feature))
}




#########################################Partitioning data for training and testing #################################################

### using 70% of train data for training and rest 30% for testing purpose
set.seed(123)
s <- sample(length(trainData$Response),41567)
train_data <- trainData[s,]
test_data <- trainData[-s,]
test_data_new <- trainData[-s,]
## removing response from test_data because this is the dependent varaibale which we need to predict
#test_data$Response <- 0
## removing id variable as it does not make much sense
testId <- test_data$Id
trainId <- train_data$Id
train_data$Id <- test_data$Id <- NULL

#######################################################################################










######################################################################################

##########################################converting train dataset into 10 subsets


## watchlist
watchlist <- list(train = train_data, test = test_data)

###converting training data into a sparse matrix excluding Response variable , it will convert all 
### other categorical variables into numbers
sparse_matrix <- sparse.model.matrix(Response~.-1, data = train_data)
head(sparse_matrix)
sparse_matrix_test <- sparse.model.matrix(Response~.-1, data = test_data)
# write.csv(test_data, file="predictedResultsForPrudentialy8.csv", row.names = FALSE)

###############################################Tried to take y=1,2,3,4,5,6,7,8 and y <3,4,5,6,7############################################
 nround.cv=200
# for(i in seq(1,8,1))
# {
#   output_vector = train_data[,127] == i
#   xgboost.cv<-xgb.cv(param=list(objective = 'binary:logistic',
#                                 max_depth         = 5,      
#                                 eta               = 0.162,  
#                                 gamma             = .01,    
#                                 subsample         = .985,     
#                                 colsample_bytree  = .847,
#                                 min_child_weight  = 0), data=sparse_matrix,
#                      label = output_vector,nfold=10, 
#                      nrounds=nround.cv, prediction=TRUE, verbose=0,early.stop.round = 3,maximize = TRUE)
#   xgboost.cv$dt
#   length(xgboost.cv$pred)
#   train_data[,paste0("Response_",i)] <- round(xgboost.cv$pred)
# 
#   # index of maximum auc:
#   ?which.max
#   min.error.idx = which.min(xgboost.cv$dt[, test.error.mean]) 
#   min.error.idx 
#   ## [1] 493
#   # minimum merror
#   xgboost.cv$dt[min.error.idx,]
#   ?xgb.cv
#   # model <- xgboost(data=sparse_matrix ,params      = list(max_depth         = 5,      
#   #                                                         eta               = 0.162,  
#   #                                                         gamma             = .01,    
#   #                                                         subsample         = .985,     
#   #                                                         colsample_bytree  = .847,
#   #                                                         min_child_weight  = 0),
#   #                  label = output_vector,nrounds = max.auc.idx,
#   #                  watchlist=watchlist, objective = 'binary:logistic' )
#   # 
#   # test_data[,paste0("Response_",i)] <- round(predict(model,sparse_matrix_test))
#   # head(test_data)
#  
# }
# 
# for(i in seq(3,7,1))
# {
#   output_vector = train_data[,127] < i
#   xgboost.cv<-xgb.cv(param=list(objective = 'binary:logistic',
#                                 max_depth         = 5,      
#                                 eta               = 0.162,  
#                                 gamma             = .01,    
#                                 subsample         = .985,     
#                                 colsample_bytree  = .847,
#                                 min_child_weight  = 0), data=sparse_matrix,
#                      label = output_vector,nfold=10, 
#                      nrounds=nround.cv, prediction=TRUE, verbose=0,early.stop.round = 3,maximize = TRUE)
#   xgboost.cv$dt
#   length(xgboost.cv$pred)
# 
#   train_data[,paste0("Response_<",i)] <- round(xgboost.cv$pred)
#   
#   # index of maximum auc:
#   min.error.idx = which.min(xgboost.cv$dt[, test.error.mean]) 
#   min.error.idx 
#   ## [1] 493
#   # minimum merror
#   xgboost.cv$dt[min.error.idx,]
#   
#   # model <- xgboost(data=sparse_matrix ,params      = list(max_depth         = 5,      
#   #                                                         eta               = 0.162,  
#   #                                                         gamma             = .01,    
#   #                                                         subsample         = .985,     
#   #                                                         colsample_bytree  = .847,
#   #                                                         min_child_weight  = 0),
#   #                  label = output_vector,nrounds = max.auc.idx,
#   #                  watchlist=watchlist, objective = 'binary:logistic' )
#   # 
#   # test_data[,paste0("Response_<",i)] <- round(predict(model,sparse_matrix_test))
#   # head(test_data)
#   # 
#   
#   
#   
# }
# feature.names <-names(train_data)[which(names(train_data) != "Response")]
# dtrain <- xgb.DMatrix(data = data.matrix(train_data[,feature.names]), label=train_data$Response)
# dtest <- xgb.DMatrix(data = data.matrix(test_data[,feature.names]), label=test_data$Response)
# 
# 
# model <- xgboost(data=dtrain ,params      = list(max_depth         = 5,
#                                                         eta               = 0.162,
#                                                         gamma             = .01,
#                                                         subsample         = .985,
#                                                         colsample_bytree  = .847,
#                                                         min_child_weight  = 0),
#                  nrounds = 4,
#                  watchlist=watchlist, objective = 'reg:linear' )
# model$raw
# 
# test_data$Response <- round(predict(model,dtest),0)
# 
# err <- mean(as.numeric(test_data$Response) != test_data_new$Response)

########But it further added 13 features to my dataset which made my model even worse#############

#############################calculated probabilities for each values in Response
y<- train_data$Response
y <- y - 1
max(y)
xgb <- xgb.cv(data = sparse_matrix, label =y, eta = 0.162, max_depth = 15, nround = 91,
               subsample=0.985,nfold = 10,
               objective = 'multi:softprob',num_class = length(levels(as.factor(y))))

min.error.idx = which.min(xgb[, test.merror.mean]) 
min.error.idx 
xgboosts <- xgboost(data = sparse_matrix, label = y, eta = 0.162, max_depth = 15, nround = min.error.idx, subsample=0.5,
               objective = 'multi:softprob',num_class = 
                 length(levels(as.factor(y))))

models <- xgb.dump(model =xgboosts,with.stats=T)


models[1:10]
names <- dimnames(data.matrix(train_data))[[2]]
importance_matrix <- xgb.importance(names,model=xgboosts)
#install.packages("Ckmeans.1d.dp")
xgb.plot.importance(importance_matrix[1:40])
#barplot(importance_matrix[,1])
plot(hist(importance_matrix$Gain))

p <- predict(xgboosts,sparse_matrix_test)
probs <- t(matrix(p, nrow=8, ncol=length(p)/8))
#l <- matrix(p,ncol = length(levels(as.factor(y))),byrow = T)
#dim(l)
pred.cv <- matrix(p, nrow=length(p)/length(levels(as.factor(y))), ncol=length(levels(as.factor(y))))
?max.col
pred.cv <- max.col(pred.cv,"last")
confusionMatrix(factor(test_data_new$Response), factor(pred.cv))

sum(importance_matrix$Frequence)
sum(importance_matrix$Gain)
sum(importance_matrix$Cover)

#############################calculated probabilities for each values in Response

####CROSS VALIDATION########################
apply(train_data,2,var)
pca = prcomp(train_data)
#########Calculated the min rmse mean error and used it as nrounds to train model
feature.names <-names(train_data)[which(names(train_data) != "Response")]
dtrain <- xgb.DMatrix(data = data.matrix(train_data[,feature.names]), label=as.factor(train_data$Response))

xgboost.cv<-xgb.cv(param=list(objective = 'reg:logistic',
                              max_depth         = 6,      
                              eta               = 0.162,  
                              gamma             = .01,    
                              subsample         = .985,     
                              colsample_bytree  = .847,
                              min_child_weight  = 0), data=dtrain,
                   nfold=10, 
                   nrounds=91, prediction=TRUE, verbose=0
                   #,early.stop.round = 10,maximize = FALSE
                   )
xgboost.cv$dt
length(xgboost.cv$pred)

# index of minimum error
min.error.idx = which.min(xgboost.cv$dt[, test.rmse.mean]) 
min.error.idx 

# minimum merror
xgboost.cv$dt[min.error.idx,]

#$$$$$$$$$$$$Linear Regression model$$$$$$$$$$$$$$$$$
feature.names <-names(train_data)[which(names(train_data) != "Response")]
dtrain <- xgb.DMatrix(data = data.matrix(train_data[,feature.names]), label=as.factor(train_data$Response))
dtest <- xgb.DMatrix(data = data.matrix(test_data[,feature.names]), label=as.factor(test_data$Response))

BoostedTrees <- xgboost(data        = dtrain, 
                          params      = list(max_depth         = 6,
                                             eta               = 0.162,
                                             gamma             = .01,
                                             subsample         = .985,
                                             colsample_bytree  = .847,
                                             min_child_weight  = 1,
                                             silent =0),
                          nround      = min.error.idx,
                          watchlist   = list(validation1 = dtrain,validation2 = dtest),
                          objective   = "reg:linear"
                          #,early.stop.round = 10,maximize = TRUE
                          )

plot(xgboost.cv$dt,las = 1 )
test_data$Response <- round(predict(BoostedTrees,dtest),0)

err <- mean(test_data$Response != test_data_new$Response)
#install.packages("DiagrammeR")
#xgb.plot.tree(feature_names = dimnames(data.matrix(train_data))[[2]],model = BoostedTrees)
importance_matrixs <- xgb.importance(feature.names, model = BoostedTrees)[1:90, ]
confusionMatrix(factor(test_data_new$Response), factor(test_data$Response))

xgb.plot.importance(importance_matrixs) +theme(text = element_text(size=12)) +
  theme(axis.text.y=element_text(colour="black"))

test_data$Response <- round(predict(BoostedTrees, data.matrix(test_data)),digits = -0)
write.csv(test_data,"predictedValueByBoostedTree.csv", row.names = FALSE)

# ######################################Random Forests#######################3
# #install.packages("randomForest")
library(randomForest)
set.seed(16)
?tuneRF
mtry <- tuneRF(train_data[,2:127], y = train_data$Response,mtryStart = 1, ntreeTry = 50, stepFactor = 2,
           trace = TRUE,  improve = 0.05, plot = TRUE, doBest = FALSE)
?randomForest
mtry
ptm4 <- proc.time()
fits4 <- randomForest(as.factor(Response) ~ . , data = train_data, importance = TRUE, ntree = 100,mtry=8)
fits4.time <- proc.time() - ptm4
varImpPlot(fits4)
#test_data$Response <- NULL
fit4.pred <- predict(fits4, test_data, type="response")
table(fit4.pred,test_data_new$Response)
fits4$error <- 1-(sum(fit4.pred==test_data_new$Response)/length(test_data_new$Response))
fits4$error
# 
# fits4
# getTree(fits4,1,labelVar = TRUE)
# library("party")
# party::prettytree(fits4@ensemble[[1]],names(fits4@data@get("input")))
# x <- ctree(Response ~ . , data = train_data)
# plot(x,type='Simple')
##################################################################################3
summary(lm((Response) ~ . , data = trainDataNew))

Return
