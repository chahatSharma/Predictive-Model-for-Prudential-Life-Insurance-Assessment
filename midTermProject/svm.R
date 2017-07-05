trainData <- read.csv("C:/Users/Angad/Dropbox (Personal)/Chahat/ADS/midTermProject/train.csv/train.csv")

dim(trainData)
str(trainData)
#install.packages("caret")
library(caret)
library(e1071)




########################################################################

categorical_features <- c("Product_Info_1", "Product_Info_2", "Product_Info_3", 
                          "Product_Info_5", "Product_Info_6", "Product_Info_7", 
                          "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", 
                          "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", 
                          "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
                          "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", 
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

#########################Adding Medical Keywords
temp <- trainData %>% select(contains("Medical_Keyword"))
trainData$Medical_Keyword_Sum <- temp %>% apply(1, sum, na.rm = T)
##############################either this or dummyvars
# for (feature in categorical_features) {
#   trainData[, feature] <- as.factor(trainData[, feature])
# }

for (feature in continuous_features) {
  trainData[is.na(trainData[, feature]), feature] <- median(trainData[, feature], na.rm = TRUE)
  trainData[, feature] <- as.numeric(trainData[, feature])
}

Mode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
for (feature in discrete_features) {
  trainData[is.na(trainData[, feature]), feature] <- Mode(trainData[, feature])
  trainData[, feature] <- as.numeric(trainData[, feature])
}




##############################either this or dummyvars
for (feature in categorical_features) {
  allLevels <- levels(unlist(factor(trainData[,"Product_Info_2"])))
  contrasts(trainData[,"Product_Info_2"]) <- contr.treatment(allLevels)
  
}



for(feature in continuous_features)
{
  scale(trainData[,feature],center = TRUE,scale=TRUE)
}

#creating dummyvar model for categorical features
#dmy <- dummyVars("~Product_Info_2",data=trainData)

#converting the categorical variables into dummy variables in the data frame
#et<-as.data.frame(predict(dmy, newdata = trainData))

#appending dummy vars to train data set
#trainData <- with(trainData,data.frame(trainData[,1:128],et))
#write.csv(trainData,"cleanTrainData1.csv",row.names=FALSE)
#head(trainData)
#dim(trainData)


##splitting data for training and testing purpose
### Taking 70% of data for training and 30% for testing

##### REDUCE THE VALUE OF 1400 IF UNABLE TO RUN #####
set.seed(123)
s <- sample(length(trainData$Response),1400)
            #41567)
train_data <- trainData[s,]
test_data <- trainData[-s,]
testId <- test_data$Id
trainId <- train_data$Id
train_data$Id <- test_data$Id <- NULL
test_data$Response <-  NULL

#################################################################
#CODE FOR SVM 
#################################################################
#install.packages("dplyr")
library(dplyr)
# trainData <- sample_n(train_data, 120)
# getwd()
#train_data
#head(train_data)


#######THERE ARE TWO SVM CODE TRY TO RUN BOTH###########
####NOTE: ON MY LAPTOP I'M NOT ABLE TO PLOT THE CODE SO APNE MAI CHALA KE DHEKO###
################################################################


###### SVM CODE 1 #####


#model <- svm(as.factor(train_data$Response) ~ ., data = train_data)
#summary(model)


?subset
x <- subset(train_data, select = -train_data$Response)
y <- as.factor(train_data$Response)
model1 <- svm(x, y, probability = TRUE) 


print(model1)
summary(model1)

# Run Prediction and you can measuring the execution time in R

pred <- predict(model1,x)
system.time(pred <- predict(model1,x))

# See the confusion matrix result of prediction, using command table to compare the result of SVM
# prediction and the class data in y variable.

(table(round(pred),y))

# Tuning SVM to find the best cost and gamma ..

svm_tune <- tune(svm, train.x=x, train.y=y, 
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
warnings()
print(svm_tune)
summary(svm_tune)
plot(svm_tune) 
#######################3
# set.seed(123)
# s <- sample(length(trainData$Response)
#             #,1400)
# ,41567)
# train_data <- trainData[s,]
# test_data <- trainData[-s,]
# testId <- test_data$Id
# trainId <- train_data$Id
# train_data$Id <- test_data$Id <- NULL
# test_data$Response <-  NULL
##############3
y <- as.factor(train_data$Response)
svm_model_after_tune <- svm(y ~ ., data=train_data, kernel="radial", cost=svm_tune$best.parameters$cost, 
                            gamma=svm_tune$best.parameters$gamma,type="C-classification")
summary(svm_model_after_tune)
svm_model_after_tune$
  #plot(svm_model_after_tune,train_data,train_data$Product_Info_3~train_data$Ins_Age)
  pred <- predict(svm_model_after_tune,test_data)
################################################################
########SVM MODDEL 2##########
########AGAR YE WALA CODE NA CHALE TO AGTER ~ ADD -->>>> ##Product_Info_4, Ins_Age, 
# Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6,
# Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4,
# Family_Hist_5#### <<-- THESE VARIABLE AT EVERY PLACE AFTER ~ ########
svm_modelDummy <- svm(train_data$Response ~, data=train_data)
summary(svm_modelDummy)


svm_model <- svm(Response~., data = train_data, kernel = "linear", cost = 10, scale = FALSE)


#Element of the support vector
svm_modle$index

svm_model <- svm(Response~., data = train_data, kernel = "linear", cost = 0.1, scale = FALSE)
plot(svm_model, train_data)
?plot()

#Cross Validation using tune 

tune_svm <- tune(svm, Response~., data = train_data, kernel = "linear", ranges = list(cost = c(0.001,0.01,0.1,1,5,10,100)))
summary(tune_svm)
bestSvmModel <- tune_svm$best.model
summary(bestSvmModel)