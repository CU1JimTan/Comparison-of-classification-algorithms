library(e1071)
library(ROCR)
library(randomForest)
library(rpart)
library(neuralnet)
library(ipred)
library(adabag)
library(caret)
library(ada)
library(gbm)
library(class)
library(ROCR)
data <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",header=FALSE)
data <- data[-1]
data[ is.na(data) ] <- 0
data[data == "?"] <- 0
apply(data,2,function(x) sum(is.na(x)))
colnames(car)<-c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","Class")
set.seed(100)
classPosition <- length(data)
ClassIndex <- length(data)
classLabels <- colnames(data)[classPosition]
rows <- nrow(data)
columns <-ncol(data)-1 
k <- 10
cat("no of instances", rows)
cat("no of attributes", columns)
cat("no of folds", k)
classlength<-length(classLabels)
value <- as.integer(10)

#classifiers=c("perceptron")
#classifiers=c("adaBoosting")
classifiers=c( "decisionTree",  "SVM" ,"logisticRegression" , "naiveBayes" , "knn" , "bagging","randomForest","perceptron","neuralNetwork","deepLearning","adaBoosting", "gradientBoosting")

buildModel <- function(algorithm,trainingData,testData, Class,classPosition)
{
  f <- as.formula(paste("as.factor(",Class,") ~","." ))
  y = factor(testData[,classPosition])
  
  if(algorithm == "decisionTree")
  {
    ctrl=rpart.control( cp = 0.001, maxdepth = 30)
    model <-  rpart(f, trainingData , method = "class",control=ctrl)
    prediction <- predict(model,testData,type="class")
    acc <-  sum((prediction == testData[,classPosition])/length(testData[,classPosition]))*100
    p = factor(prediction)
    precision <- posPredValue(p, y)
    recall <- sensitivity(p, y)
    F1 <- (2 * precision * recall) / (precision + recall)  
  }
  else if(algorithm == "perceptron")
  {
    n <-names(trainingData)
    n <- names(testData)
    trainingData<-sapply(trainingData,as.numeric)
    testData<-sapply(testData,as.numeric)
    f <- as.formula(paste(colnames(trainingData)[as.integer(value)],"~", paste(n[!n %in% colnames(trainingData)[as.integer(value)]], collapse = " + ")))
    f1<-paste(c(n[!n %in% colnames(trainingData)[as.integer(value)]]))
    creditnet <- neuralnet(f ,trainingData, hidden = 0, threshold = 0.01,act.fct = "tanh")
    temp_test <- subset(testData, select = c(f1))
    creditnet.results <- compute(creditnet, temp_test)
    results <- data.frame(actual = testData[,as.integer(value)], prediction = creditnet.results$net.result)
    results$prediction <- round(results$prediction)
    acc<-(sum(results[,2]==testData[,as.integer(value)])/nrow(testData))*100
    
    confusion_matrix<-table(results$prediction,round(testData[,as.integer(value)]))
    precision <- diag(confusion_matrix) / colSums(confusion_matrix)
    precision = mean(precision) 
    

  }
  else if(algorithm == "neuralNetwork")
  {
    n <- names(testData)
    trainingData<-sapply(trainingData,as.numeric)
    testData<-sapply(testData,as.numeric)
    f <- as.formula(paste(colnames(trainingData)[as.integer(value)],"~", paste(n[!n %in% colnames(trainingData)[as.integer(value)]], collapse = " + ")))
    f1<-paste(c(n[!n %in% colnames(trainingData)[as.integer(value)]]))
    creditnet <- neuralnet(f ,trainingData, hidden = 2, threshold = 0.2)
    temp_test <- subset(testData, select = c(f1))
    creditnet.results <- compute(creditnet, temp_test)
    results <- data.frame(actual = testData[,as.integer(value)], prediction = creditnet.results$net.result)
    results$prediction <- round(results$prediction)
    acc<-(sum(results[,2]==testData[,as.integer(value)])/nrow(testData))*100
  
    confusion_matrix<-table(results$prediction,round(testData[,as.integer(value)]))
    precision <- diag(confusion_matrix) / colSums(confusion_matrix)
    precision = mean(precision)
    
    }
  else if(algorithm == "deepLearning")
  {
    n <- names(testData)
    trainingData<-sapply(trainingData,as.numeric)
    testData<-sapply(testData,as.numeric)
    f <- as.formula(paste(colnames(trainingData)[as.integer(value)],"~", paste(n[!n %in% colnames(trainingData)[as.integer(value)]], collapse = " + ")))
    f1<-paste(c(n[!n %in% colnames(trainingData)[as.integer(value)]]))
    creditnet <- neuralnet(f ,trainingData, hidden = 20, threshold = 0.1)
    temp_test <- subset(testData, select = c(f1))
    creditnet.results <- compute(creditnet, temp_test)
    results <- data.frame(actual = testData[,as.integer(value)], prediction = creditnet.results$net.result)
    results$prediction <- round(results$prediction)
    acc<-(sum(results[,2]==testData[,as.integer(value)])/nrow(testData))*100
    
    confusion_matrix<-table(results$prediction,round(testData[,as.integer(value)]))
    precision <- diag(confusion_matrix) / colSums(confusion_matrix)
    precision = mean(precision) 
    
  }
  else if(algorithm == "SVM")
  {
    #cat("algo", algorithm)
    model <-svm(f,trainingData)
    prediction <- predict(model,testData,type="class",cost=10,gamma=0.1, kernal="linear")
    acc <-  sum((prediction == testData[,classPosition])/length(testData[,classPosition]))*100
    
    p = factor(prediction)
    precision <- posPredValue(p, y)
    recall <- sensitivity(p, y)
    F1 <- (2 * precision * recall) / (precision + recall)  
    
  }
  else if (algorithm == "naiveBayes")
  {
    #cat("algo", algorithm)    
    model <- naiveBayes(f, data = trainingData,na.action = na.omit)
    prediction <- predict(model,testData)
    acc <-  sum((prediction == testData[,classPosition])/length(testData[,classPosition]))*100
    
    p = factor(prediction)
    precision <- posPredValue(p, y)
    recall <- sensitivity(p, y)
    F1 <- (2 * precision * recall) / (precision + recall)  
  }
  
  else if(algorithm == "logisticRegression")
  {
    lrmodel <- glm(f, data = trainingData, family = "binomial")
    anova(lrmodel)
    prediction <- predict(lrmodel,testData)
    mean_sqr_err <- sum((prediction - testData$V11 )^2/nrow(testData))
    acc <- 100-mean_sqr_err
    
    confusion_matrix<-table(round(prediction),round(testData[,as.integer(value)]))
    precision <- diag(confusion_matrix) / colSums(confusion_matrix)
    precision = mean(precision) 
    
  }
  else if (algorithm == "knn")
  {
    library("class")
    set.seed(10)
    dim(trainingData)
    dim(testData)
    
    model.knn <- knn(trainingData[,1:(ClassIndex-1)], testData[,1:(ClassIndex-1)],trainingData[,ClassIndex],k =5, l=2,prob=TRUE)
    tb_name <- table("Predictions" = model.knn, Actual = testData[,ClassIndex])
    acc <- (sum(diag(tb_name)) / sum(tb_name))*100
    
    p = factor(model.knn)
    precision <- posPredValue(p, y)
    recall <- sensitivity(p, y)
    F1 <- (2 * precision * recall) / (precision + recall)
    
  }
  else if (algorithm == "bagging")
  {
    #bag <- ipred::bagging(f, data=trainingData, boos = TRUE,mfinal=10, control = rpart.control(cp = 0)) 
    bag <- ipred::bagging(f, data=trainingData, boos = TRUE,mfinal=10, length_divisor=4,iterations=1000,control = rpart.control(cp = 0))
    prediction <-predict(bag,testData)
    acc<- (sum(prediction==testData[,classPosition]))/length(testData[,classPosition])*100.0
    
    p = factor(prediction)
    precision <- posPredValue(p, y)
    recall <- sensitivity(p, y)
    F1 <- (2 * precision * recall) / (precision + recall)  
    
  }
  
  else if (algorithm == "randomForest")
  {
   # rf <- randomForest(f,data=trainingData,importance=TRUE, ntree=2000)
    rf <- randomForest(f,data=trainingData,importance=TRUE,mtry = 2,max_features=2,proximity = FALSE, ntree=1000)
    prediction<-predict(rf,testData)
    acc <- (sum(prediction==testData[,classPosition]))/length(testData[,classPosition])*100.0
    
    p = factor(prediction)
    precision <- posPredValue(p, y)
    recall <- sensitivity(p, y)
    F1 <- (2 * precision * recall) / (precision + recall)
    
  }
  
  else if (algorithm == "adaBoosting")
  {
   # adaboostmodel <- ada(f, data = trainingData, iter=20, nu=1, type="discrete")
    adaboostmodel <- ada(f, data = trainingData, iter=20, nu=1,delta = 4, bag.frac = 20,type="gentle")
    prediction <- predict(adaboostmodel,testData)
    acc <- (sum(prediction==testData[,classPosition]))/length(testData[,classPosition])*100.0
    
    p = factor(prediction)
    precision <- posPredValue(p, y)
    recall <- sensitivity(p, y)
    F1 <- (2 * precision * recall) / (precision + recall)  
    
  }
  else if (algorithm == "gradientBoosting")
  {
    set.seed(1)
    
    #gbmModel = gbm(f, data=trainingData, n.trees=2000,shrinkage=0.02,distribution="gaussian",interaction.depth=10,bag.fraction=0.1,cv.fold=10,n.minobsinnode = 50)
    gbmModel = gbm(f, data=trainingData, n.trees=2000,shrinkage=0.02,distribution="gaussian",interaction.depth=10,bag.fraction=0.4,cv.fold=10,n.minobsinnode = 50)
    
    prediction <- predict(gbmModel, newdata=testData[,-classPosition],OOB=TRUE, type = "response")
    acc <- 100 - (sum(testData[,classPosition]==round(prediction))/length(prediction)*100)
    
    
  confusion_matrix<-table(round(prediction),round(testData[,as.integer(value)]))
    precision <- diag(confusion_matrix) / colSums(confusion_matrix)
    precision = mean(precision) 
    
  }
  
  else 
  {
    cat("undefined classifer algorithm")
    acc <-0
    precision <-0
  }  
  
  
  return(list(accuracy = acc, parameter = precision))
  
}  


for(c in classifiers)
{
  accuracy_sum <-0  
  parameter_sum <-0
  for (i in 1:k)
  {
    Training_instances <- sample(1:nrow(data),size = 0.9*nrow(data))
    trainingData<-data[Training_instances,]
    testData<-data[-Training_instances,]
    
    trainingData[trainingData == "?"] <- NA
    trainingData<- trainingData[complete.cases(trainingData),]
    testData[testData == "?"] <- NA
    testData<- testData[complete.cases(testData), ]
    
    
    result <- buildModel(c,trainingData,testData,classLabels,classPosition)  
    acc = result$accuracy
    param = result$parameter
    accuracy_sum <- accuracy_sum+acc
    parameter_sum <- parameter_sum+param
  }
  
  
  avg_accuracy <- accuracy_sum/k 
  avg_parameter <- parameter_sum/k 
  cat("\n classifier =" , c , "\n")
  cat("\t Avg Accuracy =" , avg_accuracy)
  cat ("\t Avg Precision" , avg_parameter )
  
  
}


