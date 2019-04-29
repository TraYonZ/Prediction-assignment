## Data downloading

library(RCurl)

dir.create("./data")
url.training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url.training, destfile = "./data/pml-training.csv")
url.testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url.testing, destfile = "./data/pml-testing.csv")

##Reading data and data processing

train<- read.csv("./data/pml-training.csv")
test<- read.csv("./data/pml-testing.csv")

dim(train)

dim(test)

train <- train[, colSums(is.na(train)) == 0] 
test <- test[, colSums(is.na(test)) == 0] 
classe <- train$classe
trainR <- grepl("^X|timestamp|window", names(train))
train <- train[, !trainR]
trainM <- train[, sapply(train, is.numeric)]
trainM$classe <- classe
testR <- grepl("^X|timestamp|window", names(test))
test<- test[, !testR]
testM <- test[, sapply(test, is.numeric)]

##Data Partitioning

library(caret)

set.seed(12345) 
inTrain <- createDataPartition(trainM$classe, p=0.70, list=F)
train_data <- trainM[inTrain, ]
test_data <- trainM[-inTrain, ]

##Data Prediction and Modelling

setting <- trainControl(method="cv", 5)
RandomForest <- train(classe ~ ., data=train_data, method="rf", trControl=setting, ntree=250)
RandomForest

predict_RandomForest <- predict(RandomForest, test_data)
confusionMatrix(test_data$classe, predict_RandomForest)

accuracy <- postResample(predict_RandomForest, test_data$classe)
error<-1 - as.numeric(confusionMatrix(test_data$classe, predict_RandomForest)$overall[1])

## Results:
## The accuracy of the model is 98.7% and the estimated out-of-sample error is 1.3%
