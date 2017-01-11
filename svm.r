library(e1071)

## read data and create train/test split
df <- read.csv('banknote.csv',sep=',',header=TRUE)
partition <- floor(.6 * nrow(df))
set.seed(11)
df[,'class'] <- as.factor(df[,'class'])
train_ind <- sample(seq_len(nrow(df)),size=partition)
df_train <- df[train_ind,]
df_test <- df[-train_ind,]

## tune model to choose best parameters
tuned <- tune.svm(class~.,data=df_train,gamma=10^(-6:1),cost=10^(-1:1),type="C-classification")
print(summary(tuned))
#best_gamma <- tuned[['best.parameters']][['gamma']]
#best_cost <- tuned[['best.parameters']][['cost']]
model <- tuned[['best.model']]
print(summary(model))
pred <- predict(model,df_test[-5])
table(pred=pred,true=df_test[,5])
cat('acc:',sum(pred==df_test[,5])/length(df_test[,5]),'\n')
