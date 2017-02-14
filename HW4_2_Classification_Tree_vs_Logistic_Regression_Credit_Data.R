# Data Mining
# HW 4 Problem 2
#  


##### install packages #####


packages <- c("ggplot2","GGally","rpart","caret","leaps","boot","reshape2", "ROCR")

install.packages(packages)

lapply(as.list(packages),library, character.only = TRUE)

# German Credit Score Prediction

##### Starter code for German credit scoring #####

# Refer to http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data) for variable description. 
#Notice that “It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).” 
#Define your cost function accordingly!

german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(german_credit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")

# orginal response coding 1= good, 2 = bad we need 0 = good, 1 = bad
german_credit$response = german_credit$response - 1


##### Partition Train/Test #####

# seed for data split 1

set.seed(1984)

# seed for data split 1

set.seed(5001)

index <- sample(1:nrow(german_credit),nrow(german_credit)*.7)

train <- german_credit[index,]

test <- german_credit[-index,]


##### Summary Stats #####


str(train)
summary(train) #no NA values

boxplot.stats(train$amount)

table(train$response)

##### EDA #####


#prep for correlation matrix

num <- sapply(train, is.numeric)

df_cor <- train[,num]

zv <- apply(df_cor, 2, function(x) length(unique(x)) <= 2)

sum(zv)
zv

df_cor <- df_cor[, !zv]

corr <- cor(df_cor,use = "pairwise.complete.obs")

highCorr <- findCorrelation(corr, 0.70)

length(highCorr) 

colnames(corr[, highCorr,drop = FALSE]) 

ggcorr(df_cor, method = c("complete", "pearson"), nbreaks = 10) #could also use pairwise, which is the default


##### variable selection #####

# set null and full model for use in variable selection

# logistic
nullmodel.logit = glm(response ~ 1 ,family = binomial(link="logit"), data = train)
fullmodel.logit = glm(response ~ . ,family = binomial(link="logit"), data = train)

# probit

nullmodel.probit = glm(response ~ 1 ,family = binomial(link="probit"), data = train)
fullmodel.probit = glm(response ~ . ,family = binomial(link="probit"), data = train)

# complementary log-log link

nullmodel.loglog = glm(response ~ 1 ,family = binomial(link="cloglog"), data = train)
fullmodel.loglog = glm(response ~ . ,family = binomial(link="cloglog"), data = train)

# Warning messages:
# 1: glm.fit: algorithm did not converge 
# 2: glm.fit: fitted probabilities numerically 0 or 1 occurred 

#bw

backward.logit = step(fullmodel.logit, direction = "backward")

backward.probit = step(fullmodel.probit, direction = "backward")

backward.loglog = step(fullmodel.probit, direction = "backward")

#fw

forward.logit = step(nullmodel.logit, scope = list(lower = nullmodel.logit, upper = fullmodel.logit), direction = "forward")

forward.probit = step(nullmodel.probit, scope = list(lower = nullmodel.probit, upper = fullmodel.probit), direction = "forward")

forward.loglog = step(nullmodel.loglog , scope = list(lower = nullmodel.loglog , upper = fullmodel.loglog ), direction = "forward")

#step

step.logit = step(nullmodel.logit, scope = list(lower = nullmodel.logit, upper = fullmodel.logit), direction = "both")

step.probit = step(nullmodel.probit, scope = list(lower = nullmodel.probit, upper = fullmodel.probit), direction = "both")

step.loglog = step(nullmodel.loglog , scope = list(lower = nullmodel.loglog , upper = fullmodel.loglog ), direction = "both")

# Compare GLMs

#AIC

AIC(backward.logit)
AIC(backward.probit)
AIC(backward.loglog)
AIC(forward.logit)
AIC(forward.probit)
AIC(forward.loglog)
AIC(step.logit)
AIC(step.probit)
AIC(step.loglog)

# Each link had essentially the same AIC with one of the three selection methods ~ 688

#BIC

BIC(backward.logit)
BIC(backward.probit)
BIC(backward.loglog)
BIC(forward.logit)
BIC(forward.probit)
BIC(forward.loglog)
BIC(step.logit)
BIC(step.probit)
BIC(step.loglog)

# complementary log-log link with forward stepwise selection had the best BIC 833
# followed by backward selection with logit with 838

##### Logistic Regression #####

# best logit model from above by AIC/BIC was found with backward selection

summary(backward.logit)

# Residual deviance: 621.82  on 667  degrees of freedom

# important variable levels - can compare to splits in classification tree
# 
# chk_acctA12     .  
# chk_acctA13       *  
#   chk_acctA14       ***
#   duration           *  
#  
# credit_hisA33     *  
#   credit_hisA34    ***
#   purposeA41       ***
# purposeA42       ***
#   purposeA43   ***
#    
# 
# purposeA48       .  
# purposeA49        *  
#   amount           *  
# saving_acctA64   .  
# saving_acctA65  ** 
# present_empA74    *  
# installment_rate   *  
# other_debtorA103  .  
# age               *  
#   n_credits        *  
#   foreignA202      *  


###### In-sample Confusion Matrix, ROC, AUC #####

# Confusion Matrix

prob.in.sample <- predict(backward.logit,  type = "response") 

# using 1/6 for classification cut-off
prob.in.sample.binary <- as.numeric(predict(backward.logit,  type = "response") > (1/6))

#using confusionMatrix for caret package - picking 0 as "positive", but that won't affect accuracy calc
confusion.in.sample <-confusionMatrix(prob.in.sample.binary,train$response)

# Misclassification Rate: 33.29%


# In-sample ROC
pred <- prediction(prob.in.sample, train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

# In-Sample AUC
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values

# 0.838


###### Out-of-Sample Confusion Matrix, ROC, AUC #####

# Confusion Matrix

prob.out.sample <- predict(backward.logit, test, type = "response")

prob.out.sample.binary <- as.numeric(predict(backward.logit, test, type = "response") > (1/6))

#using confusionMatrix for caret package - picking 0 as "positive", but that won't affect accuracy calc
confusionMatrix(prob.out.sample.binary,test$response)

# Misclassification Rate: 39.33%

# ROC

pred <- prediction(prob.out.sample, test$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

# AUC

auc.perf = performance(pred, measure = "auc")
auc.perf@y.values

# 0.7585


##### find cutoff probability with lowest cost #####

# define the searc grid from 0.01 to 0.99
searchgrid = seq(0.01, 0.99, 0.01)
# result is a 99x2 matrix, the 1st col stores the cut-off p, the 2nd column
# stores the cost
result = cbind(searchgrid, NA)
# in the cost function, both r and pi are vectors, r=truth, pi=predicted
# probability
cost1 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < pcut)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > pcut)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

for (i in 1:length(searchgrid)) {
  pcut <- result[i, 1]
  # assign the cost to the 2nd col
  result[i, 2] <- cost1(train$response, predict(backward.logit, type = "response"))
}
plot(result, ylab = "Cost in Training Set") 

result[which.min(result[,2]),1] # 0.13 is cutoff with smallest cost

##### Cross Validation #####

searchgrid = seq(0.01, 0.4, 0.02)
result.cv = cbind(searchgrid, NA)
cost1 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi < pcut)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi > pcut)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}

# logistic
nullmodel.logit.cv = glm(response ~ 1 ,family = binomial(link="logit"), data = german_credit)
fullmodel.logit.cv = glm(response ~ . ,family = binomial(link="logit"), data = german_credit)

backward.logit.cv = step(fullmodel.logit.cv, direction = "backward")



for (i in 1:length(searchgrid)) {
  set.seed(567)
  pcut <- result.cv[i, 1]
  result.cv[i, 2] <- cv.glm(data = german_credit, glmfit = backward.logit.cv, cost = cost1, 
                         K = 3)$delta[2]
}

plot(result.cv, ylab = "CV Cost")

result[which.min(result.cv[,2]),1]

# Cross Validation yielded an optimal cut-off to reduce cost at 0.15

pcut <- 0.15

cv.final <- cv.glm(data = german_credit, glmfit = backward.logit.cv, cost = cost1, K = 3)

cv.final

# The estimated prediction error was 0.495 -> 49.5% ? 


##### Classification Tree ######

model.class.tree <- rpart(formula = response ~ ., data = train, method = "class", 
                      parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)))

# plot classification tree
rpart.plot(model.class.tree,tweak = 1.5)

##### In-Sample Misclassification Rate, ROC, AUC #####

in.sample.binary.class.tree = predict(model.class.tree, type = "class")
in.sample.binary.table <- table(train$response, in.sample.binary.class.tree, dnn = c("Truth", "Predicted"))

# Misclassification rate
1 - (in.sample.binary.table[1,1]+in.sample.binary.table[2,2])/sum(in.sample.binary.table)
# In-sample misclassification rate: 27.1%

# ROC

in.sample.pred.class.tree = predict(model.class.tree)
pred.class.tree = prediction(in.sample.pred.class.tree[, 2], train$response)
perf.class.tree = performance(pred.class.tree, "tpr", "fpr")
plot(perf.class.tree, colorize = TRUE)

# AUC 

slot(performance(pred.class.tree, "auc"), "y.values")[[1]]

# .8374

##### Out-of-Sample Misclassification Rate, ROC, AUC #####

out.sample.binary.class.tree = predict(model.class.tree, test, type = "class")
out.sample.binary.table <- table(test$response, out.sample.binary.class.tree, dnn = c("Truth", "Predicted"))

# Misclassification rate
1 - (out.sample.binary.table[1,1]+out.sample.binary.table[2,2])/sum(out.sample.binary.table)
# In-sample misclassification rate: 34.7%

# ROC

out.sample.pred.class.tree = predict(model.class.tree,test)
out.pred.class.tree = prediction(out.sample.pred.class.tree[, 2], test$response)
out.perf.class.tree = performance(out.pred.class.tree, "tpr", "fpr")
plot(out.perf.class.tree, colorize = TRUE)

# AUC 

slot(performance(out.pred.class.tree, "auc"), "y.values")[[1]]

# 0.7249


##### Prune Tree #####

plotcp(model.class.tree) # prune tree to size of 6

printcp(model.class.tree)

model.class.tree.pruned <- prune.rpart(model.class.tree, cp = model.class.tree$cptable[which.min(model.class.tree$cptable[,"xerror"]),"CP"])

# trying out rattle package's "fancyRpartPlot"
fancyRpartPlot(model.class.tree.pruned, uniform=TRUE, main="Pruned Classification Tree")

# out of sample prediction (pruned tree)



out.sample.binary.class.tree = predict(model.class.tree.pruned, test, type = "class")
out.sample.binary.table <- table(test$response, out.sample.binary.class.tree, dnn = c("Truth", "Predicted"))

# Out-of-Sample Misclassification rate for pruned tree
1 - (out.sample.binary.table[1,1]+out.sample.binary.table[2,2])/sum(out.sample.binary.table)

#36.7% slightly worse than non-pruned tree