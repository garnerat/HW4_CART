# Data Mining
# HW 4 Problem 2
#  


##### install packages #####


packages <- c("ggplot2","GGally","rpart","caret","leaps","boot","reshape2")

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

set.seed(1984)

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


##### Logistic Regression #####

model.logistic <- glm(response ~ . ,family = binomial, data = train)
summary(model.logistic)

# AIC

step.logistic <- step(model.logistic)

# BIC

step.logistic.BIC <- step(model.logistic, k = log(nrow(train)))


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
  result[i, 2] <- cost1(train$response, predict(step.logistic, type = "response"))
}
plot(result, ylab = "Cost in Training Set") #.23 is cutoff with smallest cost

###### Confusion Matrix, ROC, AUC #####

# Confusion Matrix
prob.outsample <- predict(step.logistic, test, type = "response")

prob.outsample.binary <- as.numeric(predict(step.logistic, test, type = "response") > 0.23)

confusionMatrix(prob.outsample.binary,test$response)

# ROC
install.packages("ROCR")
library(ROCR)
pred <- prediction(prob.outsample, test$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

# AUC
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values

