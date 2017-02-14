# Data Mining
# HW 4 Problem 1
#  Create Linear Model and Regression Tree on Boston Housing data, compare results

##### packages #####

packages<- c("MASS","caret","ggplot2","GGally","boot","rpart","rpart.plot")

install.packages(packages)
lapply(as.list(packages),library, character.only = TRUE)

##### Partition Train/Test #####

data("Boston")

set.seed(1984)

index <- sample(1:nrow(Boston),nrow(Boston)*.7)

# Alternate set of data

set.seed(5001)

index <- sample(1:nrow(Boston),nrow(Boston)*.7)


train <- Boston[index,]

test <- Boston[-index,]


##### Summary Stats #####


str(Boston)
summary(Boston) #no NA values



##### EDA #####

#Scatterplot Matrix and Density Plots

ggpairs(Boston) # some variable have clear correlation, some predictors 
# have clear relationship with medv, there are outliers present


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
# 5 variable have high correlation (>.7)

# loop of boxplot based on example here: https://www.r-bloggers.com/ggplot2-graphics-in-a-loop/
# need to fix axis titles and scale issues from outliers
boxplot.loop <- function(x, na.rm = TRUE, ...) {
  numer <- sapply(x, is.numeric)
  nm <- names(x[,numer])
  for (i in seq_along(nm)) {
    plots <- ggplot(data = x,aes(x = factor(0), y = x[,nm[i]])) + geom_boxplot()
    ggsave(plots,filename=paste("boxplot",nm[i],".png",sep="_"))
  }
}

boxplot.loop(train) 
# outliers are present in many variables

##### Linear Model with no variable manipulation #####

linreg.model <- lm(medv ~ ., data = train)

summary(linreg.model)

##### variable selection #####

# set null and full model for use in variable selection
nullmodel = lm(medv ~ 1, data = train)
fullmodel = lm(medv ~ ., data = train)

#bw
model.backward = step(fullmodel, direction = "backward")
#fw
model.forward = step(nullmodel, scope = list(lower = nullmodel, upper = fullmodel), 
                     direction = "forward")
#step
model.stepwise = step(nullmodel, scope = list(lower = nullmodel, upper = fullmodel), 
                      direction = "both")

# Summary

stepwise.summary <- summary(model.stepwise)
# all came to same model, so will pick one arbitrarily (stepwise)

#Diagnostic Plots

plot(model.stepwise)
# data does not seem to fit underlying assumptions perfectly based on the diagnotics plots


##### Linear Model Fit #####
# MSE

(stepwise.summary$sigma)^2

# R-squared 

stepwise.summary$r.squared

# Adjusted R-squared  (penalizes model complexity)

stepwise.summary$adj.r.squared

# AIC and BIC of the model, these are information criteria. Smaller values indicate better fit.

AIC(model.stepwise)

BIC(model.stepwise)

# Out of sample prediction

# pi is a vector that contains predicted values for test set.

pi <- predict(object = model.stepwise, newdata = test)


# Mean Squared Error (MSE): average of the squared differences between the predicted and actual values

mean((pi - test$medv)^2)
# 21.32945


#3-fold Cross Validation

model.glm.cv <- glm(formula = medv ~ lstat + rm + ptratio + dis + nox + chas + 
     zn + crim + rad + tax + black, data = Boston)

linear.cv <- cv.glm(data = Boston, glmfit = model.glm.cv, K = 3)
linear.cv$delta[2]
# 23.57908


##### Regression Tree #####

model.tree <- rpart(medv ~., data = train)

# plot model
rpart.plot(model.tree,tweak = 1.2)

# out of sample prediction (full tree)

test.pred = predict(model.tree, test)

# MSE (full tree)

mean((test.pred - test$medv)^2)


##### Prune Tree #####

plotcp(model.tree) # prune tree to size of 6

printcp(model.tree)

model.tree.pruned <- prune.rpart(model.tree, cp = model.tree$cptable[which.min(model.tree$cptable[,"xerror"]),"CP"])

# trying out rattle package's "fancyRpartPlot"
fancyRpartPlot(model.tree.pruned, uniform=TRUE, main="Pruned Classification Tree")

# out of sample prediction (pruned tree)

test.pred.prune = predict(model.tree.pruned, test)

# MSE (pruned tree)

mean((test.pred.prune - test$medv)^2)

