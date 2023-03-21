
#Q2
#In the regression prediction, the two major components of 
#error are reducible error and irreducible error. The 
#reducible error can further be decomposed into bias and
#variance. In mathematics, the expected prediction error can
# be expressed as EPE(Y,f_hat(x)) = E[(Y-f_hat(X))^2 |X=x]
# The f_hat(x) is a good estimate of the regression function f(x)
# and the f_hat(X)|X=x provides the estimated y. Squared error
# loss is used here. The EPE(Y, f_hat(x)) can be expressed as 
# E_D[(f(x) - f_hat(x))^2] + V_Y|X[Y|X=x], where the first term
# is the reducible error and the second term is irreducible error.
# D is the dataset where D=(xi, yi). The reducible error is the 
# mean squared error, calculated with the difference between 
# Y and predicted y. Bias and variance comprise the reducible error.
# Which is MSE(f(x), f_hat(x)) = (f(x)-E[f_hat(x)])^2
# + E[(f_hat(x) - E[f_hat(x)])^2]. From the equation, bias is the 
# difference between true y values and the expected value of predicted 
# y values, squared. The bias is high when the model underfits 
# the data, where the difference between true y and pred_y is large. In
# other words, the model does not capture the datapoints well.
# And the variance measures the difference between predicted y values 
# and expected pred_y values. In the case of overfitting, the model fits tightly
# into the given dataset, causing the large difference between pred_y 
# and E(pred_y). The model will suffer error when applied onto another dataset
# due to the high variance.
#  
# The irreducible error is the variance of Y given X=x, which is
# used to express the noise from the dataset. Such variance cannot
# be reduced by the model and it can be caused by measurement error
# or natural variability.


#Q3

library(dplyr)
library(ggplot2)
data(mtcars)
head(mtcars)

#Standardization
df =  mtcars %>% mutate_at(c('mpg', 'cyl','disp','hp','drat','wt',
                             'qsec','gear','carb'), ~(scale(.) %>% 
                                                        as.vector))
#drop the categorical variables
df = df[,c('mpg','cyl','disp','hp','drat','wt','qsec','gear','carb')]
apply(df,MARGIN=2,FUN=mean)
apply(df,MARGIN=2,FUN=sd)
# varify all non-categorical variables have mean=0 and sd=1

# use prcomp() to calculate the principal components, center and scale are 
# false because df is already centered
pca = prcomp(df, retx=TRUE, center=F, scale.=F)
sd = pca$sdev
# getting loadings matrix 
loadings = pca$rotation 
rownames(loadings) = colnames(df)
scores = pca$x

str(pca)
# str(pca) shows that the parameters of the pca object. sdev shows the 
# standard deviation for each of the 9 PCs. rotation shows the loadings matrix.
# The first loading vector places larger weight on cyl, disp, mpg and wt. And the
# second loading vector places larger weight on gear, carb and qsec. It implies 
# that cyl, disp and mpg have higher correlation with each other while 
# gear, carb and qsec are correlated to each other.

var <- sd^2
var.percent <- var/sum(var) * 100
dev.new()
barplot(var.percent, xlab="PC", ylab="Percent Variance", names.arg=1:length(var.percent), las=1, ylim=c(0,max(var.percent)), col="gray")
abline(h=1/ncol(df)*100, col="red")
# From the bar plot, the first 2 PC explains most of the variance in the data,
# and the reset of 7 PCs are not too important comparing with first 2.


variance_explained <- pca$sd^2/sum(pca$sd^2)
# The first PC explains 62.8% of variance and the second explains 23% of variance.

scree_df = data.frame(pc=c(1:9), var_exp = variance_explained)
ggplot(scree_df,aes(x=pc,y=var_exp)) +
  geom_line() +
  geom_point(size=3)+
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  scale_x_continuous("PC", labels = as.character(scree_df$pc), 
                     breaks = scree_df$pc)
# From the scree plot, we can use first 2 or 3 PCs to capture most of the variance
# in the data, with the elbow method. Using first 2 PCs, 86% of variance is explained.


library(devtools)
library(ggbiplot)
ggbiplot(pca) +
  scale_color_discrete(name = '') +
  theme(legend.direction = 'horizontal', legend.position = 'top')

# From the PCA biplot, with PC1 on x axis and PC2 on y axis, 
# it is easy to confirm that variables cyl, disp and wt are
# correlated to each other, from the loadings vector in brown. It is reasonable
# because more cylinders usually means higher displacement for a car.
# From the y axis, gear and qsec contributed more to the second PC.

ggbiplot(pca,labels=rownames(mtcars))

# This plot shows the name of cars instead of dots on the biplot. 

mtcars.country <- c(rep("Japan", 3), rep("US",4), rep("Europe", 7),
                    rep("US",3), "Europe", rep("Japan", 3), rep("US",4),
                    rep("Europe", 3), "US", rep("Europe", 3))

ggbiplot(pca,labels=rownames(mtcars),ellipse=T,groups=mtcars.country)
# Grouping the cars based on the country. We can see the US cars have a 
# large positive score on PC1, which also have more cylinders, higher disp and wt.
# Cars like Honda Civic and Corolla have smallest PC1, meaning those cars are
# more fuel efficient, which can also be seen from the mpg loading vector. 
# Cars in the middle such as mercedes 280 has average level of both 
# fuel efficiency and gear, carb or qsec. 
# In terms of clustering, it is obvious to group US cars as 1 cluster and 
# Euro Japananese cars as another cluster. The fuel efficiency (PC1) metrics 
# such as disp or mpg contributed most to the clear separation.  
# It can also be seen that European cars are more versatile.


#Q4
library(nnet)
library(caTools)
library(psych)

data(iris)
names(iris) = tolower(names(iris))
set.seed(100)
# use sample.split() to separate the train and test sets randomly.
train_rows = sample.split(1:nrow(iris),SplitRatio=2/3)
train = iris[train_rows,]
test = iris[!train_rows,]
table(train$species)
table(test$species)
# check class imbalance
# No obvious class imbalance is found from the output. The train set has
# a little bit more versicolor flowers, and therefore test set has a bit less.


describeBy(iris,group='species')
# From the describeBy() function, the sepal of virginica flowers are longer and 
# the setosa flowers are more circular, with smaller sepal length and larger
# sepal width. Setosa also has less standard deviation in the petal.


model = multinom(species ~ ., data=train)
summary(model)
pred = predict(model, type='class',newdata=test)
table(pred,test$species)
library(caret)
confusionMatrix(data=pred,reference=test$species)


# use pc instead original variables

iris_stan = iris %>% mutate_at(c('sepal.length','sepal.width','petal.length',
                               'petal.width'), ~(scale(.) %>% as.vector))
iris_stan = iris_stan[,c('sepal.length','sepal.width','petal.length',
                           'petal.width')]

apply(iris_stan,MARGIN=2,FUN=mean)
apply(iris_stan,MARGIN=2,FUN=sd)

pca = prcomp(iris_stan, retx=TRUE, center=F, scale.=F)
sd = pca$sdev
loadings = pca$rotation
rownames(loadings) = colnames(iris_stan)
scores = pca$x

str(pca)

var <- sd^2
var.percent <- var/sum(var) * 100
dev.new()
barplot(var.percent, xlab="PC", ylab="Percent Variance", 
        names.arg=1:length(var.percent), las=1, ylim=c(0,max(var.percent)), 
        col="gray")
abline(h=1/ncol(iris_stan)*100, col="red")

variance_explained <- pca$sd^2/sum(pca$sd^2)

plot(1:length(variance_explained), variance_explained, type = "b", 
     xlab = "Principal Component", ylab = "Variance Explained")

pc_iris = pca$x

set.seed(100)
train_rows = sample.split(1:nrow(pc_iris),SplitRatio=2/3)
pc_train = pc_iris[train_rows,]
pc_test = pc_iris[!train_rows,]
#train_pca <- as.data.frame(pc_train)
#train_pca$species <- train[ , 5]

#pc_train = pca$x
pc_train <- as.data.frame(pc_train)
pc_train$species <- train[, 5]

pc_test <- as.data.frame(pc_test)
#pred_pc <- predict(pca, newdata=pc_test) 
#test_pca <- as.data.frame(test_pca)

#test_pca$species <- test$species ???

model_pc = multinom(species ~ PC1+PC2, data=pc_train)
summary(model_pc)
pred_pc = predict(model_pc, type='class',newdata=pc_test)
pc_test$species = test[, 5]
table(pred_pc,pc_test$species)
confusionMatrix(data=pred_pc,reference=pc_test$species)



#Q5

df = read.csv('imports-85.csv',header=F)
names(df)=c('symboling','normalized-losses','make','fuel-type','aspiration',
              'num-of-doors','body-style','drive-wheels','engine-location','wheel-base',
              'length','width','height','curb-weight','engine-type','num-of-cylinders',
              'engine-size','fuel-system','bore','stroke','compression-ratio','horsepower',
              'peak-rpm','city-mpg','highway-mpg','price')

df = df[,c('normalized-losses','wheel-base','length','width','height',
           'curb-weight','engine-size','bore','stroke','compression-ratio',
           'horsepower','peak-rpm','city-mpg','highway-mpg','price')]

df[df=='?'] = NA
df = na.omit(df)
df = lapply(df,as.numeric)
df = data.frame(df)

#standarization
df =  df %>% mutate_at(names(df), ~(scale(.) %>% as.vector))
apply(df,MARGIN=2,FUN=mean)
apply(df,MARGIN=2,FUN=sd)

#calculate pc
pca = prcomp(df, retx=TRUE, center=F, scale.=F)
sd = pca$sdev
loadings = pca$rotation
rownames(loadings) = colnames(df)
scores = pca$x

#pca plots
variance_explained <- pca$sd^2/sum(pca$sd^2)
plot(1:length(variance_explained), variance_explained, type = "b", 
     xlab = "Principal Component", ylab = "Variance Explained")

ggbiplot(pca) +
  scale_color_discrete(name = '') +
  theme(legend.direction = 'horizontal', legend.position = 'top')







