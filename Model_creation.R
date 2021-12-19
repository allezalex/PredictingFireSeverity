## Individual Project for Predicting Forest Fire burnt surface
## Created by AllezAlex on the 11/12/2021

## This is my individual submission for the Final assigment 
## My code's aim is to create a model capable of using classification to 
## predict a fire's class size (in terms of area of impact) based on meteorogical data

## Instructions
## 1. Download and unzip the data file named : "forestfires.csv"
## 2. Set your data_dir variable to the path to the data directory
## 3. Run the code. It should take about 5-10 minutes
## 4. Visualise the model results

## ==============================
## Main variables
## ==============================

## Set the path to your data here
data_dir <- "~/data/forestfires.csv"
fire_data <- read.csv(data_dir)
attach(fire_data)

## ==============================
## Load packages
## ==============================

# install.packages("ggpubr")
# install.packages("moments")
# install.packages("lightgbm")
library(dplyr)
library(fastDummies)
library(ggplot2)
library(ggpubr)
library(caTools)
library(corrplot)
library(tree)
library(rpart) 
library(rpart.plot)
library(randomForest)
library(gbm) 
library(caTools)
library("purrr")
library(moments) # To calculate skewness
library(lightgbm)
library(methods)
set.seed (1) #To ease reproducibility

## ==============================
## Pre-Processing & Function Definition
## ==============================

# Check to see if any data missing - There's seems to be 0 missing values
sum(is.na(fire_data))

# Function to calculate the accuracy automatically for hyperparameter tuning
# Credits to Daniel Johnson from Guru99

accuracy_tune <- function(fit) {
  tree.pred <- predict(fit, test, type = 'class')
  mytree_cm <- table(test$area_cat, tree.pred)
  accuracy_Test <- sum(diag(mytree_cm)) / sum(mytree_cm)
  print(paste('Accuracy of model:', accuracy_Test))
}


## ==============================
## Explore Data
## ==============================

## General Exploration : When do fire happen?
# We notice here that month and day are entered as character variables, therefore we will convert them
# into factors using a defined order as level, so that they'll be sorted into the correct order when plotted

month_order <- c("jan", "feb", "mar",
                 "apr", "may", "jun",
                 "jul", "aug", "sep",
                 "oct", "nov", "dec")
day_order <- c("mon", "tue", "wed", "thu", "fri", "sat", "sun")
fire_data$month <- factor(fire_data$month, levels = month_order)
fire_data$day <- factor(fire_data$day, levels = day_order)
attach(fire_data)

## Fire Occurence monthly distribution

fires_by_month <- fire_data %>%
  group_by(month) %>%
  summarize(total_fires = n())

theme_set(theme_bw()) 

ggplot(data = fires_by_month, aes(x = month, y = total_fires, fill = "Fire Occurence")) +
  geom_col() +
  labs(
    title = "Monthly Distribution of Forest Fire",
    y = "Amount of fires",
    x = "Month",
    fill = "Legend"
  )

## General Statistics 
# For easier use of month and day string variables we convert them to numeric values
fire_data$day <- as.numeric(fire_data$day)
fire_data$month <- as.numeric(fire_data$month)

# Making sure our position variables are used as factors
fire_data$X <- as.factor(fire_data$X)
fire_data$Y <- as.factor(fire_data$Y)
attach(fire_data)

## Dependent variable 
hist(area, 30)
## As we can see our dependant variable: Area, is quite skewed on the left, we will have to work on it to avoid issues

# Distribution of data across each variables of our dataset
# Credits to Rebecca Elizabeth Kitching for the base code

par(mfrow=c(2,6),mar=c(3.90, 4.25, 2.5, 0.5))
for (variables in 1:(dim(fire_data)[2]-1)){
  selectedvar = fire_data[,variables]
  d <- density(selectedvar)
  plot(d, main = names(fire_data[variables]),xlab="") 
  curve(dnorm(x, mean=mean(selectedvar), sd=sqrt(var(selectedvar))), 
        col="red", lwd=2, add=TRUE, yaxt="n", lty=2)
  polygon(d)
  title("Density plots for our 12 Predictors", line = -39,outer = TRUE)
  }

# We can also visualise our data through a summary table
summary(fire_data)

## Independent variables
# From our summary we can see a few variables are
# Right Skewed (Mode < Median < Mean) : ISI, Rain
# Left Skewed (Mean < Median < Mode) : FFMC

## Solving Right Skewness 
# For ISI and rain we will apply a sqrt transformation
skewness(fire_data$ISI, na.rm = TRUE)
fire_data$ISI <- sqrt(fire_data$ISI)
skewness(fire_data$rain, na.rm = TRUE)
fire_data$rain <- sqrt(fire_data$rain) # rain is severely skewed so we use inverse transformation

## Solving Left Skewness 
skewness(fire_data$FFMC, na.rm = TRUE)
fire_data$FFMC <- sqrt(max(fire_data$FFMC+1) - fire_data$FFMC)

## Correlation Matrix
pairs(fire_data[,c(1,2,seq(5,12))],panel = panel.smooth)

par(mfrow=c(1,1))
M <- cor(fire_data[,1:dim(fire_data)[2]-1])
corrplot(M, method="color", outline = TRUE,type="lower",order = "hclust",
         tl.col="black", tl.srt=45, diag=FALSE,tl.cex = 1,mar=c(0,0,3,0),
         title="Correlation Matrix between Predictor and Outcome variables")

# Visualisation of fire occurences on map
# Scattering the plot with the help of the location
# read picture
library(png)
img <- readPNG("montesinho_map.png")
# plot with picture as layer
library(ggplot2)
ggplot(data= fire_data,mapping = aes(X,Y)) +
  annotation_raster(img, xmin = 0.25, xmax = 10, ymin = 0.09, ymax = 9.5) +
  geom_point()

## ======================================================
## Feature Selection using Random Forest and Boosted Tree
## ======================================================

# We are looking at using classification techniques thus we should have a categorical dependent variable
# We will classify our area data between :
# Class A - one-fourth acre or less; 
# Class B - more than one-fourth acre, but less than 10 acres; 
# Class C - 10 acres or more, but less than 100 acres;
# Class D - 100 acres or more, but less than 300 acres;
# Class E - 300 acres or more, but less than 1,000 acres;
# From National Wildfire Coordinating Group Website

# first we convert our data to acre
fire_data$acre = fire_data$area*2.471
fire_data$acre

attach(fire_data)
hist(acre, 30)

fire_data_cleaned = fire_data
fire_data_cleaned <- dummy_cols(fire_data_cleaned, select_columns = c("month","day"), remove_selected_columns = TRUE)

dim(fire_data_cleaned)
hist(fire_data_cleaned$acre, 30)

 # create our categorical variable (with multiple possible values) from existing acre variable
fire_data_cleaned$area_cat <- as.factor(ifelse(acre <= 0.25, 'A',
                                                ifelse( acre > 0.25 & acre < 10, 'B',
                                                        ifelse(acre >= 10, 'C','NA'))))

table(fire_data_cleaned$area_cat) # We now have our three types of fire and will move forward to predicting them
fire_data_cleaned <- subset( fire_data_cleaned, select = -c(acre, area))
attach(fire_data_cleaned)

## -------------
## Random Forest
## -------------

myforest=randomForest(area_cat~., data=fire_data_cleaned, importance=TRUE, na.action = na.omit, do.trace=100)
myforest

#### See importance of predictors
importance(myforest)
varImpPlot(myforest)

# Most important weather related predictors: Temperature and DC (Drought Code)
# If we remove these predictors, Accuracy will decrease by about 3%
# If we remove these predictors, Gini index (or Gini impurity) will decreasy by over 15pts.

fire_data_cleaned <- subset(fire_data_cleaned, select = -c(day_4, month_5, day_7, month_6, day_5))
attach(fire_data_cleaned)

## ==============================
## Create Model
## ==============================

# Data Partionning 
split = sample.split(fire_data_cleaned$area_cat, SplitRatio = 0.7)
train = subset(fire_data_cleaned, split==TRUE)
test = subset(fire_data_cleaned, split==FALSE)

print(dim(train))
print(dim(test))
table(train$area_cat) # To check the distribution of area cat and make sure we have good ratio
table(test$area_cat)

# Our objective here is to use models such as Random Forest and Boosted Tree and perform parameter tuning
# To build the best model to predict a fire's severity

## -------------------
## Classficiation Tree
## -------------------
# To store our results
results = data.frame()

mytree = rpart(area_cat~., data=train)
summary(mytree)
test_pred <- predict(mytree, test, type="class")
sum(test$area_cat==test_pred)/nrow(test)
results <- rbind(results, sum(test$area_cat==test_pred)/nrow(test))

# testing main hyperparameters
metric_best = 0
params_best <- c(0,0)

split_tree = sample.split(train$area_cat, SplitRatio = 0.85)
train_tree  = subset(train, split_tree==TRUE)
test_tree = subset(train, split_tree==FALSE)

for (minsplit in seq(5,50,5))
{
  for (maxdepth in seq(3,10,1))
  {
    control <- rpart.control(minsplit =minsplit,
                             maxdepth = maxdepth)
    mytree = rpart(area_cat~., control=control, method = "class",data=train_tree)    
    test_tree_pred <- predict(mytree, test_tree, type="class")
    metric_tree <- sum(test_tree$area_cat==test_tree_pred)/nrow(test_tree)
    print(c("minsplit:",minsplit, "maxdepth:",maxdepth, "-- metric:",round(metric_tree,digits=4)))
    if (metric>metric_best){
      metric_best <- metric_tree
      params_best <-c(minsplit,maxdepth)
    }
  }
}

# testing cps
control <- rpart.control(minsplit =params_best[1],
                         maxdepth = params_best[2],
                         cp=0.0001)
myoverfittedtree=rpart(area_cat~.,data = train, control=control, method="class")
printcp(myoverfittedtree)
plotcp(myoverfittedtree)

# To find the cp value that minimizes the error, we use the following command:
opt_cp=myoverfittedtree$cptable[which.min(myoverfittedtree$cptable[,"xerror"]),"CP"]
opt_cp

# Build our final model
control <- rpart.control(minsplit =params_best[1],
                         maxdepth = params_best[2],
                         cp=opt_cp)
final_tree=rpart(area_cat~.,data = train, control=control, method="class")
test_pred <- predict(final_tree, test, type="class")
sum(test$area_cat==test_pred)/nrow(test)
results <- rbind(results,sum(test$area_cat==test_pred)/nrow(test))

## -------------
## Random Forest
## -------------

# Create a random forest model with default paramters and check its OOB scores
myRandomForest <- randomForest(area_cat~., ntree=750, data=train, importance=TRUE, na.action = na.omit)
plot(myRandomForest)
myRandomForest <- if (is.null(myRandomForest$test$err.rate)) {colnames(myRandomForest$err.rate)} else {colnames(myRandomForest$test$err.rate)}
legend("top", cex =0.5, legend=myRandomForest, lty=c(1,2,3,4), col=c(1,2,3,4), horiz=T)

# default random forest performance
myRandomForest <- randomForest(area_cat~., ntree=750, data=train, importance=TRUE, na.action = na.omit)
test_pred <- predict(myRandomForest, test)
sum(test$area_cat==test_pred)/nrow(test)
results <- rbind(results,sum(test$area_cat==test_pred)/nrow(test))

metric_best = 0
params_best <- c(0,0)

split_rf = sample.split(train$area_cat, SplitRatio = 0.85)
train_rf  = subset(train, split_rf==TRUE)
test_rf = subset(train, split_rf==FALSE)

for (ntree in seq(50,750,50))
{
  for (mtry in seq(3,10,1))
  {
    myforest=randomForest(area_cat~., ntree=ntree, mtry=mtry, data=train_rf, importance=TRUE, na.action = na.omit)
    test_rf_pred <- predict(myforest, test_rf)
    metric <- sum(test_rf$area_cat==test_rf_pred)/nrow(test_rf)
    # print(c("ntree:",ntree, "mtry:",mtry, "-- metric:",round(metric,digits=4)))
    if (metric>metric_best){
      metric_best <- metric
      params_best <-c(ntree,mtry)
    }
  }
}

print("Best model uses:")
print(c("ntree:",params_best[1], "mtry:",params_best[2], "-- metric:",round(metric_best, digits=4)))

mybestforest=randomForest(area_cat~., ntree=params_best[1], mtry=params_best[2], data=train, importance=TRUE, na.action = na.omit)
test_pred <- predict(myforest, test)
sum(test$area_cat==test_pred)/nrow(test)
results <- rbind(results,sum(test$area_cat==test_pred)/nrow(test))

print(results)

