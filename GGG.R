library(tidyverse)
library(tidymodels)
library(vroom)
sample <- "sample_submission.csv"
test <- "test.csv"
train <- "train.csv"
train2 <- "trainWithMissingValues.csv"
sample1 <- vroom(sample)
test1 <- vroom(test)
train1 <- vroom(train)
train3 <- vroom(train2)
print(train3, n=400)
my_recipe <- recipe(type~., data=train3) %>% 
  step_impute_bag(hair_length, rotting_flesh, bone_length)
my_rec <- prep(my_recipe)  
train4 <- bake(my_rec, train3)
rmse_vec(train1[is.na(train3)],train4[is.na(train3)])
