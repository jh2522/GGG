library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ranger)
library(kknn)
library(naivebayes)
library(discrim)
library(kernlab)
library(keras)
library(parsnip)
library(bonsai)
library(lightgbm)
library(discrim)
sample <- "sample_submission.csv"
test <- "test.csv"
train <- "train.csv"
#train2 <- "trainWithMissingValues.csv"
sample1 <- vroom(sample)
test1 <- vroom(test)
train1 <- vroom(train)
train1
#train3 <- vroom(train2)
print(train3, n=400)
my_recipe <- recipe(type~., data=train3) %>% 
  step_impute_bag(hair_length, rotting_flesh, bone_length)
my_rec <- prep(my_recipe)  
train4 <- bake(my_rec, train3)
rmse_vec(train1[is.na(train3)],train4[is.na(train3)])
mycleandata
#randomforest
mycleandata <- train1 %>% 
  mutate(type=as.factor(type))
my_recipe <- recipe(type~., data=mycleandata) %>% 
  step_mutate(color=factor(color)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_normalize(all_numeric_predictors()) 
my_recipe1 <- prep(my_recipe)
bake(my_recipe1, new_data=mycleandata)
forest_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),
                          trees=500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

forest_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(forest_mod)

tuning_grid_forest <- grid_regular(mtry(range=c(1,5)), min_n(), levels=10)

folds_forest <- vfold_cv(mycleandata, v = 5, repeats=1)

CV_results_pen <- forest_wf %>% 
  tune_grid(resamples=folds_forest, grid=tuning_grid_forest, metrics=metric_set(accuracy))
CV_results_pen
bestTune_forest <- CV_results_pen %>% 
  select_best(metric="accuracy")
bestTune_forest
final_wf_forest <- 
  forest_wf %>% 
  finalize_workflow(bestTune_forest) %>% 
  fit(data=mycleandata)

predict <- final_wf_forest %>% 
  predict(new_data=test1, type="class")
predict
kaggle_submission <- predict %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_class) %>% 
  rename(type=.pred_class)
vroom_write(x=kaggle_submission, file="./GGGForest12.csv", delim=",")

#Knn

knn_model <- nearest_neighbor(neighbors=10) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")
knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_model)



final_wf_knn <- 
  knn_wf %>% 
  fit(data=mycleandata)

predict_knn <- final_wf_knn %>% 
  predict(new_data=test1, type="class")

kaggle_submission <- predict %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_class) %>% 
  rename(type=.pred_class)
vroom_write(x=kaggle_submission, file="./GGGKNn12.csv", delim=",")

#naivebayes

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_model)

tuning_grid_nb <- grid_regular(Laplace(), smoothness(), levels=10)

folds_nb <- vfold_cv(mycleandata, v = 5, repeats=1)

CV_results_nb <- nb_wf %>% 
  tune_grid(resamples=folds_nb, grid=tuning_grid_nb, metrics=metric_set(roc_auc))

bestTune_nb <- CV_results_nb %>% 
  select_best(metric="roc_auc")

final_wf_nb <- 
  nb_wf %>% 
  finalize_workflow(bestTune_nb) %>% 
  fit(data=mycleandata)

predict <- final_wf_nb %>% 
  predict(new_data=test1, type="class")


kaggle_submission <- predict %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_class) %>% 
  rename(type=.pred_class)
vroom_write(x=kaggle_submission, file="./GGGnaive12.csv", delim=",")

#SVM

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

rad_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(svmRadial)

tuning_grid_rad <- grid_regular(rbf_sigma(), cost(), levels=10)

folds_rad <- vfold_cv(mycleandata, v = 5, repeats=1)

CV_results_rad <- rad_wf %>% 
  tune_grid(resamples=folds_rad, grid=tuning_grid_rad, metrics=metric_set(roc_auc))
CV_results_rad
bestTune_rad <- CV_results_rad %>% 
  select_best(metric="roc_auc")
bestTune_rad

final_wf_rad <- 
  rad_wf %>% 
  finalize_workflow(bestTune_rad) %>% 
  fit(data=mycleandata)

predict <- final_wf_rad %>% 
  predict(new_data=test1, type="class")

kaggle_submission <- predict %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_class) %>% 
  rename(type=.pred_class)
vroom_write(x=kaggle_submission, file="./GGGsvm12.csv", delim=",")


# neural networks
library(remotes)
library(tensorflow)
library(keras)
library(reticulate)
install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
1
keras::install_keras()
tensorflow::install_tensorflow()
mycleandata <- train1 %>% 
  mutate(type=as.factor(type))
my_recipe <- recipe(type~., data=mycleandata) %>% 
  update_role(id, new_role="id") %>% 
  step_mutate(color=factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)
nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>% 
  set_engine("keras") %>% 
  set_mode("classification")

nn_tunegrid <- grid_regular(hidden_units(range=c(1, 1000)), levels = 5)

nn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nn_model)


folds_nn <- vfold_cv(mycleandata, v = 5, repeats=1)


CV_results_nn <- nn_wf %>% 
  tune_grid(resamples=folds_nn, grid=nn_tunegrid, metrics=metric_set(accuracy))
CV_results_rad
bestTune_rad <- CV_results_rad %>% 
  select_best(metric="roc_auc")
bestTune_rad
CV_results_nn %>% collect_metrics() %>% 
  filter(.metric=="accuracy") %>% 
  ggplot(aes(x=hidden_units, y = mean)) + geom_line()
tuned_nn
final_wf_rad <- 
  rad_wf %>% 
  fit(data=mycleandata)
?hidden_units

# Boost
mycleandata <- train1 %>% 
  mutate(type=as.factor(type))
my_recipe <- recipe(type~., data=mycleandata) %>% 
  step_mutate(color=factor(color)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_normalize(all_numeric_predictors()) 

boost_model <- boost_tree(tree_depth=10, trees=100,learn_rate = .1) %>% 
  set_engine("lightgbm") %>% 
  set_mode("classification")

boost_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(boost_model)

tuning_grid_boost <- grid_regular(learn_rate(), tree_depth(), levels=5)

folds_boost <- vfold_cv(mycleandata, v = 5, repeats=1)

CV_results_boost <- boost_wf %>% 
  tune_grid(resamples=folds_boost, grid=tuning_grid_boost, metrics=metric_set(accuracy))
CV_results_boost
bestTune_boost <- CV_results_boost %>% 
  select_best(metric="accuracy")
bestTune_boost

final_wf_boost <- 
  boost_wf %>% 
  #finalize_workflow(bestTune_boost) %>% 
  fit(data=mycleandata)

predict <- final_wf_boost %>% 
  predict(new_data=test1, type="class")

kaggle_submission <- predict %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_class) %>% 
  rename(type=.pred_class)
vroom_write(x=kaggle_submission, file="./GGGboost12.csv", delim=",")
