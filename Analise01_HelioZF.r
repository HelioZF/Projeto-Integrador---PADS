# Carregando as bibliotecas
library(tidyverse)
library(pROC)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(yardstick)
library(rsample)
library(dplyr)
library(glmnet)
library(rpart)
library(partykit)
library(vip)
set.seed(907)

#importando os Dados
dados <- read.csv("dados_para_o_R.csv") # nolint
head(dados)

# C) ---------------------------------------------------------------------------
dados |>
  ggplot(aes(x = origin, y = sales, color = operates_within_2_years)) +
  geom_point()

# Criando uma tabela para comparar os modelos
models_tibble <- tibble(modelo = c("lm", "ridge", "LASSO", 
                                   "Arv.Decisão", "Rd.Forest"),
                        RQM = NA * length(modelo), RMSE = NA * length(modelo))

models_tibble

# i) regressão linear ----------------------------------------------------------

# Convertendo variáveis categóricas para fatores
dados_reg <- dados %>%
  mutate(across(where(is.character), as.factor))

# Verificando variáveis com apenas um nível e removendo-as
dados_modelo <- dados_reg %>%
  select(where(~ n_distinct(.) > 1))

glimpse(dados_modelo)
# Separando o database em treino e teste
split <- initial_split(dados_modelo, prop = 0.8)

treinamento <- training(split)
teste <- testing(split)

# Ajustando o modelo de regressão linear
fit <- lm(operates_within_2_years ~ ., data = treinamento)
summary(fit)

# Avaliando o modelo

y_pred <- predict(fit, newdata = teste)
y_real  <- teste$operates_within_2_years

# Criando um tibble com valores reais e previstos
resultados <- tibble(
  truth = y_real,     # Valores reais
  estimate = y_pred     # Valores previstos
)
# Calculando o R-quadrado (R²)
rsq_result <- rsq(resultados, truth = truth, estimate = estimate)
rmse_result <- rmse(resultados, truth = truth, estimate = estimate)

models_tibble$RQM[models_tibble$modelo == "lm"] <- rsq_result$.estimate
models_tibble$RMSE[models_tibble$modelo == "lm"] <- rmse_result$.estimate
models_tibble

# ii) regressão ridge ----------------------------------------------------------

# Separando os dados em treino e teste
split <- initial_split(dados_modelo, prop = 0.8)
treinamento <- training(split)
teste <- testing(split)

# Convertendo as variáveis preditoras e resposta para matriz
x_train <- model.matrix(operates_within_2_years ~ .,
                        data = treinamento)[, -1] # Exclui a coluna de intercepto #nolint
y_train <- treinamento$operates_within_2_years

x_test <- model.matrix(operates_within_2_years ~ ., data = teste)[, -1]
y_test <- teste$operates_within_2_years

# Ajustando o modelo Ridge com glmnet
ridge_model <- glmnet(x_train, y_train, alpha = 0)  # alpha = 0 para Ridge
# Validação cruzada para encontrar o melhor lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)

# Obtendo o melhor lambda
best_lambda <- cv_ridge$lambda.min
cat("Melhor lambda:", best_lambda, "\n")

# Fazendo previsões no conjunto de teste
predicoes <- predict(cv_ridge, s = best_lambda, newx = x_test)

# Calculando o RSQ para avaliar o modelo
resultados <- tibble(truth = y_test, estimate = as.vector(predicoes))

rsq_result <- rsq(resultados, truth = truth, estimate = estimate)
rmse_result <- rmse(resultados, truth = truth, estimate = estimate)

models_tibble$RQM[models_tibble$modelo == "ridge"] <- rsq_result$.estimate
models_tibble$RMSE[models_tibble$modelo == "ridge"] <- rmse_result$.estimate
models_tibble
# iii) regressão LASSO ---------------------------------------------------------

# ajustando o modelo
lasso_model <- glmnet(x_train, y_train, alpha = 1) # alpha =1 para Lasso

#enconrando o melhor lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
best_lambda <- cv_lasso$lambda.min
cat("Melhor lambda:", best_lambda, "\n")

# Fazendo previsões no conjunto de teste
predicoes <- predict(cv_lasso, s = best_lambda, newx = x_test)

# Calculando o RSQ para avaliar o modelo
resultados <- tibble(truth = y_test, estimate = as.vector(predicoes))

rsq_result <- rsq(resultados, truth = truth, estimate = estimate)
rmse_result <- rmse(resultados, truth = truth, estimate = estimate)

models_tibble$RQM[models_tibble$modelo == "LASSO"] <- rsq_result$.estimate
models_tibble$RMSE[models_tibble$modelo == "LASSO"] <- rmse_result$.estimate

models_tibble

# iv) árvore de decisão ------------------------------------------
library(rpart.plot)
tree <- rpart(operates_within_2_years ~ ., data = treinamento, method = "anova")
windows() + rpart.plot(tree, roundint = FALSE)
windows() + plotcp(tree)

cp_ot <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
cp_ot

tree <- prune(tree, cp = cp_ot)
windows() + rpart.plot(tree, roundint = FALSE)
tree_predict <- predict(tree, newdata = teste)

resultados_arv_decisao <- tibble(truth = y_test, estimate = tree_predict)
rsq_result <- rsq(resultados_arv_decisao, truth = truth, estimate = estimate)
rmse_result <- rmse(resultados_arv_decisao, truth = truth, estimate = estimate)

models_tibble$RQM[models_tibble$modelo == "Arv.Decisão"] <- rsq_result$.estimate
models_tibble$RMSE[models_tibble$modelo == "Arv.Decisão"] <- rmse_result$.estimate

models_tibble

# v) floresta aleatória
library(ranger)

rd_forest_model <- ranger(
  dependent.variable.name = "operates_within_2_years",
  data = treinamento,
  num.trees = 500,
  mtry = floor(sqrt(ncol(treinamento) - 1)),
  importance = 'impurity', 
  seed = 123
)

predicoes_rd_forest <- predict(rd_forest_model, data = teste)$predictions

resultados_rd_forest <- tibble(truth = y_test, estimate = predicoes_rd_forest)
rsq_result_rd_forest <- rsq(resultados_rd_forest, truth = truth, estimate = estimate)
rmse_result_rd_forest <- rmse(resultados_rd_forest, truth = truth, estimate = estimate)

models_tibble$RQM[models_tibble$modelo == "Rd.Forest"] <- rsq_result_rd_forest$.estimate
models_tibble$RMSE[models_tibble$modelo == "Rd.Forest"] <- rmse_result_rd_forest$.estimate

models_tibble