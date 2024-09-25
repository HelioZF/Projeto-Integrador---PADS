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

# criando tibble para comparar modelos:
models_tibble <- tibble(modelo = c("glm", "ridge", "LASSO", 
                                   "Arv.Decisão", "Rd.Forest"),
                        Accuracy = NA * length(modelo), Precision = NA * length(modelo), ROC_Curve = NA * length(modelo))
models_tibble

# Definindo a função para calcular as métricas e salvar no models_tibble
adiciona_tabela <- function(modelo_nome, resultados, models_tibble) {
  
  # Reordenando os níveis da variável 'truth'
  resultados$truth <- factor(resultados$truth, levels = c(1, 0))
  # Reordenando os níveis da variável 'estimate' para corresponder a 'truth'
  resultados$estimate <- factor(resultados$estimate, levels = c(1, 0))


  
  # Calculando as métricas
  accuracy_result <- yardstick::accuracy(resultados, truth = truth, estimate = estimate)
  precision_result <- yardstick::precision(resultados, truth = truth, estimate = estimate)
  roc_auc_result <- yardstick::roc_auc(resultados, truth = truth, .pred_1)
  
  # Atualizando a tabela models_tibble com os resultados
  models_tibble$Accuracy[models_tibble$modelo == modelo_nome] <- accuracy_result$.estimate
  models_tibble$Precision[models_tibble$modelo == modelo_nome] <- precision_result$.estimate
  models_tibble$ROC_Curve[models_tibble$modelo == modelo_nome] <- roc_auc_result$.estimate
  
  # Retornando a tabela atualizada
  return(models_tibble)
}

#sarrafo universal para classificação de todos os modelos
sarrafo <- 0.5

# Separando os dados em treino e teste
split <- initial_split(dados, prop = 0.8)
treinamento <- training(split)
teste <- testing(split)

treinamento$operates_within_2_years <- factor(treinamento$operates_within_2_years, levels = c(0, 1))
teste$operates_within_2_years <- factor(teste$operates_within_2_years, levels = c(0, 1))

# i) regressão logistica ----------------------------------------------------------

# Preparando os dados
x_train <- model.matrix(operates_within_2_years ~ ., data = treinamento)[, -1]
y_train <- treinamento$operates_within_2_years

x_test <- model.matrix(operates_within_2_years ~ ., data = teste)[, -1]
y_test <- teste$operates_within_2_years

# Definindo a sequência de lambdas (incluindo valores muito pequenos)
lambda_seq <- 10^seq(-6, 0, length = 100)

# Ajustando o modelo com cv.glmnet()
cv_logistic <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial", lambda = lambda_seq)

# Obtendo o melhor lambda
best_lambda <- cv_logistic$lambda.min
cat("Melhor lambda:", best_lambda, "\n")

# Fazendo previsões no conjunto de teste
y_pred_prob <- predict(cv_logistic, s = best_lambda, newx = x_test, type = "response")
y_pred_prob <- as.numeric(y_pred_prob)

# Obtendo as previsões de classe com base no threshold
y_pred <- as.factor(ifelse(y_pred_prob > sarrafo, 1, 0))

# Valores reais das classes no conjunto de teste
y_real <- as.factor(y_test)

# Criando um tibble com os resultados
resultados <- tibble(
  truth = y_real,
  estimate = y_pred,
  .pred_1 = y_pred_prob
)

# Atualizando a tabela models_tibble
models_tibble <- adiciona_tabela("glm", resultados, models_tibble)

# ii) regressão ridge ----------------------------------------------------------

# Convertendo as variáveis preditoras e resposta para matriz
x_train <- model.matrix(operates_within_2_years ~ ., data = treinamento)[, -1]
y_train <- treinamento$operates_within_2_years

x_test <- model.matrix(operates_within_2_years ~ ., data = teste)[, -1]
y_test <- teste$operates_within_2_years

# Ajustando o modelo Ridge com glmnet
ridge_model <- glmnet(x_train, y_train, alpha = 0, family = "binomial")

# Validação cruzada para encontrar o melhor lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial")

# Obtendo o melhor lambda
best_lambda <- cv_ridge$lambda.min
cat("Melhor lambda:", best_lambda, "\n")

# Fazendo previsões no conjunto de teste
y_pred_prob <- predict(ridge_model, s = best_lambda, newx = x_test, type = "response")

# Convertendo as probabilidades para vetor numérico
y_pred_prob <- as.numeric(y_pred_prob)

# Obtendo as previsões de classe com base no threshold
y_pred <- as.factor(ifelse(y_pred_prob > sarrafo, 1, 0))

# Valores reais das classes no conjunto de teste
y_real <- as.factor(teste$operates_within_2_years)

# Criando um tibble com os resultados
resultados <- tibble(
  truth = y_real,
  estimate = y_pred,
  .pred_1 = y_pred_prob
)

models_tibble <- adiciona_tabela("ridge", resultados, models_tibble)
# iii) regressão LASSO ---------------------------------------------------------


# ajustando o modelo
lasso_model <- glmnet(x_train, y_train, alpha = 1, family = "binomial") # alpha =1 para Lasso

#enconrando o melhor lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
best_lambda <- cv_lasso$lambda.min
cat("Melhor lambda:", best_lambda, "\n")

# Fazendo previsões no conjunto de teste
y_pred_prob <- predict(lasso_model, s = best_lambda, newx = x_test, type = "response")

# Convertendo as probabilidades para vetor numérico
y_pred_prob <- as.numeric(y_pred_prob)

# Obtendo as previsões de classe com base no threshold
y_pred <- as.factor(ifelse(y_pred_prob > sarrafo, 1, 0))

# Valores reais das classes no conjunto de teste
y_real <- as.factor(teste$operates_within_2_years)

# Criando um tibble com os resultados
resultados <- tibble(
  truth = y_real,
  estimate = y_pred,
  .pred_1 = y_pred_prob
)

models_tibble <- adiciona_tabela("LASSO", resultados, models_tibble)



# iv) árvore de decisão ------------------------------------------
library(rpart.plot)
tree <- rpart(operates_within_2_years ~ ., data = treinamento, method = "class")

# Visualizando a árvore e o gráfico de CP
windows()
rpart.plot(tree, roundint = FALSE)
windows()
plotcp(tree)

# Encontrando o CP ótimo e podando a árvore
cp_ot <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
tree <- prune(tree, cp = cp_ot)
windows()
rpart.plot(tree, roundint = FALSE)

# Fazendo previsões no conjunto de teste
y_pred_prob <- predict(tree, newdata = teste, type = "prob")[, 2]  # Probabilidade da classe 1

# Aplicando o threshold 'sarrafo' para obter as previsões de classe
y_pred <- as.factor(ifelse(y_pred_prob > sarrafo, 1, 0))

# Convertendo os valores reais em fator
y_real <- as.factor(teste$operates_within_2_years)

# Criando o tibble de resultados
resultados <- tibble(
  truth = y_real,       # Valores reais
  estimate = y_pred,    # Valores previstos
  .pred_1 = y_pred_prob # Probabilidades previstas para a classe 1
)

# Calculando a AUC da curva ROC
models_tibble <- adiciona_tabela("Arv.Decisão", resultados, models_tibble)

# v) floresta aleatória ------------------------------------------

# Criando folds de validação cruzada a partir do conjunto de treinamento
folds <- vfold_cv(treinamento, v = 5, strata = operates_within_2_years)

# Inicializando uma lista para armazenar os resultados
results_list <- list()

# Loop sobre cada fold
for (i in seq_along(folds$splits)) {
  
  # Extraindo o split atual
  split <- folds$splits[[i]]
  
  # Obtendo os dados de treinamento e validação para este fold
  train_data <- analysis(split)
  valid_data <- assessment(split)
  
  # Treinando o modelo no conjunto de treinamento com probability = TRUE
  rd_forest_model <- ranger(
    dependent.variable.name = "operates_within_2_years",
    data = train_data,
    num.trees = 500,
    mtry = floor(sqrt(ncol(train_data) - 1)),
    importance = 'impurity',
    seed = 123,
    probability = TRUE  # Importante para obter probabilidades
  )
  
  # Fazendo previsões no conjunto de validação
  predictions <- predict(rd_forest_model, data = valid_data)
  
  # Verificando se predictions$predictions é uma matriz
  if (is.matrix(predictions$predictions)) {
    # Acessando a coluna correspondente à classe "1"
    y_pred_prob <- predictions$predictions[, "1"]
  } else {
    # Caso contrário, acessamos a probabilidade diretamente
    y_pred_prob <- predictions$predictions
  }
  
  y_pred <- as.factor(ifelse(y_pred_prob > sarrafo, 1, 0))
  
  y_real <- as.factor(valid_data$operates_within_2_years)
  
  # Armazenando os resultados em um tibble
  resultados <- tibble(
    truth = y_real,       # Valores reais
    estimate = y_pred,    # Valores previstos
    .pred_1 = y_pred_prob # Probabilidades previstas para a classe '1'
  )
  
  results_list[[i]] <- resultados
}

# Combinando os resultados de todos os folds
resultados_cv <- bind_rows(results_list)

# Atualizando a tabela models_tibble com os resultados da validação cruzada
models_tibble <- adiciona_tabela("Rd.Forest", resultados_cv, models_tibble)

vip(rd_forest_model) + windows()
models_tibble
