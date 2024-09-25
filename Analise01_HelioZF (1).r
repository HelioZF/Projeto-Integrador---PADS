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

# criando uma função para realizar

# Definindo a função para calcular as métricas e salvar no models_tibble
adiciona_tabela <- function(modelo_nome, resultados, models_tibble) {
  
  # Reordenando os níveis da variável 'truth'
  resultados$truth <- factor(resultados$truth, levels = c("1", "0"))
  # Reordenando os níveis da variável 'estimate' para corresponder a 'truth'
  resultados$estimate <- factor(resultados$estimate, levels = c("1", "0"))

  
  # Calculando as métricas
  accuracy_result <- accuracy(resultados, truth = truth, estimate = estimate)
  precision_result <- precision(resultados, truth = truth, estimate = estimate)
  roc_auc_result <- roc_auc(resultados, truth, .pred_1)
  
  # Atualizando a tabela models_tibble com os resultados
  models_tibble$Accuracy[models_tibble$modelo == modelo_nome] <- accuracy_result$.estimate
  models_tibble$Precision[models_tibble$modelo == modelo_nome] <- precision_result$.estimate
  models_tibble$ROC_Curve[models_tibble$modelo == modelo_nome] <- roc_auc_result$.estimate
  
  # Retornando a tabela atualizada
  return(models_tibble)
}



# i) regressão logistica ----------------------------------------------------------

# Separando o database em treino e teste
split <- initial_split(dados, prop = 0.8)

treinamento <- training(split)
teste <- testing(split)

# Ajustando o modelo de regressão logística
fit <- glm(operates_within_2_years ~ ., data = treinamento, family = "binomial")

# Fazendo previsões probabilísticas
y_pred_prob <- predict(fit, newdata = teste, type = "response")

# Convertendo as probabilidades em 0 ou 1 com base no limiar (sarrafo)
y_pred <- as.factor(ifelse(y_pred_prob > sarrafo, 1, 0))

# Convertendo os valores reais em fator
y_real <- as.factor(teste$operates_within_2_years)

# Criando um tibble com os valores reais, previstos e as probabilidades
resultados <- tibble(
  truth = y_real,       # Valores reais
  estimate = y_pred,    # Valores previstos
  .pred_1 = y_pred_prob # Probabilidades previstas para a classe 1
)

models_tibble <- adiciona_tabela("glm", resultados, models_tibble)

# ii) regressão ridge ----------------------------------------------------------

# Separando os dados em treino e teste
split <- initial_split(dados, prop = 0.8)
treinamento <- training(split)
teste <- testing(split)

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

# Definindo o threshold (ajuste conforme necessário)
sarrafo <- 0.5

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

# Definindo o threshold (ajuste conforme necessário)
sarrafo <- 0.5

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

# Fazendo previsões no conjunto de teste
predictions <- predict(rd_forest_model, data = teste)

y_pred_prob <- predictions$predictions[, "1"]

y_pred <- as.factor(ifelse(y_pred_prob > sarrafo, 1, 0))

y_real <- as.factor(teste$operates_within_2_years)

resultados <- tibble(
  truth = y_real,       # Valores reais
  estimate = y_pred,    # Valores previstos
  .pred_1 = y_pred_prob # Probabilidades previstas para a classe '1'
)

models_tibble <- adiciona_tabela("Rd.Forest", resultados, models_tibble)

vip(rd_forest_model) + windows()
models_tibble
