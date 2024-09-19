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

set.seed(907)

dados <- read.csv("Aprendizagem Estatistica de Maquina/Analise de Dados/Analise01/sao-paulo-properties-april-2019.csv", sep = ",") # nolint
head(dados)

# A) ---------------------------------------------------------------------------
(freq_abs <- table(dados$Negotiation.Type))
(freq_rlt <- freq_abs / length(dados$Negotiation.Type))


# B) ---------------------------------------------------------------------------

dados %>%
  ggplot(aes(x = Condo, y = Price, color = Negotiation.Type)) +
  windows() +
  geom_point()

# Fica dificil notar com essa visualização a diferença entre os gráficos pois estamos # nolint
# comparando valores de aluguel com valores de venda de imóveis, assim sendo os valores # nolint
# de aluguel ficam sem nos informar nada.


# C) ---------------------------------------------------------------------------
dados |>
  ggplot(aes(x = Condo, y = Price, color = Negotiation.Type)) +
  windows() +
  geom_point() +
  scale_color_manual(values = c("rent" = "red", "sale" = "blue")) +
  facet_wrap(~ Negotiation.Type, scales = "free") +
  theme_minimal() +
  labs(title = "Preço vs. Condomínio por Tipo de Negociação",
       x = "Condomínio",
       y = "Preço")

# Exibindo o gráfico
plot


# D) ---------------------------------------------------------------------------

# Criando tabela e encontrando os distritos
distritos <- dados |>
  count(District, sort = TRUE) |>
  head(10)
distritos

# Adicionando os Distritos na tabela
(top_districts$Distrito <- distritos)

# Criar o gráfico de barras com as frequências ordenadas de forma decrescente
ggplot(distritos, aes(x = reorder(District, -n), y = n)) +
  windows() +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Top 10 Distritos por Frequência",
       x = "Distrito",
       y = "Frequência") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), # Ajusta o ângulo e posicionamento dos rótulos #nolint
    plot.margin = margin(t = 10, r = 10, b = 50, l = 10) # Aumenta a margem inferior para evitar cortes nos textos #nolint
  )

# E) ---------------------------------------------------------------------------

# Filtrando o modelo
dados_modelo <- dados %>%
  filter(Negotiation.Type ==  "rent") %>%
    select(- Negotiation.Type)
glimpse(dados_modelo)
colnames(dados_modelo)

# Criando uma tabela para comparar os modelos
models_tibble <- tibble(modelo = c("lm", "ridge", "LASSO", 
                                   "Arv.Decisão", "Rd.Forest"),
                        RQM = NA * length(modelo), RMSE = NA * length(modelo))

models_tibble
# i) regressão linear ----------------------------------------------------------

# Convertendo variáveis categóricas para fatores
dados_modelo <- dados_modelo %>%
  mutate(across(where(is.character), as.factor))

# Verificando variáveis com apenas um nível e removendo-as
dados_modelo <- dados_modelo %>%
  select(where(~ n_distinct(.) > 1))

glimpse(dados_modelo)
# Separando o database em treino e teste
split <- initial_split(dados_modelo, prop = 0.8, strata = Price)

treinamento <- training(split)
teste <- testing(split)

# Ajustando o modelo de regressão linear
fit <- lm(Price ~ ., data = treinamento)
summary(fit)

# Avaliando o modelo

y_pred <- predict(fit, newdata = teste)
y_real  <- teste$Price

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
split <- initial_split(dados_modelo, prop = 0.8, strata = Price)
treinamento <- training(split)
teste <- testing(split)

# Convertendo as variáveis preditoras e resposta para matriz
x_train <- model.matrix(Price ~ .,
                        data = treinamento)[, -1] # Exclui a coluna de intercepto #nolint
y_train <- treinamento$Price

x_test <- model.matrix(Price ~ ., data = teste)[, -1]
y_test <- teste$Price

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
tree <- rpart(Price ~ ., data = treinamento, method = "anova")
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
  dependent.variable.name = "Price",
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

# obs: A definição das preditoras utilizadas no modelo é de livre escolha 

#Avaliação do erro:
# Como podemos notar, a métrica RQM indica o quão bem nosso modelo foi capaz de explicar a variabilidade dos dados
# Assim sendo o modelo que melhor explica os nossos dados é o que possuí o maior RQM. No caso o melhor modelo para prever
# a variavel Price dentre os modelos treinado é o modelo de Random Forest, que possuí o maior RQM de 0.702.
# Além do RQM também temos o RMSE que está com um valor alto para todos os modelos, isso deve-se a alta variabilidade
# do banco de dados, uma proposta para poder melhorar nossas previsões é criar uma segmentação do banco de dados, com um
# modelo preditor para cada segmentação, de forma a separar os casos, pois a depender da região e do tipo de apartamento 
# o preço pode mudar abruptamente. Assim poderiamos gerar um modelo de decisão para decidir um modelo preditor para predizer
# o preço de forma mais precisa.