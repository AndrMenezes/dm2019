rm(list = ls())

# See work1 to get the data

# Carrega pacotes que serão utilizados ------------------------------------
bibs <- c("MASS", "tidyverse", "rsample", "ROCR", "xtable", "knitr", "randomForest", "adabag" ,"gbm")
sapply(bibs, require, character.only = T)

# Função para arredondamento ----------------------------------------------
FF <- function(x,Digits=4,Width=4){(formatC(x,digits=Digits,width=Width,format="f"))}

# Leitura dos dados -------------------------------------------------------
colunas <- cols(age = col_double(), sex = col_factor(levels = NULL),
                cp = col_factor(levels = NULL), trestbps = col_double(),
                chol = col_double(), fbs = col_factor(levels = NULL),
                restecg = col_factor(levels = NULL), thalach = col_double(),
                exang = col_factor(levels = NULL), oldpeak = col_double(),
                slope = col_factor(levels = NULL), ca = col_integer(),
                thal = col_factor(levels = NULL), target = col_factor(levels = NULL))
heart <- read_csv(file = paste0(dname,'/dados/heart-disease-uci.zip'), col_types = colunas)

## Ao colocar a categoria 0 como referencia, estamos modelando a probabilidade do individuo ter a doença!!!!
heart$target <- relevel(heart$target, ref = "0")

# Padronizando variáveis contínuas ----------------------------------------
heart <- heart %>%
  mutate_if(is_double, function(x) (x - mean(x)) / sd(x))


# Criando variável inteira para usar no boosting --------------------------
heart$target_num <- ifelse(heart$target == "0", 0, 1)

# Separando dados em treino e teste  --------------------------------------
set.seed(666)
heart_split <- initial_split(heart, prop = 3/4, strata = "target")
heart_train <- training(heart_split)
heart_test  <- testing(heart_split)

# Separando

# número de preditores
p <- ncol(heart) - 2

# Definindo o número de árvores (B) ---------------------------------------
cv_stats <- function(split, B)
{
  # Separação dos dados treino
  df_analysis <- analysis(split)
  df_valid    <- assessment(split)
  # Ajuste dos modelos
  m_bag <- randomForest(target ~ ., data = select(df_analysis, -target_num), ntree = B,mtry = p,
                        xtest = select(df_valid, -target_num, -target), ytest = df_valid$target)
  m_rf <- randomForest(target ~ ., data = select(df_analysis, -target_num), ntree = B, mtry = sqrt(p),
                       xtest = select(df_valid, -target_num, -target), ytest = df_valid$target)
  m_boo <- gbm(formula = target_num ~ ., data = select(df_analysis, -target), distribution = "adaboost",
               n.trees = B, shrinkage = 0.01, keep.data = F)
  # Erro de predição na validação e treinamento
  prob_tra <- predict(m_boo, newdata = df_analysis, n.trees = B, type = "response")
  prob_val <- predict(m_boo, newdata = df_valid, n.trees = B, type = "response")
  pred_val <- ifelse(prob_val >= 0.5, "1", "0")
  pred_tra <- ifelse(prob_tra >= 0.5, "1", "0")

  err_boo <- c(mean(pred_val != df_valid$target, na.rm = T),
               mean(pred_tra != df_analysis$target, na.rm = T))
  err_bag <- c(mean(m_bag$test$predicted != df_valid$target, na.rm = T),
               mean(m_bag$predicted != df_analysis$target, na.rm = T))
  err_rf  <- c(mean(m_rf$test$predicted != df_valid$target, na.rm = T),
               mean(m_rf$predicted != df_analysis$target, na.rm = T))

  m_err <- matrix(c(err_bag, err_rf, err_boo), ncol = 2, byrow = T)
  colnames(m_err) <- c("valid", "train")
  m_err <- as_tibble(m_err) %>%
    mutate(model = c("bagging", "random_forest", "boosting"))
  return(m_err)
}
get_ntrees <- function(B, l_splits)
{
  tb <- map_df(l_splits, cv_stats, B = B) %>%
    group_by(model) %>%
    summarise_all(mean) %>%
    mutate(ntrees = B)
  return(tb)
}
B <- 1:300
set.seed(666)
cv_samples <- vfold_cv(data = heart_train, v = 5, strata = target)
tb_trees <- map_df(B, get_ntrees, l_splits = cv_samples$splits)
write_rds(tb_trees, path = paste0(dname,'/dados/tb_trees.rds'))


df <- tb_trees %>%
  gather(key = "base", value = err, -c(model, ntrees)) %>%
  mutate(base = factor(base, labels = c("Out-of-bag", "Validação")),
         model = factor(fct_relevel(model, "bagging", "random_forest", "boosting"),
                        labels = c("Bagging", "Random Forest", "Boosting")))
ggplot(df, aes(x=ntrees, y = err, col = base)) +
  facet_wrap(~model, scales = "free") +
  geom_line() +
  scale_x_continuous(breaks = c(1,seq(50, 300, by = 50))) +
  labs(y = expression(bar(err)), x = "N° de árvores", col = "") +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  theme(text = element_text(family = "Palatino", size = 16),
        panel.grid.major = element_line(size = 0.6),
        panel.grid.minor = element_blank(),
        legend.position = "top")


ggplot(tb_B, aes(x = B, y = erro, col = model)) +
  geom_line() +
  geom_vline(xintercept = 200, col = "black") +
  scale_x_continuous(breaks = seq(0, 300, by = 50)) +
  labs(y = expression(bar(err)), x = "N° de árvores", col = "") +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  theme(text = element_text(family = "Palatino", size = 16),
        panel.grid.major = element_line(size = 0.6),
        panel.grid.minor = element_blank(),
        legend.position = "top")

# Avaliação dos classificadores na base teste -----------------------------
# Funções úteis -----------------------------------------------------------
compute_stats <- function(prob, obs, pc = 0.5, df = FALSE)
{
  # Classificando observações
  pred <- factor(ifelse(prob >= pc, "1", "0"), levels = levels(obs))
  # Matriz de confusão (confusion matrix)
  cfm <- table(obs, pred)
  # Calculando da sensibilidade, especificidade e erro de predição
  sens  <- cfm[1] / (cfm[1] + cfm[3])
  espec <- cfm[4] / (cfm[2] + cfm[4])
  err   <- mean(pred != obs)
  # Calculando AUC usando biblioteca ROCR
  obj_rocr <- prediction(predictions = prob, labels = obs)
  auc      <- performance(obj_rocr, measure = "auc")@y.values[[1]]
  # Calculando melhor ponto de corte (pc), i.e., que maximiza sensibilidade e especificidade
  roc  <- performance(obj_rocr, measure = "tpr", x.measure = "fpr")
  esp  <- 1 - roc@x.values[[1]]
  sen  <- roc@y.values[[1]]
  ind  <- which.max(esp + sen)
  best_pc <- obj_rocr@cutoffs[[1]][ind]
  # Organizando saida
  m_out <- matrix(c(err, sens, espec, auc, best_pc), ncol = 5,
                  dimnames = list("", c("err", "sens", "espec", "auc", "pc")))
  if(df)
  {
    df <- data.frame(espec = esp, sensi = sen)
    lt <- list(medidas = m_out, cfm = cfm, df = df)
    return(lt)
  }
  else return(m_out)
}

## Ajuste dos classificadores
fit_bag <- randomForest(target ~ ., data = heart_train[, -15], ntree = 200,
                        mtry = p, importance = T)
fit_rf  <- randomForest(target ~ ., data = heart_train[, -15], ntree = 200,
                        mtry = sqrt(p), importance = T)
fit_boo <- gbm(formula = target_num ~ ., data = heart_train[, -14], distribution = "bernoulli",
               n.trees = 200, interaction.depth = 1, shrinkage = 0.1)

## Probabilidade predita
prob_bag <- predict(fit_bag, newdata = heart_test, type = "prob")[,2]
prob_rf <- predict(fit_rf, newdata = heart_test, type = "prob")[, 2]
prob_boo <- predict(fit_boo, newdata = heart_test, type = "response", n.trees = 200)

## Computando as estatísticas de performance
lt_bag <- compute_stats(prob = prob_bag, obs = heart_test$target, df = T)
lt_rf  <- compute_stats(prob = prob_rf, obs = heart_test$target, df = T)
lt_boo <- compute_stats(prob = prob_boo, obs = heart_test$target, df = T)

# Curva ROC na amostra teste ----------------------------------------------
df <- lt_bag$df %>%
  mutate(model = "Bagging") %>%
  bind_rows(mutate(lt_rf$df, model = "Random Forest")) %>%
  bind_rows(mutate(lt_boo$df, model = "Boosting")) %>%
  mutate(model = factor(fct_relevel(model, "Bagging", "Random Forest", "Boosting")))
ggplot(df, aes(x = 1 - espec, y = sensi, colour = model)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", col = "lightgray") +
  scale_y_continuous(limits = c(0, 1)) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = "1 - Especificidade", y = "Sensibilidade", col = "") +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  theme(text = element_text(family = "Palatino", size = 16),
        legend.position = "top",
        panel.grid.minor = element_blank())


