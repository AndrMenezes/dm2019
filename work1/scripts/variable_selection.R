rm(list = ls())
FF <- function(x,Digits=4,Width=4){(formatC(x,digits=Digits,width=Width,format="f"))}

# Bibliotecas -------------------------------------------------------------
bibs <- c("tidyverse", "rsample", "MASS", "ROCR", "xtable")
sapply(bibs, require, character.only = T)

# Diretório ---------------------------------------------------------------
dname <- "/home/andrefbm/Dropbox/Unicamp/MI420 - Mineração de Dados/Trabalhos/Classificadores"
# dname <- "C:/Users/User/Dropbox/Unicamp/MI420 - Mineração de Dados/Trabalhos/Classificadores"

# Leitura dos dados -------------------------------------------------------
colunas <- cols(age = col_double(), sex = col_factor(levels = NULL),
                cp = col_factor(levels = NULL), trestbps = col_double(),
                chol = col_double(), fbs = col_factor(levels = NULL),
                restecg = col_factor(levels = NULL), thalach = col_double(),
                exang = col_factor(levels = NULL), oldpeak = col_double(),
                slope = col_factor(levels = NULL), ca = col_integer(),
                thal = col_factor(levels = NULL), target = col_factor(levels = NULL))
heart <- read_csv(file = paste0(dname,'/dados/heart-disease-uci.zip'), col_types = colunas)
glimpse(heart)

## Ao colocar a categoria 0 como referencia, estamos modelando a probabilidade do individuo ter a doença!!!!
heart$target <- relevel(heart$target, ref = "0") 

## Padronizando variáveis contínuas
heart <- heart %>% 
  mutate_if(is_double, function(x) (x - mean(x)) / sd(x))

# Análise descritiva das variáveis ----------------------------------------
## Variáveis Continuas
tb_cont <- heart %>% 
  select(target) %>% 
  bind_cols(select_if(heart, is_double)) %>% 
  gather(key = "preditora", value = "valor", -target)

x11()
ggplot(tb_cont, aes(x = valor, y = ..density.., fill = target, group = target)) +
  facet_wrap(~preditora, scales = "free", ncol = 5) +
  geom_histogram(bins = 30, col = "black") +
  geom_density(alpha = 0.8) +
  labs(x = "") +
  theme_bw()

## Variáveis Categóricas
tb_cat <- heart %>% 
  select_if(function(x) is.factor(x) | is.integer(x)) %>% 
  gather(key = "preditora", value = "valor", -target)
tb_cat %>% filter(preditora == "sex") %>% 
  ggplot(aes(x = target, fill = valor)) +
  geom_bar() 
x11();ggplot(heart, aes(x = target, fill = factor(sex, ordered = T))) +
  geom_bar()


# Separando dados em treino e teste ---------------------------------------
set.seed(666)
heart_split <- initial_split(heart, prop = 3/4, strata = "target")
heart_train <- training(heart_split) 
heart_test  <- testing(heart_split) 

# Regressão Logística: Seleção de Variáveis -------------------------------

## Criando todas as possíveis estruturas de preditores
yname  <- "target"
xs     <- names(heart)[-match(yname, names(heart))]
p      <- length(xs)
k      <- 2^p - 1 
id     <- unlist(lapply(1:p, function(j) combn(1:p, j, simplify = FALSE)), recursive = FALSE)
models <- lapply(id, function(j) paste0(yname, " ~ ", paste0(xs[j], collapse = " + ")))
length(models)
tb_mod <- as_tibble(do.call('rbind', models)) %>% 
  rename(modelo = V1) %>% 
  mutate(nvars = 1 + stringi::stri_count(modelo, fixed = "+")) %>% 
  filter(nvars > 4) # Seleciona apenas modelos com 5 ou mais preditores
tb_mod

## Função para ajustar regressão logísticas e retornar medidas
stats_logistic <- function(splits, formula)
{
  # Ajustando o modelo na base  de treino
  fit <- glm(formula = as.formula(formula), data = analysis(splits), family = binomial(link = "logit"))
  # Prediçõo na base de treino e validação
  valid <- assessment(splits)
  lvsl  <- levels(valid$target)
  prob  <- predict(fit, newdata = valid, type = "response")
  pred  <- factor(ifelse(prob >= 0.5, "1", "0"), levels = lvsl) 
  # Calculando AUC
  auc <- performance(prediction(prob, valid$target), measure = "auc")@y.values[[1]]
  # Calculando especificidade, sensibilidade e erro de predição
  cfm   <- table(valid$target, pred)
  sens  <- cfm[1] / (cfm[1] + cfm[3])
  espec <- cfm[4] / (cfm[2] + cfm[4])
  err_valid <- mean(pred != valid$target)
  err_train <- mean(factor(ifelse(fit$fitted.values >= 0.5, "1", "0"), levels = lvsl) != fit$y)
  # Organiza medidas
  ma  <- matrix(c(AIC(fit), BIC(fit), err_valid, err_train, sens, espec, auc), ncol = 7, 
                dimnames = list("", c("AIC", "BIC", "err_valid", "err_train", "sens", "espec", "auc"))) 
  return(ma)
}

## Função para aplicar CV no modelo especificado por "formula"
cv_logistic <- function(formula, cv)
{
  lt <- map(cv$splits, stats_logistic, formula)
  mr <- do.call(rbind, lt)
  tb <- t(colMeans(mr)) %>% 
    as_tibble() %>% 
    mutate(predictors = stringr::str_remove(formula, "target ~ "), 
           nvars = 1 + stringi::stri_count(formula, fixed = "+"))
  return(tb)  
}

## Aplicando CV em todos os modelos (demora em torno de 664.254 seg)
set.seed(666)
cv1  <- vfold_cv(heart_train, v = 10, strata = 'target')
out1 <- do.call(rbind, map(tb_mod$modelo, cv_logistic, cv1))
saveRDS(object = out1, file = paste0(dname, "/cv_logistic.rds"))

## Os 3 melhores modelos de regressão logistica conforme a menor taxa de erro  
top_3_rl <- out1 %>% 
  rename(predictors = modelo) %>% 
  top_n(3, -err_valid) %>% 
  arrange(err_valid, -auc) %>% 
  mutate_if(is_double, function(x) FF(x, 4)) %>% 
  mutate(predictors = paste0("$\\texttt{", predictors, "}$")) 


# Discriminante Linear e Quadrático: Seleção de Variáveis -----------------

## Seleciona apenas as variáveis contínuas
heart_cont <- heart_train %>% 
  dplyr::select(target) %>% 
  bind_cols(select_if(heart_train, is_double))

## Criando todas combinações de modelos
yname  <- "target"
xs     <- names(heart_cont)[-match(yname, names(heart_cont))]
p      <- length(xs)
k      <- 2^p - 1 
id     <- unlist(lapply(1:p, function(j) combn(1:p, j, simplify = FALSE)), recursive = FALSE)
models <- sapply(id, function(j) paste0(yname, " ~ ", paste0(xs[j], collapse = " + ")))
length(models)

## Função para ajustar modelo LDA e QDA e retornar medidas
stats_discrim <- function(splits, formula)
{
  # Guardando dados de treino e validação 
  train <- analysis(splits)
  valid  <- assessment(splits)
  # Ajustando modelos
  fit_lda  <- lda(formula = as.formula(formula), data = train, method = "mle")
  fit_qda  <- qda(formula = as.formula(formula), data = train, method = "mle")
  # Realizando predições
  pred_lda <- predict(fit_lda, newdata = valid)
  pred_qda <- predict(fit_qda, newdata = valid)
  # Calculando AUC
  auc_lda <- performance(prediction(pred_lda$posterior[, 2], valid$target), measure = "auc")@y.values[[1]]
  auc_qda <- performance(prediction(pred_qda$posterior[, 2], valid$target), measure = "auc")@y.values[[1]]
  # Calculando especificidade, sensibilidade e erro de predição
  cfm_lda  <- table(valid$target, pred_lda$class)
  cfm_qda  <- table(valid$target, pred_qda$class)
  sens_lda <- cfm_lda[1] / (cfm_lda[1] + cfm_lda[3])
  espe_lda <- cfm_lda[4] / (cfm_lda[2] + cfm_lda[4])
  sens_qda <- cfm_qda[1] / (cfm_qda[1] + cfm_qda[3])
  espe_qda <- cfm_qda[4] / (cfm_qda[2] + cfm_qda[4])
  err_lda  <- mean(valid$target != pred_lda$class) 
  err_qda  <- mean(valid$target != pred_qda$class) 
  # Calculando erro de predição na base de treino
  err_lda_tr <- mean(train$target != predict(fit_lda)$class) 
  err_qda_tr <- mean(train$target != predict(fit_qda)$class) 
  # Guardando informações
  mr <- matrix(c(err_lda, err_qda,
                 err_lda_tr, err_qda_tr,
                 sens_lda, sens_qda,
                 espe_lda, espe_qda,
                 auc_lda, auc_qda), nrow = 2, ncol = 5)
  colnames(mr) <- c("err_valid", "err_train", "sens", "espec", "auc")
  tb <- as_tibble(mr) %>% 
    mutate(model = c("lda", "qda"))
  return(tb)
}

## Função para aplicar CV no modelo especificado no argumento "formula"
cv_discrim <- function(formula, cv)
{
  lt <- map(cv$splits, stats_discrim, formula)
  mr <- do.call(rbind, lt)
  tb <- mr %>% 
    group_by(model) %>% 
    summarise_all(mean) %>% 
    mutate(predictors = stringr::str_remove(formula, "target ~ "), 
           nvars = as.integer(1 + stringi::stri_count(formula, fixed = "+")))
  return(tb)  
}

## Aplicando CV em todos os modelos para os classificadores LDA e QDA (demora em torno de 4.836 seg)
set.seed(666)
cv2  <- vfold_cv(data = heart_cont, v = 10, strata = 'target')
out2 <- do.call(rbind, map(models, cv_discrim, cv2))

## Os 3 melhores modelos de discriminante linear e quadrático conforme a menor taxa de erro  
top_3_discrm <- out2 %>% 
  group_by(model) %>% 
  top_n(n = 3, wt = -err_valid) %>% 
  ungroup() %>% 
  arrange(model, err_valid, -auc) %>% 
  mutate_if(is_double, function(x) FF(x, 4)) %>% 
  mutate(predictors = paste0("$\\texttt{", predictors, "}$"))

## Tabela com resultados da seleção de preditores para os três classificadores
top_3 <- top_3_rl %>% 
  mutate(model = "logistic", nvars = as.integer(nvars)) %>% 
  dplyr::select(model, everything(), -AIC, -BIC) %>%
  bind_rows(top_3_discrm)
top_3$model <- ""
top_3[1,1] <- "\\multirow{3}{*}{Logística}"
top_3[4,1] <- "\\multirow{3}{*}{LDA}"
top_3[7,1] <- "\\multirow{3}{*}{QDA}"

print.xtable(xtable(top_3), include.rownames = F, include.colnames = T, 
             sanitize.text.function = force)
##########################################################################################
sessionInfo()
