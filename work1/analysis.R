rm(list = ls())

## See the report.pdf to get the data

# Carrega pacotes que serão utilizados ------------------------------------
bibs <- c("MASS", "tidyverse", "rsample", "ROCR", "xtable", "knitr")
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
heart <- read_csv(file = '/dados/heart-disease-uci.zip', col_types = colunas)

## Ao colocar a categoria 0 como referencia, estamos modelando a probabilidade do individuo ter a doença!!!!
heart$target <- relevel(heart$target, ref = "0")

# Padronizando variáveis contínuas ----------------------------------------
heart <- heart %>%
  mutate_if(is_double, function(x) (x - mean(x)) / sd(x))


# Separando dados em treino e teste  --------------------------------------
set.seed(666)
heart_split <- initial_split(heart, prop = 3/4, strata = "target")
heart_train <- training(heart_split)
heart_test  <- testing(heart_split)


# Melhores modelos para cada classificador --------------------------------
mod_log <- target ~ sex + cp + trestbps + fbs + restecg + thalach + oldpeak + ca + thal
mod_lda <- target ~ thalach + oldpeak
mod_qda <- target ~ trestbps + thalach + oldpeak


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

bt_stats <- function(splits)
{
  # Guardando dados de treino e validação
  train <- analysis(splits)
  valid <- assessment(splits)
  lvsl  <- levels(valid$target)
  # Ajustando modelos
  fit_log <- glm(formula = mod_log, data = train, family = binomial(link = "logit"))
  fit_lda <- lda(formula = mod_lda, data = train, method = "mle")
  fit_qda <- qda(formula = mod_qda, data = train, method = "mle")
  # Calculando probabilidades
  prob_log <- predict(fit_log, newdata = valid, type = "response")
  prob_lda <- predict(fit_lda, newdata = valid)[[2]][,2]
  prob_qda <- predict(fit_qda, newdata = valid)[[2]][,2]
  # Computando as estatísticas de performance
  tb <- rbind(compute_stats(prob = prob_log, obs = valid$target),
              compute_stats(prob = prob_lda, obs = valid$target),
              compute_stats(prob = prob_qda, obs = valid$target))
  tb <- tb %>% as_tibble() %>% mutate(model = c("logistic", "lda", "qda"))
}

# Performance dos classificadores na amostra treino -----------------------
set.seed(666)
bt_resamples <- bootstraps(data = heart_train, times = 2000, strata = "target")
bt_out <- map_df(bt_resamples$splits, bt_stats)
bt_out <- bt_out %>% gather(key = "medida", value = "boot", -model)
write_rds(bt_out, path = paste0(dname, "/dados/bt_out.rds"))



# Distribuição Bootstrap das medidas --------------------------------------
bt_out <- read_rds(paste0(dname, "/dados/bt_out.rds"))
bt_out %>%
  filter(medida != "pc") %>%
  mutate(model = factor(fct_relevel(model, "logistic", "lda", "qda"), labels = c("Logistica", "LDA", "QDA")),
         medida = factor(fct_relevel(medida, "err", "auc", "sens", "espec"),
                         labels = c(expression(widehat(Err)), "AUC", "Sensibilidade", "Especificidade"))) %>%
  ggplot(aes(x=model, y = boot, fill = model)) +
  facet_wrap(~medida, scales = "free", labeller = label_parsed) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set1") +
  labs(x = "", y = "", fill = "") +
  theme_bw() +
  theme(text         = element_text(family = "Palatino", size = 16),
        axis.title.x = element_blank(),
        axis.text.x  = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "top")


# Desempenho na base teste ------------------------------------------------
## Ajuste dos modelos
fit_log <- glm(formula = mod_log, data = heart_train, family = binomial(link = "logit"))
fit_lda <- lda(formula = mod_lda, data = heart_train, method = "mle")
fit_qda <- qda(formula = mod_qda, data = heart_train, method = "mle")
# Calculando probabilidades
prob_log <- predict(fit_log, newdata = heart_test, type = "response")
prob_lda <- predict(fit_lda, newdata = heart_test)[[2]][,2]
prob_qda <- predict(fit_qda, newdata = heart_test)[[2]][,2]
# Selecionando melhor ponto de corte obtido no treino
pcs <- pc_boot %>% filter(medida == "pc") %>% dplyr::select(mediana, media, model)
# Computando as estatísticas de performance
lt_log <- compute_stats(prob = prob_log, obs = heart_test$target, pc = pcs$mediana[2], df = T)
lt_lda <- compute_stats(prob = prob_lda, obs = heart_test$target, pc = pcs$mediana[1], df = T)
lt_qda <- compute_stats(prob = prob_qda, obs = heart_test$target, pc = pcs$mediana[3], df = T)

# Curva ROC na amostra teste ----------------------------------------------
df <- lt_log$df %>%
  mutate(model = "Logística") %>%
  bind_rows(mutate(lt_lda$df, model = "LDA")) %>%
  bind_rows(mutate(lt_qda$df, model = "QDA")) %>%
  mutate(model = factor(fct_relevel(model, "Logística", "LDA", "QDA")))
ggplot(df, aes(x = 1 - espec, y = sensi, colour = model)) +
  geom_line(size = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", col = "lightgray") +
  scale_y_continuous(limits = c(0, 1)) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = "1 - Especificidade", y = "Sensibilidade", col = "") +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  theme(text = element_text(family = "Palatino", size = 8),
        legend.position = "top",
        panel.grid.minor = element_blank())

##########################################################################################
sessionInfo()

