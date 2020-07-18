## Carrega pacotes que serão utilizados
bibs <- c("tidyverse", "rsample", "ROCR", "xtable", "knitr", "e1071")
sapply(bibs, require, character.only = T)

## Função para arredondamento
FF <- function(x,Digits=4,Width=4){(formatC(x,digits=Digits,width=Width,format="f"))}

## Função para calcular estatísticas de performance
compute_stats <- function(prob, obs, pc = 0.5, df = FALSE)
{
  # Classificando observações
  pred <- factor(ifelse(prob >= pc, "Ts65Dn", "Control"), levels = levels(obs))
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


## ---- Leitura_dados
## See report3 to get the data
dados <- read_csv("Data_Cortex_Nuclear.csv",
                  col_types = cols(Genotype = col_factor(levels = NULL))) %>%
  select(MouseID:Genotype)
names(dados)   <- gsub("_N", "", names(dados))

## Retirando variáveis com bastante valores faltantes
nas   <- sort(apply(dados, 2, function(z) sum(is.na(z))), decreasing = T)
dados <- dados %>%
  select(-c(names(nas)[1:8])) %>%
  na.omit()

## Ao colocar a categoria Control como referencia, estamos modelando a probabilidade do rato ter genótipo Ts65Dn
dados$Genotype <- relevel(dados$Genotype, ref = "Control")

## Padronizando variáveis contínuas
dados <- dados %>%
   mutate_if(is_double, function(x) (x - mean(x)) / sd(x))

## Separando dados em treino e teste
set.seed(666)
dados_split <- initial_split(dados, prop = 3/4, strata = "Genotype")
dados_train <- training(dados_split)
dados_test  <- testing(dados_split)

## Modelo selecionado para a regressão logística
ff = Genotype ~ pAKT + pELK + pERK + pNR1 + pNR2A + pPKCAB + BRAF +
    CREB + ERK + TRKA + APP + SOD1 + MTOR + DSCR1 + NR2B + pNUMB +
    TIAM1 + NUMB + BAX + ERBB4 + GluR3 + P3525 + SHH + SYP

## Ajuste dos modelos
m_log <- glm(ff, data = dados_train, family = binomial(link = "logit"))
m_svm_linear <- svm(Genotype ~ ., data = select(dados_train, -MouseID),
                    kernel = "linear",
                    cost = 0.5,
                    prob = T)
m_svm_poly <- svm(Genotype ~ ., data = select(dados_train, -MouseID), prob = T,
                    kernel = "polynomial",
                    cost = 0.5,
                    gamma = 0.01,
                    coef0 = 1,
                    degree = 4)
m_svm_radial <- svm(Genotype ~ ., data = select(dados_train, -MouseID), prob = T,
                    kernel = "radial",
                    cost = 10,
                    gamma = 0.01)
m_svm_sigmoid <- svm(Genotype ~ ., data = select(dados_train, -MouseID), prob = T,
                    kernel = "sigmoid",
                    cost = 10,
                    gamma = 0.01,
                    coef0 = -1)

## Probabilidades preditas
p_log         = predict(m_log, newdata = dados_test, type = "response")
p_svm_linear  = attr(predict(m_svm_linear, newdata = dados_test, probability = T), "probabilities")[, 2]
p_svm_poly    = attr(predict(m_svm_poly, newdata = dados_test, probability = T), "probabilities")[, 2]
p_svm_radial  = attr(predict(m_svm_radial, newdata = dados_test, probability = T), "probabilities")[, 2]
p_svm_sigmoid = attr(predict(m_svm_sigmoid, newdata = dados_test, probability = T), "probabilities")[, 2]

## Calculando estatísticas de performance
lt_log         = compute_stats(prob = p_log, obs = dados_test$Genotype, df = T)
lt_svm_linear  = compute_stats(prob = p_svm_linear, obs = dados_test$Genotype, df = T)
lt_svm_poly    = compute_stats(prob = p_svm_poly, obs = dados_test$Genotype, df = T)
lt_svm_radial  = compute_stats(prob = p_svm_radial, obs = dados_test$Genotype, df = T)
lt_svm_sigmoid = compute_stats(prob = p_svm_sigmoid, obs = dados_test$Genotype, df = T)

## ---- Tabela
tab <- rbind(lt_log$medidas[,-5], lt_svm_linear$medidas[,-5], lt_svm_poly$medidas[,-5],
             lt_svm_radial$medidas[, -5], lt_svm_sigmoid$medidas[, -5]) %>%
  as_tibble() %>%
  mutate_all(FF) %>%
  mutate(model = c("Logística", "Linear", "Polinomial", "Radial", "Sigmoide")) %>%
  dplyr::select(model, err, auc, everything())
names(tab) <- c("Modelo", "$\\mathrm{Err}_{\\mathcal{T}}$", "AUC", "Sens.", "Espec.")
print.xtable(xtable(tab, align = c("l", "l", rep("c", 4)), label = "tab:medidas",
                    caption = "Medidas de performance dos classificadores na amostra teste."),
             table.placement = "H",
             booktabs = T,
             include.rownames = F,
             include.colnames = T,
             sanitize.text.function = force)

## ---- Curva ROC
df <- lt_log$df %>%
  mutate(model = "Logística") %>%
  bind_rows(mutate(lt_svm_linear$df, model = "Linear")) %>%
  bind_rows(mutate(lt_svm_poly$df, model = "Polinomial")) %>%
  bind_rows(mutate(lt_svm_radial$df, model = "Radial")) %>%
  bind_rows(mutate(lt_svm_sigmoid$df, model = "Sigmoide")) %>%
  mutate(model = factor(fct_relevel(model, "Logística", "Linear", "Polinomial", "Radial", "Sigmoide")))
ggplot(df, aes(x = 1 - espec, y = sensi, colour = model)) +
  geom_line(size = 0.4) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", col = "lightgray") +
  scale_y_continuous(limits = c(0, 1)) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(x = "1 - Especificidade", y = "Sensibilidade", col = "") +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  theme(text = element_text(family = "Palatino", size = 8),
        legend.position = "top",
        panel.grid.minor = element_blank())

