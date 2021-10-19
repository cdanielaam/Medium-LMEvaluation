library(ggplot2)
library(hrbrthemes)
library(tidyverse)
library(blandr)
library(readr)


#import and read data:
df <- read_csv("df.csv")

#view all data:
View(df)
#view first 10 rows:
head(df, 10)

#Scatter-plot all predictors:
ggplot(df, aes(BMXHIP, y = value, color = variable)) + 
  geom_point(aes(y = BMXWT, col = "Weight")) + 
  geom_point(aes(y = BMXHT, col = "Height")) +
  geom_point(aes(y = BMXWAIST, col = "Waist Circumference"))


#Scatter-plot for weight:
ggplot(df, aes(x=BMXWT, y=BMXHIP)) +
  geom_point() +
  geom_smooth(method=lm , color="red", se=FALSE) +
  theme_ipsum()

#Scatter-plot for height:
ggplot(df, aes(x=BMXHT, y=BMXHIP)) +
  geom_point() +
  geom_smooth(method=lm , color="red", se=FALSE) +
  theme_ipsum()

#Scatter-plot for waist circumference:
ggplot(df, aes(x=BMXWAIST, y=BMXHIP)) +
  geom_point() +
  geom_smooth(method=lm , color="red", se=FALSE) +
  theme_ipsum()

#Pearson correlations:
library("ggpubr")

#with height:
cor(df$BMXHIP, df$BMXHT, method = c("pearson"))

#with weight:
cor(df$BMXHIP, df$BMXWT, method = c("pearson"))

#with waist circumference:
cor(df$BMXHIP, df$BMXWAIST, method = c("pearson"))

df$mean_BMXHIP <- mean(df$BMXHIP)
df$sd_BMXHIP <- sd(df$BMXHIP)

#Linear Regression Models:
#Model 1:
lm1<-lm(df$BMXHIP ~ df$BMXHT)
summary(lm1)
AIC(lm1)
BIC(lm1)
 #Bland-Altman:
df$BMXHIP_1 <- 78.90301 + (0.16066*df$BMXHT) #variable with predicted values
df$md_1 <- df$BMXHIP-df$BMXHIP_1 #differences between measured and predicted
df$average_md_1 <- (df$BMXHIP+df$BMXHIP_1)/2 #average between measured and predicted
df$mean_BMXHIP_1 <- mean(df$BMXHIP_1) #mean of predicted values
df$sd_BMXHIP_1 <- sd(df$BMXHIP_1) #sd of predicted values
df$mean_md_1 <- mean(df$md_1) #mean of the differences between measured and predicted
df$sd_md_1 <- sd(df$md_1) #sd of the differences between measured and predicted

df$LLA_1 <- df$mean_md_1 - (1.96*df$sd_md1) #lower limit of agreement
df$ULA_1 <- df$mean_md_1 + (1.96*df$sd_md1) #upper limit of agreement


plot(x=df$BMXHIP, y=df$md_1, ylab = "Measured HipC- Predicted HipC from Model 1", xlab="Measured HipC")
abline(h=0)
abline(h=-0.0004395559, col='blue', lty = 2) #Bias 
abline(h=-29.12615, col='red', lty=2) #LLA line
abline(h=29.12527, col='red', lty=2) #ULA line
#expr <- vector("expression", 4)
#expr[[1]] <- bquote(Mean[diff]==-0.0004395559)
#expr[[2]] <- bquote(SD[diff]==14.86006)
#expr[[3]] <- bquote(LL==-29.12615)
#expr[[4]] <- bquote(UL==29.12527)
#legend("topleft", bty = "n", legend = expr)

df$LAAmplitude_1 <- 29.12615 + 29.12527 #Limit of Agreement amplitude

#Pearson correlation between obtained and predicted values:
cor(df$BMXHIP, df$BMXHIP_1, method = c("pearson"))

#Model 2:
lm2 <- lm(df$BMXHIP ~ df$BMXWT)
summary(lm2)
AIC(lm2)
BIC(lm2)
 #Bland_Altman:
df$BMXHIP_2 <- 59.709755 + (0.568801*df$BMXWT)
df$md_2 <- df$BMXHIP-df$BMXHIP_2
df$average_md_2 <- (df$BMXHIP+df$BMXHIP_2)/2 #average between measured and predicted
df$mean_BMXHIP_2 <- mean(df$BMXHIP_2)
df$sd_BMXHIP_2 <- sd(df$BMXHIP_2)
df$mean_md_2 <- mean(df$md_2)
df$sd_md_2 <- sd(df$md_2)

df$LLA_2 <- df$mean_md_2 - (1.96*df$sd_md_2) #lower limit of agreement
df$ULA_2 <- df$mean_md_2 + (1.96*df$sd_md_2) #upper limit of agreement

plot(x=df$BMXHIP, y=df$md_2)
abline(h=0)
abline(h=-0.000000068122, col='blue', lty=2) #Bias 
abline(h=-13.97443, col='red', lty=2) #LLA line
abline(h=13.97443, col='red', lty=2) #ULA line
#expr <- vector("expression", 4)
#expr[[1]] <- bquote(Mean[diff]==-0.000000068122)
#expr[[2]] <- bquote(SD[diff]==7.129811)
#expr[[3]] <- bquote(LL==-13.97443)
#legend("topleft", bty = "n", legend = expr)

df$LAAmplitude_2 <- 13.97443 + 13.97443 #Limit of Agreement amplitude

#Model 3:
lm3 <- lm(df$BMXHIP ~ df$BMXWAIST)
summary(lm3)
AIC(lm3)
BIC(lm3)
 #Bland-Altman:
df$BMXHIP_3 <- 35.59334 + (0.71649*df$BMXWAIST)
df$md_3 <- df$BMXHIP-df$BMXHIP_3
df$average_md_3 <- (df$BMXHIP+df$BMXHIP_3)/2 #average between measured and predicted
df$mean_BMXHIP_3 <- mean(df$BMXHIP_3)
df$sd_BMXHIP_3 <- sd(df$BMXHIP_3)
df$mean_md_3 <- mean(df$md_3)
df$sd_md_3 <- sd(df$md_3)

df$LLA_3 <- df$mean_md_3 - (1.96*df$sd_md_3) #lower limit of agreement
df$ULA_3 <- df$mean_md_3 + (1.96*df$sd_md_3) #upper limit of agreement

plot(x=df$BMXHIP, y=df$md_3)
abline(h=0)
abline(h=0.0003125338, col='blue', lty=2) #Bias 
abline(h=-14.11896, col='red', lty=2) #LLA line
abline(h=14.11958, col='red', lty=2) #ULA line
#expr <- vector("expression", 4)
#expr[[1]] <- bquote(Mean[diff]==0.0003125338)
#expr[[2]] <- bquote(SD[diff]==7.203708)
#expr[[3]] <- bquote(LL==-14.11896)
#expr[[4]] <- bquote(UL==14.11958)
#legend("topleft", bty = "n", legend = expr)

df$LAAmplitude_3 <- 14.11896 + 14.11958 #Limit of Agreement amplitude

#Model 4:
lm4 <- lm(df$BMXHIP ~ df$BMXHT + df$BMXWT)
summary(lm4)
AIC(lm4)
BIC(lm4)
 #Bland-Altman:
df$BMXHIP_4 <- 137.920164 + (-0.518439*df$BMXHT) + (0.669106*df$BMXWT)
df$md_4 <- df$BMXHIP-df$BMXHIP_4
df$average_md_4 <- (df$BMXHIP+df$BMXHIP_4)/2 #average between measured and predicted
df$mean_BMXHIP_4 <- mean(df$BMXHIP_4)
df$sd_BMXHIP_4 <- sd(df$BMXHIP_4)
df$mean_md_4 <- mean(df$md_4)
df$sd_md_4 <- sd(df$md_4)

df$LLA_4 <- df$mean_md_4 - (1.96*df$sd_md_4) #lower limit of agreement
df$ULA_4 <- df$mean_md_4 + (1.96*df$sd_md_4) #upper limit of agreement

plot(x=df$BMXHIP, y=df$md_4)
abline(h=0)
abline(h=0.001440743, col='blue', lty=2) #Bias 
abline(h=-10.52911, col='red', lty=2) #LLA line
abline(h=10.53199, col='red', lty=2) #ULA line
#expr <- vector("expression", 4)
#expr[[1]] <- bquote(Mean[diff]==0.001440743)
#expr[[2]] <- bquote(SD[diff]==5.372729)
#expr[[3]] <- bquote(LL==-10.52911)
#expr[[4]] <- bquote(UL==10.53199)
#legend("topleft", bty = "n", legend = expr)

df$LAAmplitude_4 <- 10.52911 + 10.53199 #Limit of Agreement amplitude

#Model 5:
lm5 <- lm(df$BMXHIP ~ df$BMXHT + df$BMXWAIST)
summary(lm5)
AIC(lm5)
BIC(lm5)
 #Bland-Altman:
df$BMXHIP_5 <- 50.561862 + (-0.095965*df$BMXHT) + (0.726806*df$BMXWAIST)
df$average_md_5 <- (df$BMXHIP+df$BMXHIP_5)/2 #average between measured and predicted
df$md_5 <- df$BMXHIP-df$BMXHIP_5
df$mean_BMXHIP_5 <- mean(df$BMXHIP_5)
df$sd_BMXHIP_5 <- sd(df$BMXHIP_5)
df$mean_md_5 <- mean(df$md_5)
df$sd_md_5 <- sd(df$md_5)

df$LLA_5 <- df$mean_md_5 - (1.96*df$sd_md_5) #lower limit of agreement
df$ULA_5 <- df$mean_md_5 + (1.96*df$sd_md_5) #upper limit of agreement

plot(x=df$BMXHIP, y=df$md_5)
abline(h=0)
abline(h=0.00001758386, col='blue', lty=2) #Bias 
abline(h=-13.99613, col='red', lty=2) #LLA line
abline(h=13.99616, col='red', lty=2) #ULA line
#expr <- vector("expression", 4)
#expr[[1]] <- bquote(Mean[diff]==0.00001758386)
#expr[[2]] <- bquote(SD[diff]==7.14089)
#expr[[3]] <- bquote(LL==-13.99613)
#legend("topleft", bty = "n", legend = expr)

df$LAAmplitude_5 <- 13.99613 + 13.99616 #Limit of Agreement amplitude

#Model 6:
lm6 <- lm(df$BMXHIP ~ df$BMXWT + df$BMXWAIST)
summary(lm6)
AIC(lm6)
BIC(lm6)
 #Bland_Altman:
df$BMXHIP_6 <- 45.073412 + (0.307425*df$BMXWT) + (0.365595*df$BMXWAIST)
df$md_6 <- df$BMXHIP-df$BMXHIP_6
df$average_md_6 <- (df$BMXHIP+df$BMXHIP_6)/2 #average between measured and predicted
df$mean_BMXHIP_6 <- mean(df$BMXHIP_6)
df$sd_BMXHIP_6 <- sd(df$BMXHIP_6)
df$mean_md_6 <- mean(df$md_6)
df$sd_md_6 <- sd(df$md_6)

df$LLA_6 <- df$mean_md_6 - (1.96*df$sd_md_6) #lower limit of agreement
df$ULA_6 <- df$mean_md_6 + (1.96*df$sd_md_6) #upper limit of agreement

plot(x=df$BMXHIP, y=df$md_6)
abline(h=0)
abline(h=-0.000002641691, col='blue', lty=2) #Bias 
abline(h=-12.79524, col='red', lty=2) #LLA line
abline(h=12.79523, col='red', lty=2) #ULA line
#expr <- vector("expression", 4)
#expr[[1]] <- bquote(Mean[diff]==-0.000002641691)
#expr[[2]] <- bquote(SD[diff]==6.52818)
#expr[[3]] <- bquote(LL==-12.79524)
#expr[[4]] <- bquote(UL==12.79523)
#legend("topleft", bty = "n", legend = expr)

df$LAAmplitude_6 <- 12.79524 + 12.79523 #Limit of Agreement amplitude


#Model 7:
lm7 <- lm(df$BMXHIP ~ df$BMXHT + df$BMXWT + df$BMXWAIST)
summary(lm7)
AIC(lm7)
BIC(lm7)
 #Bland-Altman:
df$BMXHIP_7 <- 130.824112 + (-0.488518*df$BMXHT) + (0.617183*df$BMXWT) + (0.064531*df$BMXWAIST)
df$md_7 <- df$BMXHIP-df$BMXHIP_7
df$average_md_7 <- (df$BMXHIP+df$BMXHIP_7)/2 #average between measured and predicted
df$mean_BMXHIP_7 <- mean(df$BMXHIP_7)
df$sd_BMXHIP_7 <- sd(df$BMXHIP_7)
df$mean_md_7 <- mean(df$md_7)
df$sd_md_7 <- sd(df$md_7)

df$LLA_7 <- df$mean_md_7 - (1.96*df$sd_md_7) #lower limit of agreement
df$ULA_7 <- df$mean_md_7 + (1.96*df$sd_md_7) #upper limit of agreement

plot(x=df$BMXHIP, y=df$md_7)
abline(h=0)
abline(h=0.00004144073, col='blue', lty=2) #Bias 
abline(h=-10.4971, col='red', lty=2) #LLA line
abline(h=10.49718, col='red', lty=2) #ULA line
#expr <- vector("expression", 4)
#expr[[1]] <- bquote(Mean[diff]==0.00004144073)
#expr[[2]] <- bquote(SD[diff]==5.355683)
#expr[[3]] <- bquote(LL==-10.4971)
#expr[[4]] <- bquote(UL==10.49718)
#legend("topleft", bty = "n", legend = expr)

df$LAAmplitude_7 <- 10.4971 + 10.49718 #Limit of Agreement amplitude

#Z-Score for all Models:
df$zscore_BMXHIP <- (df$BMXHIP-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP)

df$zscore_BMXHIP_1 <- (df$BMXHIP_1-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP_1)
ggplot(df, aes(x=x) ) +
  # Top
  geom_density( aes(x = zscore_BMXHIP, y = ..density..), fill="#69b3a2" ) +
  geom_label( aes(x=3, y=0.25, label="Measured BMXHIP"), color="#69b3a2") +
  # Bottom
  geom_density( aes(x = zscore_BMXHIP_1, y = -..density..), fill= "#404080") +
  geom_label( aes(x=3, y=-0.25, label="Model 1 BMXHIP"), color="#404080") +
  theme_ipsum() +
  xlab("Z-Score BMXHIP")


df$zscore_BMXHIP_2 <- (df$BMXHIP_2-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP_2)
ggplot(df, aes(x=x) ) +
  # Top
  geom_density( aes(x = zscore_BMXHIP, y = ..density..), fill="#69b3a2" ) +
  geom_label( aes(x=3, y=0.25, label="Measured BMXHIP"), color="#69b3a2") +
  # Bottom
  geom_density( aes(x = zscore_BMXHIP_2, y = -..density..), fill= "#404080") +
  geom_label( aes(x=3, y=-0.25, label="Model 2 BMXHIP"), color="#404080") +
  theme_ipsum() +
  xlab("Z-Score BMXHIP")

df$zscore_BMXHIP_3 <- (df$BMXHIP_3-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP_3)
ggplot(df, aes(x=x) ) +
  # Top
  geom_density( aes(x = zscore_BMXHIP, y = ..density..), fill="#69b3a2" ) +
  geom_label( aes(x=3, y=0.25, label="Measured BMXHIP"), color="#69b3a2") +
  # Bottom
  geom_density( aes(x = zscore_BMXHIP_3, y = -..density..), fill= "#404080") +
  geom_label( aes(x=3, y=-0.25, label="Model 3 BMXHIP"), color="#404080") +
  theme_ipsum() +
  xlab("Z-Score BMXHIP")

df$zscore_BMXHIP_4 <- (df$BMXHIP_4-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP_4)
ggplot(df, aes(x=x) ) +
  # Top
  geom_density( aes(x = zscore_BMXHIP, y = ..density..), fill="#69b3a2" ) +
  geom_label( aes(x=3, y=0.25, label="Measured BMXHIP"), color="#69b3a2") +
  # Bottom
  geom_density( aes(x = zscore_BMXHIP_4, y = -..density..), fill= "#404080") +
  geom_label( aes(x=3, y=-0.25, label="Model 4 BMXHIP"), color="#404080") +
  theme_ipsum() +
  xlab("Z-Score BMXHIP")

df$zscore_BMXHIP_5 <- (df$BMXHIP_5-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP_5)
ggplot(df, aes(x=x) ) +
  # Top
  geom_density( aes(x = zscore_BMXHIP, y = ..density..), fill="#69b3a2" ) +
  geom_label( aes(x=3, y=0.25, label="Measured BMXHIP"), color="#69b3a2") +
  # Bottom
  geom_density( aes(x = zscore_BMXHIP_5, y = -..density..), fill= "#404080") +
  geom_label( aes(x=3, y=-0.25, label="Model 5 BMXHIP"), color="#404080") +
  theme_ipsum() +
  xlab("Z-Score BMXHIP")

df$zscore_BMXHIP_6 <- (df$BMXHIP_6-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP_6)
ggplot(df, aes(x=x) ) +
  # Top
  geom_density( aes(x = zscore_BMXHIP, y = ..density..), fill="#69b3a2" ) +
  geom_label( aes(x=3, y=0.25, label="Measured BMXHIP"), color="#69b3a2") +
  # Bottom
  geom_density( aes(x = zscore_BMXHIP_6, y = -..density..), fill= "#404080") +
  geom_label( aes(x=3, y=-0.25, label="Model 6 BMXHIP"), color="#404080") +
  theme_ipsum() +
  xlab("Z-Score BMXHIP")

df$zscore_BMXHIP_7 <- (df$BMXHIP_7-df$mean_BMXHIP)/df$sd_BMXHIP
mean(df$zscore_BMXHIP_7)
ggplot(df, aes(x=x) ) +
  # Top
  geom_density( aes(x = zscore_BMXHIP, y = ..density..), fill="#69b3a2" ) +
  geom_label( aes(x=3, y=0.25, label="Measured BMXHIP"), color="#69b3a2") +
  # Bottom
  geom_density( aes(x = zscore_BMXHIP_7, y = -..density..), fill= "#404080") +
  geom_label( aes(x=3, y=-0.25, label="Model 7 BMXHIP"), color="#404080") +
  theme_ipsum() +
  xlab("Z-Score BMXHIP")
