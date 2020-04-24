data {
    int<lower=0> N1;             // number of observations
    int<lower=0> N2;
    int<lower=0> N_inference;   // number of values not tested
    
    real x1[N1];
    real x2[N2];
    real x_inference[N_inference];
    
    real y1[N1];
    real y2[N2];
    
    real LDRmu1;
    real LDRmu2;
    real LDRsigma1;
    real LDRsigma2;
    
    real HDRmu1;
    real HDRmu2;
    real HDRsigma1;
    real HDRsigma2;
    
    real Imu1;
    real Imu2;
    real Isigma1;
    real Isigma2;
    
    real Smu1;
    real Smu2;
    real Ssigma1;
    real Ssigma2;
    } 
parameters {
    real LDR1;
    real LDR2;
    
    real HDR1;
    real HDR2;
    
    real I1;
    real I2;
    
    real S1;
    real S2;
    
    real<lower=0> sigma1;
    real<lower=0> sigma2;
    }
transformed parameters {
    real y_predict1[N1];
    real y_predict2[N2];
    
    for (j in 1:N1)
        y_predict1[j] = LDR1 + (HDR1 - LDR1) / (1 + 10.0^(S1 * (I1 - x1[j])));
    
    for (k in 1:N2)
        y_predict2[k] = LDR2 + (HDR2 - LDR2) / (1 + 10.0^(S2 * (I2 - x2[k])));
    }
model {
    LDR1 ~ normal(LDRmu1, LDRsigma1);
    LDR2 ~ normal(LDRmu2, LDRsigma1);
    
    HDR1 ~ normal(HDRmu1, HDRsigma1);
    HDR2 ~ normal(HDRmu2, HDRsigma2);
    
    I1 ~ normal(Imu1, Isigma1);
    I2 ~ normal(Imu2, Isigma2);
    
    S1 ~ normal(Smu1, Ssigma1);
    S2 ~ normal(Smu2, Ssigma2);
    
    y1 ~ normal(y_predict1, sigma1);
    y2 ~ normal(y_predict2, sigma2);
    }
generated quantities{
    real y_predict_inference1[N_inference];
    real y_predict_inference2[N_inference];
    
    real diffLDR;
    real diffHDR;
    real diffI;
    real diffS;
    
    for (j in 1:N_inference)
        y_predict_inference1[j] = LDR1 + (HDR1 - LDR1) / (1 + 10.0^(S1 * (I1 - x_inference[j])));
    
    for (k in 1:N_inference)
        y_predict_inference2[k] = LDR2 + (HDR2 - LDR2) / (1 + 10.0^(S2 * (I2 - x_inference[k])));
    
    diffLDR = LDR2 - LDR1;
    diffHDR = HDR2 - HDR1;
    diffI = I2 - I1;
    diffS = S2 - S1;
    
    }