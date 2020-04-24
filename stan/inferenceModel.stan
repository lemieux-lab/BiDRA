data {
    int<lower=0> N;             // number of observations
    int<lower=0> N_inference;   // number of values not tested
    real x[N];
    real x_inference[N_inference];
    real y[N];
    real LDRmu;
    real LDRsigma;
    real HDRmu;
    real HDRsigma;
    real Imu;
    real Isigma;
    real Smu;
    real Ssigma;
    }    
parameters {
    real LDR;          // low dose plateau
    real HDR;          // high dose plateau
    real I;          // ic50
    real<lower=0> S; // slope
    real<lower=0> sigma;
    }
model {
    LDR ~ normal(LDRmu, LDRsigma);
    HDR ~ normal(HDRmu, HDRsigma);
    I ~ normal(Imu, Isigma);
    S ~ normal(Smu, Ssigma);
    
    for (i in 1:N) {
        y[i] ~ normal(LDR + (HDR - LDR) / (1 + 10^(S * (I - x[i]))), sigma);
    }
    }
generated quantities{
    real y_predict_inference[N_inference];
    
    for (k in 1:N_inference)
        y_predict_inference[k] = LDR + (HDR - LDR) / (1 + 10.0^(S * (I - x_inference[k])));
    }
