data {
    int<lower=0> N;        
    real x[N];
    real y[N];

    int<lower=0> N_infer;
    real x_infer[N_infer];

    real HDR_mu;
    real HDR_sigma;

    real LDR_mu;
    real LDR_sigma;

    real I_alpha;
    real I_beta;

    real S_mu;
    real S_sigma;

    real s_pos; 
    real s_scale;
}

parameters {
    real HDR;
    real LDR;                    
    real<lower=-1> I;          
    real<lower=0> S; 
    real<lower=0> sigma;
}

model {
    LDR ~ normal(LDR_mu, LDR_sigma) T[0, ];
    HDR ~ normal(HDR_mu, HDR_sigma);
    I ~ uniform(I_alpha, I_beta);
    S ~ lognormal(S_mu, S_sigma);
    sigma ~ cauchy(s_pos, s_scale);
    
    for (i in 1:N) {
        y[i] ~ normal(LDR + (HDR - LDR) / (1 + 10^(S * (I - x[i]))), sigma);
    }
}

generated quantities {
    real y_infer[N_infer];

    for (i in 1:N_infer) {
        y_infer[i] = LDR + (HDR - LDR) / (1 + 10^(S * (I - x_infer[i])));
    }
}
