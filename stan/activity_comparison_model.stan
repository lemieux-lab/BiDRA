data {
    int N1;        
    real x1[N1];
    real y1[N1];

    int N2;        
    real x2[N2];
    real y2[N2];
    
    int N_infer;
    real x_infer[N_infer];
    
    real LDR_mu[2];
    real LDR_sigma[2];
    
    real HDR_mu[2];
    real HDR_sigma[2];
    
    real I_alpha[2];
    real I_beta[2];

    real S_mu[2];
    real S_sigma[2];

    real s_pos[2]; 
    real s_scale[2];
} 

parameters {
    real HDR1;
    real HDR2;

    real LDR1;
    real LDR2;

    real<lower=-1> I1;
    real<lower=1> I2;

    real<lower=0> S1;
    real<lower=0> S2;

    real<lower=0> sigma1;
    real<lower=0> sigma2;
}

model {
    LDR1 ~ normal(LDR_mu[1], LDR_sigma[1]) T[0, ];
    LDR2 ~ normal(LDR_mu[2], LDR_sigma[2]) T[0, ];

    HDR1 ~ normal(HDR_mu[1], HDR_sigma[1]);
    HDR2 ~ normal(HDR_mu[2], HDR_sigma[2]);

    I1 ~ uniform(I_alpha[1], I_beta[1]);
    I2 ~ uniform(I_alpha[2], I_beta[2]);

    S1 ~ lognormal(S_mu[1], S_sigma[1]);
    S2 ~ lognormal(S_mu[2], S_sigma[2]);

    sigma1 ~ cauchy(s_pos[1], s_scale[1]);
    sigma2 ~ cauchy(s_pos[2], s_scale[2]);
    
    for (i in 1:N1) {
        y1[i] ~ normal(LDR1 + (HDR1 - LDR1) / (1 + 10^(S1 * (I1 - x1[i]))), sigma1);
    }

    for (j in 1:N2) {
        y2[j] ~ normal(LDR2 + (HDR2 - LDR2) / (1 + 10^(S2 * (I2 - x2[j]))), sigma2);
    }

}
generated quantities{
    real y_infer1[N_infer];
    real y_infer2[N_infer];
    
    real diffLDR;
    real diffHDR;
    real diffI;
    real diffS;

    for (i in 1:N_infer) {
        y_infer1[i] = LDR1 + (HDR1 - LDR1) / (1 + 10^(S1 * (I1 - x_infer[i])));
        y_infer2[i] = LDR2 + (HDR2 - LDR2) / (1 + 10^(S2 * (I2 - x_infer[i])));
    }
    
    diffLDR = LDR2 - LDR1;
    diffHDR = HDR2 - HDR1;
    diffI = I2 - I1;
    diffS = S2 - S1;
    
    }