data {
    int<lower=0> N;        
    real x[N];
    real y[N];
    }    
parameters {
    real LDR;          
    real<lower=0> HDR;          
    real<lower=-1> I;          
    real<lower=0> S; 
    real<lower=0> sigma;
    }
model {
    LDR ~ normal(0,5);
    HDR ~ normal(100,20);
    I ~ uniform(-1,10);
    S ~ lognormal(0.5,1);
    sigma ~ cauchy(0,0.2);
    
    for (i in 1:N) {
        y[i] ~ normal(LDR + (HDR - LDR) / (1 + 10^(S * (I - x[i]))), sigma);
    }
}
