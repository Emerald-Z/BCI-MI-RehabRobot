%% statistical analysis of results


%% evidence accumulation
alpha = 0.01;
for t=1:T
    p_t = classify; % classify with final model
    P(t) = alpha * P(t-1) + (1-alpha) * p_t

end