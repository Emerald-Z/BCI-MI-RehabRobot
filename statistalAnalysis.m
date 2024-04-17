%% statistical analysis of results
% per sample - assume quadratic

%% session 2
sample_probabilities = cell(4, 1);
for i=1:4
    sampleFeatures = online_feats{i, 1};
    % sampleFeatures
    [labels, scores] = predict(final_model, sampleFeatures);
    disp(scores)
    sample_probabilities{i} = scores;
end


%% evidence accumulation
alpha = 0.01;
for t=1:T
    p_t = classify; % classify with final model
    P(t) = alpha * P(t-1) + (1-alpha) * p_t

end