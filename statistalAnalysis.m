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
alpha = 0.1;
Pt = cell(4, 1);
for i=1:4
    temp = zeros(size(sample_probabilities{i}));
    for t = 1:length(sample_probabilities{i})
        if t == 1
            temp(t, :) = sample_probabilities{i}(t, :); % For the first sample, use the original probability distribution
        else
            temp(t, :) = alpha * temp(t-1, :) + (1 - alpha) * sample_probabilities{i}(t, :); % Apply exponential smoothing
        end
    end
    Pt{i} = temp;
end
