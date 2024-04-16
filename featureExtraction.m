%% call extract features
%do for all signals being processed -- go trial by trial
%runs.eeg{1}{j} <-- 1 = run #, j = sample number
for i=1:3
    offline_feats = cell(3,1);
    for  j=1:20
        feats = cell(20, 1);
        filteredSignal = runs.eeg{i}{j};
        window_size = 5;
        fs = 1;
        label = runs.labels{i}(j);
        overlap = 0.75;
        
        [MAV, VAR, RMS, WL, ZC, SSC, AR, labels] = extract_features(window_size, overlap, fs, filteredSignal, label);
        
        %implement PCA
        % Combine features into a matrix
        featureMatrix = [MAV; VAR; RMS; WL; ZC; SSC; AR]';
        
        % Standardize features
        featureMatrix = zscore(featureMatrix);
        
        % Perform PCA
        [coeff, score, latent, tsquared, explained] = pca(featureMatrix);
        
        % Plot the explained variance to see how many components to retain
        figure;
        plot(cumsum(explained), 'o-');
        xlabel('Number of Principal Components');
        ylabel('Cumulative Variance Explained (%)');
        title('Explained Variance by PCA Components');
        
        % Choose components that explain, e.g., at least 95% of the variance
        cumVar = cumsum(explained);
        numComponents = find(cumVar >= 95, 1, 'first');
        model_input_feats = score(:, 1:numComponents);
        disp(size(model_input_feats))
        feats{j} = model_input_feats;
    end
    offline_feats{i} = feats;
end
