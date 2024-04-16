%% call extract features
%do for all signals being processed
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
