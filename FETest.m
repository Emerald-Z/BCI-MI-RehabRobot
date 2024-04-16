%% call extract features
%do for all signals being processed -- go trial by trial
%runs.eeg{1}{j} <-- 1 = run #, j = sample number
runs = Subject1.offline.runs;
offline_feats = cell(3,1);

for i=1:3
    feats = cell(20, 1);
    featsR = cell(20, 1);
    featureMatrix = [];
    for j=1:20
        filteredSignal = runs.eeg{i}{j};
        window_size = 0.2;
        fs = 512;
        label = runs.labels{i}(j);
        overlap = 0.75;
        
        [MAV, VAR, RMS, WL, ZC, SSC, AR, labels] = extract_features(window_size, overlap, fs, filteredSignal, label);
        
        %implement PCA
        % Combine features into a matrix
        temp = [MAV; VAR; RMS; WL; ZC; SSC; AR]';
        featureMatrix = [featureMatrix; temp];
        
        % Standardize features
        
    end

    featureMatrix = zscore(featureMatrix);
        
    % Perform PCA
    [coeff, score, latent, tsquared, explained] = pca(featureMatrix);
    
    % Plot the explained variance to see how many components to retain
    % figure;
    % plot(cumsum(explained), 'o-');
    % xlabel('Number of Principal Components');
    % ylabel('Cumulative Variance Explained (%)');
    % title('Explained Variance by PCA Components');
    
    % Choose components that explain, e.g., at least 95% of the variance
    cumVar = cumsum(explained);
    numComponents = find(cumVar >= 95, 1, "first");
    model_input_feats = score(:, 1:numComponents);
    % disp(size(model_input_feats));

    % Extract the top scoring features based on principal components
    top_features = coeff(:, 1:numComponents);
    
    % Find indices of highest scoring features
    [~, sorted_indices] = sort(sum(abs(top_features), 2), 'descend');
    disp(size(featureMatrix(:, sorted_indices(1:4))));
    feats{j} = model_input_feats(:, 1:3);
    % featsR{j} = 

    offline_feats{i} = featureMatrix;
end

for i=1:4
    online_feats = cell(3,1);
    for  j=1:20
        feats = cell(20, 1);
        filteredSignal = session2.eeg{i}{j};
        window_size = 5;
        fs = 1;
        label = session2.labels.type{i}(j);
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
        % figure;
        % plot(cumsum(explained), 'o-');
        % xlabel('Number of Principal Components');
        % ylabel('Cumulative Variance Explained (%)');
        % title('Explained Variance by PCA Components');
        
        % Choose components that explain, e.g., at least 95% of the variance
        cumVar = cumsum(explained);
        numComponents = find(cumVar >= 95, 1, 'first');
        model_input_feats = score(:, 1:numComponents);
        disp(size(model_input_feats))
        feats{j} = model_input_feats;
    end
    online_feats{i} = feats;
end
