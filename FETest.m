%% call extract features
%do for all signals being processed -- go trial by trial
%runs.eeg{1}{j} <-- 1 = run #, j = sample number
runs = Subject3.offline.runs;
offline_feats = cell(3,1);
offline_labels = cell(3,1);
window_size = 1.2;
overlap = 0.68;

for i=1:3
    feats = cell(20, 1);
    featsR = cell(20, 1);
    featureMatrix = [];
    of = cell(20, 1);
    for j=1:20
        filteredSignal = runs.eeg{i}{j};
        fs = 512;
        label = runs.labels{i}(j);

        
        [MAV, VAR, RMS, WL, ZC, SSC, AR, EN, FRAC, CWT, labels] = extract_features(window_size, overlap, fs, filteredSignal, label);
        of{j} = ones(length(labels), 1) * label;
        
        %implement PCA
        % Combine features into a matrix
        temp = [MAV; VAR; SSC; ZC; AR; EN]';
        temp = zscore(temp);
        featureMatrix = [featureMatrix; temp];
        
        % Standardize features
        
    end

    % featureMatrix = zscore(featureMatrix);
        
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
    % feats{j} = model_input_feats(:, 1:3);
    % featsR{j} = 

    offline_feats{i} = featureMatrix;
    offline_labels{i} = of;
end

session2 = Subject1.online.session2;
online_feats = cell(4,1);
online_labels = cell(4,1);
for i=1:4
    feats = cell(20, 1);
    ol = cell(20,1);
    featureMatrix = [];

    for j=1:20
        filteredSignal = session2.eeg{i}{j};
        fs = 512;
        label = session2.labels.type{1, i}(j);
        % sustain trial
        if session2.labels.sustain{1, i}(j) == 0
            filteredSignal = filteredSignal;
        else
            % find where sustain starts
            sustainStart = session2.labels.sustain_idx{1, i}(j);
            filteredSignal = filteredSignal(1:sustainStart); % where sustain starts
        end
        [MAV, VAR, RMS, WL, ZC, SSC, AR, labels] = extract_features(window_size, overlap, fs, filteredSignal, label);
        disp(length(labels))
        ol{j} = ones(length(labels), 1) * label;
        % Combine features into a matrix
        temp = [MAV; VAR; WL; SSC; AR]';
        temp = zscore(temp);

        featureMatrix = [featureMatrix; temp];
        %implement PCA
        % Combine features into a matrix
        % featureMatrix = [MAV; VAR; RMS; WL; ZC; SSC; AR]';
        % 
        % % Standardize features
        % 
        % % Perform PCA
        % [coeff, score, latent, tsquared, explained] = pca(featureMatrix);
        % 
        % % Plot the explained variance to see how many components to retain
        % % figure;
        % % plot(cumsum(explained), 'o-');
        % % xlabel('Number of Principal Components');
        % % ylabel('Cumulative Variance Explained (%)');
        % % title('Explained Variance by PCA Components');
        % 
        % % Choose components that explain, e.g., at least 95% of the variance
        % cumVar = cumsum(explained);
        % numComponents = find(cumVar >= 95, 1, 'first');
        % model_input_feats = score(:, 1:numComponents);
        % % disp(size(model_input_feats))
        % feats{j} = model_input_feats;
    end
    
    online_feats{i} = featureMatrix;
    online_labels{i} = ol;
end
