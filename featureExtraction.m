%% call extract features
%do for all signals being processed -- go trial by trial
%runs.eeg{1}{j} <-- 1 = run #, j = sample number
runs = Subject1.offline.runs;
offline_feats = cell(3,1);
feats_used = zeros(1, 8);

%save('avg_coeffs.mat', 'avg_coeffs');
%save('avg_mean.mat', 'avg_mean');
avg_coeffs = load('avg_coeffs.mat').avg_coeffs;
avg_mean = load('avg_mean.mat').avg_mean;

feat_indices = [2, 5, 3, 6, 7];
%avg_coeffs = zeros(7,7);
%avg_mean = zeros(1,7);

for i=1:3
    feats = cell(20, 1);
    featsR = cell(20, 1);
    feats_fixed = cell(20,1);
    for j=1:20
        filteredSignal = runs.eeg{i}{j};
        window_size = 0.2;
        fs = 512;
        label = runs.labels{i}(j);
        overlap = 0.75;
        
        [MAV, VAR, RMS, WL, ZC, SSC, AR,  EN, FRAC, GFP, GD, labels] = extract_avg_features(window_size, overlap, fs, filteredSignal, label);
        
        %implement PCA
        % Combine features into a matrix
        featureMatrix = [MAV; VAR; RMS; WL; ZC; SSC; AR]';
        
        % Standardize features
        featureMatrix = zscore(featureMatrix);
        
        % Perform PCA
        [coeff, score, latent, tsquared, explained, mu] = pca(featureMatrix);
        %disp(size(mu))

        X_centered = bsxfun(@minus, featureMatrix, avg_mean); 
        new_feats = X_centered * avg_coeffs;
        
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

        %disp(size(featureMatrix))
        
        % Find indices of highest scoring features
        [~, sorted_indices] = sort(sum(abs(top_features), 2), 'descend');
        disp(sorted_indices)
        %for ind=1:5
        %    feats_used(sorted_indices(ind)) = feats_used(sorted_indices(i)) + 1;
        %end
        %disp(size(featureMatrix(:, sorted_indices(1:4))));
        size(model_input_feats)
        feats{j} = model_input_feats(:, 1:3);
        featsR{j} = featureMatrix(:, sorted_indices(1:4));
        feats_fixed{j} = new_feats(:, feat_indices);
        %avg_coeffs = avg_coeffs + coeff;
        %avg_mean = avg_mean + mu;
    end
    %disp("HERE")
    %disp(size(feats_fixed{1}))
    %disp(size(featsR{1}))
    offline_feats{i} = feats_fixed;
end

%avg_coeffs = avg_coeffs/60;
%avg_mean = avg_mean/60;

%disp(feats_used)
%disp(avg_coeffs)
%disp(size(offline_feats))

for i=1:4
    online_feats = cell(3,1);
    for  j=1:20
        feats = cell(20, 1);
        feats_fixed = cell(20,1);
        filteredSignal = session2.eeg{i}{j};
        window_size = 0.2;
        fs = 512;
        label = session2.labels.type{i}(j);
        overlap = 0.75;


        
        [MAV, VAR, RMS, WL, ZC, SSC, AR, labels] = extract_avg_features(window_size, overlap, fs, filteredSignal, label);
        
        %implement PCA
        % Combine features into a matrix
        featureMatrix = [MAV; VAR; RMS; WL; ZC; SSC; AR]';
        
        % Standardize features
        featureMatrix = zscore(featureMatrix);
        
        % Perform PCA
        [coeff, score, latent, tsquared, explained] = pca(featureMatrix);

        X_centered = bsxfun(@minus, featureMatrix, avg_mean); 
        new_feats = X_centered * avg_coeffs;
        
        % Plot the explained variance to see how many components to retain
        % figure;
        % plot(cumsum(explained), 'o-');
        % xlabel('Number of Principal Components');
        % ylabel('Cumulative Variance Explained (%)');
        % title('Explained Variance by PCA Components');
        
        % Choose components that explain, e.g., at least 95% of the variance
        cumVar = cumsum(explained);
        %numComponents = find(cumVar >= 95, 1, 'first');
        model_input_feats = score(:, 1:4);
        [sortedLoadings, featureIdx] = sort(abs(coeff(:,1)), 'descend');
        %disp(size(model_input_feats))
        feats{j} = model_input_feats;
        feats_fixed{j} = new_feats(:, feat_indices);
    end
    online_feats{i} = feats_fixed;
end
