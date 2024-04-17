%% Feature Extraction
runs = Subject1.offline.runs;
offline_feats = cell(3,1);

fs = 512;

% Initialize variables to track the best result
best_accuracy = 0;
best_window_size = 0;
best_overlap = 0;

% Define the window sizes and overlaps to test
window_sizes = [0.5, 1, 1.5];
window_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9];
window_sizes = [2.2, 2.3, 2.4];
overlaps = [0.5, 0.75, 0.9];
overlaps = [0.5, 0.53, 0.55, 0.58, 0.6, 0.63, 0.65, 0.68, 0.7];

for window_size = window_sizes
    for overlap = overlaps
        % Offline features
        min_temp_offline_size = inf;
        min_temp_online_size = inf;
        for i=1:3
            featureMatrix = [];
            for j=1:20
                filteredSignal = runs.eeg{i}{j};
                label = runs.labels{i}(j);
                [MAV, VAR, RMS, WL, ZC, SSC, AR, labels] = extract_features(window_size, overlap, fs, filteredSignal, label);
                % Combine features into a matrix
                temp = [MAV; VAR; RMS; WL; ZC; SSC; AR]';
                min_temp_offline_size = min(min_temp_offline_size, size(temp, 1));
                featureMatrix = [featureMatrix; temp];
            end
            offline_feats{i} = featureMatrix;
        end
        
        % Online features
        for i=1:4
            featureMatrix = [];
            for j=1:20
                filteredSignal = session2.eeg{i}{j};
                label = session2.labels.type{i}(j);
                [MAV, VAR, RMS, WL, ZC, SSC, AR, labels] = extract_features(window_size, overlap, fs, filteredSignal, label);
                % Combine features into a matrix
                temp = [MAV; VAR; RMS; WL; ZC; SSC; AR]';
                min_temp_online_size = min(min_temp_online_size, size(temp, 1));
                featureMatrix = [featureMatrix; temp];
            end
            online_feats{i} = featureMatrix;
        end
        
        % Shorten featureMatrix to the smallest temp value found (issues
        % with length of data)
        for i=1:3
            %size(offline_feats{i})
            offline_feats{i} = offline_feats{i}(1:min_temp_offline_size*20, :);
            %size(offline_feats{i})
        end
        for i=1:4
            online_feats{i} = online_feats{i}(1:min_temp_online_size*20, :);
        end

        %% Classification
        numFeatures = 7;

        % Build classifier using offline data and test with run-wise cross-validation
        k = 3; % num runs
        classificationErrors = zeros(k, 1);
        CM_fold_avg=0;
        CM_test_avg=0;
        %FEATUREL=zeros(3, 20 * dataLen);
        FEATUREL=zeros(3, 20 * min_temp_offline_size);
        for i=1:3
            for j=1:20
                %FEATUREL(i, (j-1)* dataLen + 1 : j* dataLen) = offline.labels{i, 1}(j,1) * ones(dataLen, 1);
                FEATUREL(i, (j-1)* min_temp_offline_size + 1 : j* min_temp_offline_size) = offline.labels{i, 1}(j,1) * ones(min_temp_offline_size, 1);
            end
        end

        bestClassifiers=zeros(4,3);
        %DATA = zeros(3, 20 *dataLen, numFeatures);
        DATA = zeros(3, 20 *min_temp_offline_size, numFeatures);

        for i=1:3
            %DATA(i, :,:) = offline_feats{i,1}(1:min(feat_sizes), :);
            DATA(i, :,:) = offline_feats{i,1};

        end

        % Linear Classifier
        MAV_ALL_acc=zeros(1, k);
        for i=1:k
            test = squeeze(DATA(i, :, :));

            other_indices = setdiff(1:size(DATA, 1), i);
            train = vertcat(DATA(other_indices, :, :));
            train = reshape(train, [], numFeatures);
            train_labels = vertcat(FEATUREL(other_indices, :, :));
            train_labels = reshape(train_labels', [], 1);
            test_labels = squeeze(FEATUREL(i, :));

            [TstMAVVARFALL, TstMAVVARErrALL] = classify(test,train, train_labels);
            [~, ~, MAV_ALL_acc(i), ~] = confusion(test_labels, TstMAVVARFALL);
        end
        CM_fold_avg=mean(MAV_ALL_acc) / 100;
        [maxValue, maxIndex] = max(MAV_ALL_acc);
        disp("Linear");
        disp(CM_fold_avg * 100);
        bestClassifiers(1, 1)=maxIndex;
        bestClassifiers(1, 2)=maxValue/100;

        % Quadratic Classifier
        cvAccuracy = zeros(k, 1);
        classifiers1=cell(k,1);

        for i = 1:k
            other_indices = setdiff(1:size(DATA, 1), i);
            cvTrainFeatures = vertcat(DATA(other_indices, :, :));
            cvTrainFeatures = reshape(cvTrainFeatures, [], numFeatures);
            cvTrainLabels = vertcat(FEATUREL(other_indices, :, :));
            cvTrainLabels = reshape(cvTrainLabels', [], 1);
            cvValidationFeatures = squeeze(DATA(i, :, :));
            cvValidationLabels = squeeze(FEATUREL(i, :));

            classifiers1{i} = fitcdiscr(cvTrainFeatures, cvTrainLabels, 'DiscrimType', 'quadratic');

            [predictedLabels, ~] = predict(classifiers1{i}, cvValidationFeatures);
            cvAccuracy(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
        end

        CMQ_fold_avg = mean(cvAccuracy);
        [maxValue, maxIndex] = max(cvAccuracy);
        disp("Quadratic");
        disp(CMQ_fold_avg * 100);
        bestClassifiers(2, 1)=maxIndex;
        bestClassifiers(2, 2)=maxValue;

        % SVM Classifier
        cvAccuracy2 = zeros(k, 1);
        classifiers2=cell(k,1);
        t = templateLinear;

        for i = 1:k
            other_indices = setdiff(1:size(DATA, 1), i);

            cvTrainFeatures = vertcat(DATA(other_indices, :, :));
            cvTrainFeatures = reshape(cvTrainFeatures, [], numFeatures);
            cvTrainLabels = vertcat(FEATUREL(other_indices, :, :));
            cvTrainLabels = reshape(cvTrainLabels', [], 1);
            cvValidationFeatures = squeeze(DATA(i, :, :));
            cvValidationLabels = squeeze(FEATUREL(i, :));

            classifiers2{i} = fitcecoc(cvTrainFeatures, cvTrainLabels, 'Learners', t, 'Coding', 'onevsone');

            predictedLabels = predict(classifiers2{i}, cvValidationFeatures);
            cvAccuracy2(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
        end

        CMS_fold_avg = mean(cvAccuracy2);
        [maxValue, maxIndex] = max(cvAccuracy2);
        disp("SVM");
        disp(CMS_fold_avg * 100);
        bestClassifiers(3, 1)=maxIndex;
        bestClassifiers(3, 2)=maxValue;

        % Gaussian Kernel Classifier
        cvAccuracy3 = zeros(k, 1);
        classifiers3=cell(k,1);

        for i = 1:k
            other_indices = setdiff(1:size(DATA, 1), i);

            cvTrainFeatures = vertcat(DATA(other_indices, :, :));
            cvTrainFeatures = reshape(cvTrainFeatures, [], numFeatures);
            cvTrainLabels = vertcat(FEATUREL(other_indices, :, :));
            cvTrainLabels = reshape(cvTrainLabels', [], 1);
            cvValidationFeatures = squeeze(DATA(i, :, :));
            cvValidationLabels = squeeze(FEATUREL(i, :));

            classifiers3{i} = fitckernel(cvTrainFeatures, cvTrainLabels);

            predictedLabels = predict(classifiers3{i}, cvValidationFeatures);
            cvAccuracy3(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
        end

        CMG_fold_avg = mean(cvAccuracy3);
        [maxValue, maxIndex] = max(cvAccuracy3);
        disp("Gaussian Kernel");
        disp(CMG_fold_avg * 100)
        bestClassifiers(4, 1)=maxIndex;
        bestClassifiers(4, 2)=maxValue;

        % Final classification on all offline data
        [~, maxIndex] = max(bestClassifiers(:, 2));

        trainLabels = reshape(FEATUREL, [], 1);
        trainFeatures = reshape(DATA, [], numFeatures);
        final_model = fitcecoc(trainFeatures, trainLabels, 'Learners', t, 'Coding', 'onevsone');

        % Assess on session 2, 3
        finalCVaccuracy = zeros(2, 1);
        online = Subject1.online.session2;
        online_k = 4;


        %{
        TESTFEATUREL=zeros(online_k, 20 * dataLen);
        for i=1:online_k
            for j=1:20
                TESTFEATUREL(i, (j-1)* dataLen + 1 : j* dataLen) = online.labels.type{i}(j) * ones(dataLen, 1);
            end
        end

        TESTDATA = zeros(online_k, 20 * dataLen, numFeatures);
        for i=1:online_k
            TESTDATA(i, :,:) = online_feats{i,1}(1:20*dataLen, :);
        end

        validationLabels = reshape(TESTFEATUREL, [], 1);
        validationFeatures = reshape(TESTDATA, [], numFeatures);
        predictedLabels = predict(final_model, validationFeatures);

        finalCVAccuracy(1) = sum(predictedLabels == validationLabels(1:online_k*20*dataLen)) / length(validationLabels);
        disp(['Final CV Accuracy (Window Size: ', num2str(window_size), ', Overlap: ', num2str(overlap), '): ', num2str(finalCVAccuracy(1))]);
        %}
        
        
        % Calculate the average accuracy from offline classifiers
        offline_accuracies = [CM_fold_avg, CMQ_fold_avg, CMS_fold_avg, CMG_fold_avg];
        best_avg_accuracy = max(offline_accuracies);

        % Update the best result if the current offline average accuracy is higher
        if best_avg_accuracy > best_accuracy
            best_accuracy = best_avg_accuracy;
            best_window_size = window_size;
            best_overlap = overlap;
        end

        disp(['Best Offline Accuracy (Window Size: ', num2str(window_size), ', Overlap: ', num2str(overlap), '): ', num2str(best_avg_accuracy)]);
    end
end

disp(['Best Result - Window Size: ', num2str(best_window_size), ', Overlap: ', num2str(best_overlap), ', Accuracy: ', num2str(best_accuracy)]);
