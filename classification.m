%% what is our train/test split? 
% in the guide we build classifier on offline data and test it on online
% test linear, quad, SVM
offline = Subject3.offline.runs;
numFeatures = 5;
%% build classifier using offline data and test with run wise cross val -> independence
% Linear - k - 1 fold, test on last run of subject data
dataLen = length(offline_feats{1,1}{1});
k = 3; % num runs
classificationErrors = zeros(k, 1);
CM_fold_avg=0;
CM_test_avg=0;
FEATUREL=zeros(3, dataLen * 20);
for i=1:3
    for j=1:20
        FEATUREL(i, (j-1)* dataLen + 1 : j* dataLen) = offline.labels{i, 1}(j,1) * ones(dataLen, 1);
    end
end

bestClassifiers=zeros(4,3); % the index of the classifier and its accuracy, test acc
DATA = zeros(3, dataLen * 20, numFeatures);
for i=1:3
    DATA(i, :,:) = vertcat(offline_feats{i,1}{:});
end

for i=1:size(DATA,1)
    DATA(i, :, :) = zscore(squeeze(DATA(i, :, :)));
end

CC_avg=zeros(2,2,1,k); 
CM=zeros(2,2);
MAV_ALL_acc=zeros(1, k);
for i=1:k
    % linear classifier
    test = squeeze(DATA(i, :, :));

    other_indices = setdiff(1:size(DATA, 1), i);
    train = vertcat(DATA(other_indices, :, :));
    train = reshape(train, [], numFeatures);
    train_labels = vertcat(FEATUREL(other_indices, :, :));
    train_labels = reshape(train_labels', [], 1);
    test_labels = squeeze(FEATUREL(i, :));
    % linear
    [TstMAVVARFALL TstMAVVARErrALL] = classify(test,train, train_labels);
    [CC_avg(:,:,1,i) dum1 MAV_ALL_acc(i) dum2] = confusion(test_labels, TstMAVVARFALL);

    % disp(MAV_ALL_acc(i))
    % TstAcc_MAVVAR_All_stor(i) = TstAcc_MAVVAR_ALL;
    CM(:,:,1) = CM(:,:,1) + confusionmat(test_labels, TstMAVVARFALL');
end
CM_fold_avg=mean(MAV_ALL_acc);
[maxValue, maxIndex] = max(MAV_ALL_acc);
disp("Max linear");
disp(maxValue)
disp("Mean linear");
disp(CM_fold_avg);
bestClassifiers(1, 1)=maxIndex;
bestClassifiers(1, 2)=maxValue/100;

% indices=(maxIndex-1)*(numFeatures*30)+1:maxIndex*(numFeatures*30);
% remaining_indices = setdiff(1:150*numFeatures, indices);
% 
% [MAVVARLabels MAVVARErr] = classify([run6_r; vrun6_r]', MAVVAR_Data_All_classes(:,remaining_indices)',MAVVAR_Labels_All_classes(remaining_indices));
% [TstCM_MAVVAR_ALL dum1 CM_test_avg dum2] = confusion(run6_labels, TstMAVVARFALL);
% bestClassifiers(1, 3)=CM_test_avg;

% display
% CM_MAVVAR_All_temp = confusionmat(run6_labels, TstMAVVARFALL);
% subplot(2, 2, sensor);
% cc=confusionchart(CM_MAVVAR_All_temp, {'Pinch','Point','Grasp'});
% cc.RowSummary = 'row-normalized';
% cc.Title=strcat('confusion matrix for MAV and VAR features for all 3 classes - Sensor ', sensor_labels{sensor});
% 

% Quadratic
cvAccuracy = zeros(k, 1); % Array to store cross-validation accuracy
CMQ_fold_avg=0;
CMQ_test_avg=0;
classifiers1=cell(k,1);

% Perform cross-validation to optimize classifier
for i = 1:k
    % Split data into training and validation sets for current fold
    other_indices = setdiff(1:size(DATA, 1), i);
    train = vertcat(DATA(other_indices, :, :));

    cvTrainFeatures = vertcat(DATA(other_indices, :, :));
    cvTrainFeatures = reshape(cvTrainFeatures, [], numFeatures);
    cvTrainLabels = vertcat(FEATUREL(other_indices, :, :));
    cvTrainLabels = reshape(cvTrainLabels', [], 1);
    cvValidationFeatures = squeeze(DATA(i, :, :));
    cvValidationLabels = squeeze(FEATUREL(i, :));
    
    % Train quadratic LDA classifier
    classifiers1{i} = fitcdiscr(cvTrainFeatures, cvTrainLabels, 'DiscrimType', 'quadratic');
    
    % Predict on validation set
    [predictedLabels, scores] = predict(classifiers1{i}, cvValidationFeatures);
    % disp(scores);
    % Compute accuracy for current fold
    cvAccuracy(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
end

% Average accuracy across folds to find optimal parameters
CMQ_fold_avg = mean(cvAccuracy);
[maxValue, maxIndex] = max(cvAccuracy);
disp("Max quadratic");
disp(maxValue * 100)
disp("Mean quadratic");
disp(CMQ_fold_avg * 100);
bestClassifiers(2, 1)=maxIndex;
bestClassifiers(2, 2)=maxValue;

% % Compute accuracy on testing set
% accuracy = sum(predictedLabels' == testingLabels) / length(run6_labels);
% bestClassifiers(2, 3)=accuracy;
% % Display test accuracy
% disp(['Subject ', num2str(1), ' - Test Accuracy: ', num2str(accuracy)]);

% Plot confusion matrix
% confusionMatrix = confusionmat(testingLabels, predictedLabels);
% 
% subplot(2, 2, sensor);
% cc=confusionchart(confusionMatrix, {'Pinch','Point','Grasp'});
% cc.RowSummary = 'row-normalized';
% cc.Title=strcat('confusion matrix for MAV and VAR features for all 3 classes - Sensor ', sensor_labels{sensor});

% SVM

% svmModel = fitcecoc(trainingData, trainingLabels, 'Learners', t, 'Coding', 'onevsone');
% Cross-validation
cvAccuracy2 = zeros(k, 1); % Array to store cross-validation accuracy
CMS_fold_avg=0;
CMS_test_avg=0;
classifiers2=cell(k,1);
t = templateLinear;

% Perform cross-validation to optimize classifier
for i = 1:k
    % Split data into training and validation sets for current fold
    other_indices = setdiff(1:size(DATA, 1), i);

    cvTrainFeatures = vertcat(DATA(other_indices, :, :));
    cvTrainFeatures = reshape(cvTrainFeatures, [], numFeatures);
    cvTrainLabels = vertcat(FEATUREL(other_indices, :, :));
    cvTrainLabels = reshape(cvTrainLabels', [], 1);
    cvValidationFeatures = squeeze(DATA(i, :, :));
    cvValidationLabels = squeeze(FEATUREL(i, :));
    
    % Train quadratic LDA classifier
    classifiers2{i} = fitcecoc(cvTrainFeatures, cvTrainLabels, 'Learners', t, 'Coding', 'onevsone');
    
    % Predict on validation set
    predictedLabels = predict(classifiers2{i}, cvValidationFeatures);
    
    % Compute accuracy for current fold
    cvAccuracy2(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
end

% Average accuracy across folds to find optimal parameters
CMS_fold_avg = mean(cvAccuracy2);
[maxValue, maxIndex] = max(cvAccuracy2);
disp("Max SVM");
disp(maxValue * 100)
disp("Mean SVM");
disp(CMS_fold_avg * 100);
bestClassifiers(3, 1)=maxIndex;
bestClassifiers(3, 2)=maxValue;

%% Gaussian Kernel
cvAccuracy3 = zeros(k, 1); % Array to store cross-validation accuracy
CMG_fold_avg=0;
CMG_test_avg=0;
classifiers3=cell(k,1);

% Perform cross-validation to optimize classifier
for i = 1:k
    % Split data into training and validation sets for current fold
    other_indices = setdiff(1:size(DATA, 1), i);

    cvTrainFeatures = vertcat(DATA(other_indices, :, :));
    cvTrainFeatures = reshape(cvTrainFeatures, [], numFeatures);
    cvTrainLabels = vertcat(FEATUREL(other_indices, :, :));
    cvTrainLabels = reshape(cvTrainLabels', [], 1);
    cvValidationFeatures = squeeze(DATA(i, :, :));
    cvValidationLabels = squeeze(FEATUREL(i, :));
    
    % Train quadratic LDA classifier
    classifiers3{i} = fitckernel(cvTrainFeatures, cvTrainLabels);
    
    % Predict on validation set
    predictedLabels = predict(classifiers3{i}, cvValidationFeatures);
    
    % Compute accuracy for current fold
    cvAccuracy3(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
    
    subplot(1, k, i);
    confusionMatrix = confusionmat(cvValidationLabels, predictedLabels);
    cc=confusionchart(confusionMatrix, {'Rest','Reach'});
    cc.RowSummary = 'row-normalized';
end

% Average accuracy across folds to find optimal parameters
CMG_fold_avg = mean(cvAccuracy3);
[maxValue, maxIndex] = max(cvAccuracy3);
disp("Max Gaussian Kernel");
disp(maxValue * 100)
disp("Max Gaussian Kernel");
disp(CMG_fold_avg * 100);
bestClassifiers(4, 1)=maxIndex;
bestClassifiers(4, 2)=maxValue;

% cc.Title=strcat('confusion matrix for 2 classes - Model ', sensor_labels{sensor});

%% final classification on all offline data
[maxValue, maxIndex] = max(bestClassifiers(:,2));
disp(maxIndex);
%example on SVM
trainLabels = reshape(FEATUREL, [], 1);
trainFeatures = reshape(DATA, [], numFeatures);
%final_model = fitcecoc(trainFeatures, trainLabels, 'Learners', t, 'Coding', 'onevsone');
% final_model = fitcdiscr(trainFeatures, trainLabels, 'DiscrimType', 'quadratic');
final_model = fitckernel(trainFeatures, trainLabels, 'ScoreTransform', 'doublelogit');

%% assess on session 2, 3 
% session 2

finalCVaccuracy = zeros(6, 2);
online = Subject3.online.session2;
dataLen = 7;
online_k = 4;
% features = 1000000;
% for i=1:online_k
%     if length(online_feats{i}) < features
%         features = length(online_feats{i});
%     end
% end

TESTDATA = cell(online_k, 1);
for i=1:online_k
    data=[];
    for j=1:20
        data = [data; online_feats{i,1}{j}];
    end
    TESTDATA{i} = data;
end

% Get the size of each double array and find the minimum size
% sizes = cellfun(@size, TESTDATA, 'UniformOutput', false);
% minSize = min(cat(1, sizes{:}), [], 1);
% 
% % Reshape each double array to the minimum size
% truncatedArrays = cellfun(@(x) x(1:minSize(1), 1:minSize(2), :), TESTDATA, 'UniformOutput', false);
% 
% % Convert the cell array to a 4x5xn double array
% result = cat(3, truncatedArrays{:});
% TESTDATA=permute(result, [3, 2, 1]);

TESTFEATUREL=cell(online_k, 1);
for i=1:online_k
    data=[];
    for j=1:20   
        data = [data; online_labels{i, 1}{j, 1}];

        % right = prev + length(online_labels{i, 1}{j, 1}) - 1;
        % if right < prev
        %     break
        % end
        % if right > minSize(1)
        %     right = minSize(1);
        %     TESTFEATUREL(i, prev : right) = online_labels{i, 1}{j, 1}(1: right - prev + 1);
        % 
        % else
        %     TESTFEATUREL(i, prev : right) = online_labels{i, 1}{j, 1};
        % end
        % disp(online_labels{i, 1}{j, 1}(1: right - prev + 1))
        % 
        % prev = prev + length(online_labels{i, 1}{j, 1}) ;
    end
    TESTFEATUREL{i} = data;

end

%% run-wise testing!
for i=1:online_k

    validationLabels = TESTFEATUREL{i};
    validationFeatures = TESTDATA{i};
    [predictedLabels, scores] = predict(final_model, validationFeatures);
    % disp(scores)
    
    
    % Compute accuracy for current fold
    finalCVAccuracy(i, 1) = sum(predictedLabels == validationLabels) / length(validationLabels);
    disp(finalCVAccuracy(i, 1))
    subplot(2, 4, i);
    confusionMatrix = confusionmat(validationLabels-1, predictedLabels-1);
    cc=confusionchart(confusionMatrix, {'Rest','Reach'});
    cc.RowSummary = 'row-normalized';
    cc.Title = "Online Session 1 Confusion Matrices: Subject 3"
    
end

% session 3
online = Subject3.online.session3;
online_k = 6;

TESTDATA = cell(online_k, 1);
for i=1:online_k
    data=[];
    for j=1:20
        data = [data; online_feats2{i,1}{j}];
    end
    TESTDATA{i} = data;
end

TESTFEATUREL=cell(online_k, 1);
for i=1:online_k
    data=[];
    for j=1:20   
        data = [data; online_labels2{i, 1}{j, 1}];
    end
    TESTFEATUREL{i} = data;

end

%% run-wise testing!
for i=1:online_k

    validationLabels = TESTFEATUREL{i};
    validationFeatures = TESTDATA{i};
    [predictedLabels, scores] = predict(final_model, validationFeatures);
    % disp(scores)
    
    
    % Compute accuracy for current fold
    finalCVAccuracy(i, 2) = sum(predictedLabels == validationLabels) / length(validationLabels);
    disp(finalCVAccuracy(i, 2))
    subplot(2, 4, i);
    confusionMatrix = confusionmat(validationLabels-1, predictedLabels-1);
    cc=confusionchart(confusionMatrix, {'Rest','Reach'});
    cc.RowSummary = 'row-normalized';
    cc.Title = "Online Session 2 Confusion Matrices: Subject 3";
    
end


%% AUC 
[X,Y,T,AUC] = perfcurve(validationLabels, predictedLabels, 1);
% perfcurve() calculates the true positive rate (X), false positive rate (Y), thresholds (T), and AUC

% Plot ROC curve
plot(X,Y)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('Receiver Operating Characteristic (ROC) Curve')
grid on

% hold on
% fill(X, Y, 'b', 'FaceAlpha', 0.2) % Fill area under the curve with blue and 20% opacity

% Display AUC
fprintf('Area Under the Curve (AUC): %.4f\n', AUC);
