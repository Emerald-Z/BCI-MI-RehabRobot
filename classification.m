%% what is our train/test split? 
% in the guide we build classifier on offline data and test it on online
% test linear, quad, SVM
offline = Subject1.offline.runs;

%% build classifier using offline data and test with run wise cross val -> independence
% Linear - k - 1 fold, test on last run of subject data
dataLen = 1794;
k = 3; % num runs
classificationErrors = zeros(k, 1);
CM_fold_avg=0;
CM_test_avg=0;
FEATUREL=zeros(3, 20 * dataLen);
for i=1:3
    for j=1:20
        FEATUREL(i, (j-1)* dataLen + 1 : j* dataLen) = offline.labels{i, 1}(j,1) * ones(dataLen, 1);
    end
end

bestClassifiers=zeros(3,3); % the index of the classifier and its accuracy, test acc
DATA = zeros(3, 20 *dataLen, 4);
for i=1:3
    DATA(i, :,:) = vertcat(offline_feats{i,1}{:});
end

CC_avg=zeros(2,2,1,k); 
CM=zeros(2,2);
MAV_ALL_acc=zeros(1, 5);
for i=1:k
    % linear classifier
    test = squeeze(DATA(i, :, :));

    other_indices = setdiff(1:size(DATA, 1), i);
    train = vertcat(DATA(other_indices, :, :));
    train = reshape(train, [], 4);
    train_labels = vertcat(FEATUREL(other_indices, :, :));
    train_labels = reshape(train_labels', [], 1);
    test_labels = squeeze(FEATUREL(i, :));
    % linear
    [TstMAVVARFALL TstMAVVARErrALL] = classify(test,train, train_labels);
    [CC_avg(:,:,1,i) dum1 MAV_ALL_acc(i) dum2] = confusion(test_labels, TstMAVVARFALL);
    % TstAcc_MAVVAR_All_stor(i) = TstAcc_MAVVAR_ALL;
    CM(:,:,1) = CM(:,:,1) + confusionmat(test_labels, TstMAVVARFALL');
end
CM_fold_avg=sum(MAV_ALL_acc)/5;
[maxValue, maxIndex] = max(MAV_ALL_acc);
disp(maxValue)
bestClassifiers(1, 1)=maxIndex;
bestClassifiers(1, 2)=maxValue;

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
classifiers1=cell(5,1);
numFeatures = length(MAV_features_1{1,1});

% Perform cross-validation to optimize classifier
for i = 1:k
    % Split data into training and validation sets for current fold
    cvTrainIdx = remaining_indices;
    cvTrainFeatures = [DATA(1:i,:,:) DATA(i:k,:,:)];
    cvTrainLabels = [FEATUREL(1:i,:) FEATUREL(i:k,:)];
    cvValidationFeatures = DATA(i, :,:);
    cvValidationLabels = FEATUREL(i, :);
    
    % Train quadratic LDA classifier
    classifiers1{i} = fitcdiscr(cvTrainFeatures', cvTrainLabels, 'DiscrimType', 'quadratic');
    
    % Predict on validation set
    predictedLabels = predict(classifiers1{i}, cvValidationFeatures');
    
    % Compute accuracy for current fold
    cvAccuracy(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
end

% Average accuracy across folds to find optimal parameters
CMQ_fold_avg = mean(cvAccuracy);
[maxValue, maxIndex] = max(cvAccuracy);
disp(maxValue * 100)
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
cvAccuracy2 = zeros(k, 1); % Array to store cross-validation accuracy
CMS_fold_avg=0;
CMS_test_avg=0;
classifiers2=cell(5,1);

svmModel = fitcecoc(trainingData, trainingLabels, 'Learners', t, 'Coding', 'onevsone');

% Cross-validation
for i = 1:k
    % Split data into training and validation sets for current fold
    cvTrainIdx = remaining_indices;
    cvTrainFeatures = [DATA(1:i,:,:) DATA(i:k,:,:)];
    cvTrainLabels = [FEATUREL(1:i,:) FEATUREL(i:k,:)];
    cvValidationFeatures = DATA(i, :,:);
    cvValidationLabels = FEATUREL(i, :);
    
    % Train quadratic LDA classifier
    classifiers2{i} = fitcecoc(cvTrainFeatures', cvTrainLabels, 'Learners', t, 'Coding', 'onevsone');
    
    % Predict on validation set
    predictedLabels = predict(classifiers2{i}, cvValidationFeatures');
    
    % Compute accuracy for current fold
    cvAccuracy2(i) = sum(predictedLabels == cvValidationLabels') / length(cvValidationLabels);
end

fprintf('Linear model CV accuracy, subject 1: %f\n', 1-ldaCVError); 
fprintf('SVM CV accuracy: %f\n', 1-svmCVError);

CMS_fold_avg = mean(cvAccuracy2);
[maxValue, maxIndex] = max(cvAccuracy2);
disp(maxValue * 100)
bestClassifiers(2, 1)=maxIndex;
bestClassifiers(2, 2)=maxValue;


%% final classification on all offline data
[maxValue, maxIndex] = max(bestClassifiers(2, :));
disp(maxIndex)



%% assess on session 2, 3 at sample and trial level (evidence accumulation framework)


