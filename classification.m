%% what is our train/test split? 
% in the guide we build classifier on offline data and test it on online
% test linear, quad, SVM

%% build classifier using offline data and test with run wise cross val -> independence
% Linear - k - 1 fold, test on last run of subject data
k = 5; % num runs
classificationErrors = zeros(k, 1);
CM_fold_avg=0
CM_test_avg=0;
MAVF=MAV_features_1;
VARF=VAR_features_1;
FEATUREL=featureLabels_1;
bestClassifiers=zeros(3,3); % the index of the classifier and its accuracy, test acc

%labels
MAVVAR_Data_All_classes=[run1_r(1:end-1) run2_r run3_r run4_r run5_r; 
    vrun1_r(1:end-1) vrun2_r vrun3_r vrun4_r vrun5_r];
MAVVAR_Labels_All_classes=[];
for i=1:150
    MAVVAR_Labels_All_classes=[MAVVAR_Labels_All_classes FEATUREL(i)*ones(1,numFeatures)];
end
run6_labels = [];
for i=151:180
    run6_labels=[run6_labels FEATUREL(i)*ones(1, numFeatures)];
end

CC_avg=zeros(3,3,1,10); 
CM=zeros(3,3,sensor);
MAV_ALL_acc=zeros(1, 5);
for i=1:k-1
    % linear classifier
    indices=(i-1)*(numFeatures*30)+1:i*(numFeatures*30);
    remaining_indices = setdiff(1:150*numFeatures, indices);

    % linear
    [TstMAVVARFALL TstMAVVARErrALL] = classify(MAVVAR_Data_All_classes(:,indices)',MAVVAR_Data_All_classes(:,remaining_indices)',MAVVAR_Labels_All_classes(remaining_indices));
    [CC_avg(:,:,1,i) dum1 MAV_ALL_acc(i) dum2] = confusion(MAVVAR_Labels_All_classes(indices), TstMAVVARFALL);
    % TstAcc_MAVVAR_All_stor(i) = TstAcc_MAVVAR_ALL;
    CM(:,:,1) = CM(:,:,1) + confusionmat(MAVVAR_Labels_All_classes(indices), TstMAVVARFALL');
end
CM_fold_avg=sum(MAV_ALL_acc)/5;
[maxValue, maxIndex] = max(MAV_ALL_acc);
disp(maxValue)
bestClassifiers(1, 1)=maxIndex;
bestClassifiers(1, 2)=maxValue;
% test on 6th run

indices=(maxIndex-1)*(numFeatures*30)+1:maxIndex*(numFeatures*30);
remaining_indices = setdiff(1:150*numFeatures, indices);

[MAVVARLabels MAVVARErr] = classify([run6_r; vrun6_r]', MAVVAR_Data_All_classes(:,remaining_indices)',MAVVAR_Labels_All_classes(remaining_indices));
[TstCM_MAVVAR_ALL dum1 CM_test_avg dum2] = confusion(run6_labels, TstMAVVARFALL);
bestClassifiers(1, 3)=CM_test_avg;

% 
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
for i = 1:k-1
    % Split data into training and validation sets for current fold
    indices=(i-1)*(numFeatures * 30)+1:i*(numFeatures * 30);
    remaining_indices = setdiff(1:150*numFeatures, indices);
    cvTrainIdx = remaining_indices;
    cvTrainFeatures = MAVVAR_Data_All_classes1(:, remaining_indices);
    cvTrainLabels = MAVVAR_Labels_All_classes1(remaining_indices);
    cvValidationFeatures = MAVVAR_Data_All_classes1(:, indices);
    cvValidationLabels = MAVVAR_Labels_All_classes1(indices);
    
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

% Test the classifier on remaining runs (runs 6 to end)
testingFeatures = [run6_r; vrun6_r]';
testingLabels = run6_labels;

% Predict on testing set
predictedLabels = predict(classifiers1{maxIndex}, testingFeatures);

% Compute accuracy on testing set
accuracy = sum(predictedLabels' == testingLabels) / length(run6_labels);
bestClassifiers(2, 3)=accuracy;
% Display test accuracy
disp(['Subject ', num2str(1), ' - Test Accuracy: ', num2str(accuracy)]);

% Plot confusion matrix
% confusionMatrix = confusionmat(testingLabels, predictedLabels);
% 
% subplot(2, 2, sensor);
% cc=confusionchart(confusionMatrix, {'Pinch','Point','Grasp'});
% cc.RowSummary = 'row-normalized';
% cc.Title=strcat('confusion matrix for MAV and VAR features for all 3 classes - Sensor ', sensor_labels{sensor});

% SVM

ldaModel = fitcdiscr(trainingData, trainingLabels, 'DiscrimType', 'linear');
svmModel = fitcecoc(trainingData, trainingLabels, 'Learners', t, 'Coding', 'onevsone');

% Cross-validation
cvLDA = crossval(ldaModel, 'KFold', numTrainingRuns);
ldaCVError = kfoldLoss(cvLDA);
svmCVModel = crossval(svmModel, 'KFold', numTrainingRuns);
svmCVError = kfoldLoss(svmCVModel);

fprintf('Linear model CV accuracy, subject 1: %f\n', 1-ldaCVError); 
fprintf('SVM CV accuracy: %f\n', 1-svmCVError);

% Choose the best model based on cross-validation error
if ldaCVError < svmCVError
    subj1_bestModel{sensor} = ldaModel;
else
    subj1_bestModel{sensor} = svmModel;
end

%% final classification on all offline data


%% assess on session 2, 3 at sample and trial level (evidence accumulation framework)


