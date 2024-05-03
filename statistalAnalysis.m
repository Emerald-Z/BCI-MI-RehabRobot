%% statistical analysis of results
% per sample - assume quadratic

%% session 2
sample_probabilities = cell(4, 20, 1);
for i=1:4
    idx = 1;
    for j=1:20
        len = length(online_labels{i, 1}{j, 1});
        sampleFeatures = online_feats{i, 1}{j, 1}; %(idx : idx + len - 1, :);
        % sampleFeatures
        [labels, scores] = predict(final_model, sampleFeatures); % scores: rest, reach
        disp(scores)
        sample_probabilities{i, j} = scores;
        idx = idx + len;
    end
end

sample_probabilities2 = cell(4, 20, 1);
for i=1:4
    idx = 1;
    for j=1:20
        len = length(online_labels2{i, 1}{j, 1});
        sampleFeatures = online_feats2{i, 1}{j, 1}; %(idx : idx + len - 1, :);
        % sampleFeatures
        [labels, scores] = predict(final_model, sampleFeatures); % scores: rest, reach
        disp(scores)
        sample_probabilities2{i, j} = scores;
        idx = idx + len;
    end
end

titles=["Rest", "Reach"];


%% evidence accumulation
alpha = .7;
Pt2 = cell(4, 20, 1);
for i=1:4
    for j=1:20
        class = session3.labels.type{1, i}(j);
        temp = zeros(size(sample_probabilities2{i, j}));
        if length(sample_probabilities2{i, j}) == 0
            continue
        end
        shape = size(sample_probabilities2{i, j});
        for t = 1:shape(1)
            if t == 1
                temp(t, :) = sample_probabilities2{i, j}(t, :); % For the first sample, use the original probability distribution
                %temp(t, :) = [0.5, 0.5]; % Initialize first sample to 50% probability for both classes
            else
                temp(t, :) = alpha * temp(t-1, :) + (1 - alpha) * sample_probabilities2{i, j}(t, :); % Apply exponential smoothing
            end
        end
        Pt2{i, j} = temp;

    end
end

% display random 4 from each session to get threshold
% trials = randi([1, 20], 1, 6);
figure;
titles=["Rest", "Reach"];
for i=1:20
    subplot(4, 5, i);
    X = 1:length(Pt{1, i});
    scatter(X, Pt{1, i}, 12, 'filled');  % 100 is marker size, 'filled' fills markers
    % hold on
    % plot([1, length(Pt{1, 1})], [reach_thresh, reach_thresh], 'g--'); % 'r--' specifies red dashed line
    % hold on
    % plot([1, length(Pt{1, 1})], [rest_thresh, rest_thresh], 'o--'); % 'r--' specifies red dashed line
    % 
    xlabel('Sample Point');
    ylabel('Probability');
    title(strcat('Run 1 Trial ', num2str(i), titles(online_labels{1, 1}{i, 1}(1))))
    legend({'Rest', 'Reach'}, 'Location', 'best');
    % titles(online_labels{1, 1}{i, 1})
end
% horizontal pars at thresholds = .55, .35 rest
% decision if sustained or not
% see if the plots are underneath the lines
%do it by run, and trial
% 0-160 time steps???
% how do you do it per sample
reach_thresh = 0.55;
rest_thresh = 0.75;
trials = randi([1, 20], 1, 4);

figure;
for i=1:4
    subplot(2, 2, i);
    X = 1:length(Pt{1, trials(i)});
    % plot(Pt{1}, 'r.')
    scatter(X, Pt{1, trials(i)}, 12, 'filled');  % 100 is marker size, 'filled' fills markers
    hold on
    plot([1, length(Pt{1, trials(i)})], [reach_thresh, reach_thresh], 'g--'); % 'r--' specifies red dashed line
    hold on
    plot([1, length(Pt{1, trials(i)})], [rest_thresh, rest_thresh], 'o--'); % 'r--' specifies red dashed line
    
    xlabel('Sample Point');
    ylabel('Probability');
    title(strcat('Evidence Accumulation: Run 1 Trial ', num2str(trials(i))))
    legend({'Rest', 'Reach'}, 'Location', 'best');
end


% for each session, go through each with the threshold and get an accuracy
% - trial based
% session 2
trial_results = zeros(4, 20, 2);
correct=zeros(4, 4); % rest, reach
thresholded_correct = zeros(4, 2);
for run=1:4
    for trial=1:20
        if length(Pt{run, trial}) == 0
            continue;
        end
        reach = Pt{run, trial}(end, 2) >= reach_thresh; %any();
        rest = Pt{run, trial}(end, 1) >= rest_thresh;
        trial_results(run, trial, 1) = reach; % if 1, it doesn't pass
        trial_results(run, trial, 2) = rest;           
        
        % final timestep
        if session2.labels.type{1, run}(trial) == 1 && rest
            correct(run, 1) = correct(run, 1) + 1;
        end

        if session2.labels.type{1, run}(trial) == 2 && reach
            correct(run, 2) = correct(run, 2) + 1;
        end

        % average
        if session2.labels.type{1, run}(trial) == 1 && ...
                mean(Pt{run, trial}(:, 1)) >= rest_thresh
            correct(run, 3) = correct(run, 3) + 1;
        end
        if session2.labels.type{1, run}(trial) == 2 && ...
                mean(Pt{run, trial}(:, 2)) >= reach_thresh
            correct(run, 4) = correct(run, 4) + 1;
        end
        
        first_rest=0;
        rest_flag=0;
        first_reach=0;
        reach_flag=0;
        for t=1:length(Pt{run, trial}(:, 2))
        % at any point
            if Pt{run, trial}(t, 1) >= rest_thresh && ~rest_flag
                first_rest = Pt{run, trial}(t, 1);
                rest_flag = 1;
            end
            if Pt{run, trial}(t, 2) >= reach_thresh && ~reach_flag
                first_reach = Pt{run, trial}(t, 2);
                reach_flag = 1;
            end
        end
        if session2.labels.type{1, run}(trial) == 1 && ...
                first_rest <= first_reach
            thresholded_correct(run,1) = thresholded_correct(run,1) + 1;
        end
        if session2.labels.type{1, run}(trial) == 2 && ...
                first_reach <= first_rest
            %thresholded_correct(run, 4) = thresholded_correct(run, 4) + 1;
            thresholded_correct(run, 2) = thresholded_correct(run, 2) + 1;
        end

    end
    % disp(correct/10);
end

% session 3
trial_results = zeros(4, 20, 2);
correct=zeros(4, 4); % rest, reach
thresholded_correct2 = zeros(4, 2);
for run=1:4
    for trial=1:20
        if length(Pt{run, trial}) == 0
            continue;
        end
        reach = Pt{run, trial}(end, 2) >= reach_thresh; %any();
        rest = Pt{run, trial}(end, 1) >= rest_thresh;
        trial_results(run, trial, 1) = reach; % if 1, it doesn't pass
        trial_results(run, trial, 2) = rest;           
        
        % final timestep
        if session3.labels.type{1, run}(trial) == 1 && rest
            correct(run, 1) = correct(run, 1) + 1;
        end

        if session3.labels.type{1, run}(trial) == 2 && reach
            correct(run, 2) = correct(run, 2) + 1;
        end

        % average
        if session3.labels.type{1, run}(trial) == 1 && ...
                mean(Pt{run, trial}(:, 1)) >= rest_thresh
            correct(run, 3) = correct(run, 3) + 1;
        end
        if session3.labels.type{1, run}(trial) == 2 && ...
                mean(Pt{run, trial}(:, 2)) >= reach_thresh
            correct(run, 4) = correct(run, 4) + 1;
        end
        
        first_rest=0;
        rest_flag=0;
        first_reach=0;
        reach_flag=0;
        for t=1:length(Pt{run, trial}(:, 2))
        % at any point
            if Pt{run, trial}(t, 1) >= rest_thresh && ~rest_flag
                first_rest = Pt{run, trial}(t, 1);
                rest_flag = 1;
            end
            if Pt{run, trial}(t, 2) >= reach_thresh && ~reach_flag
                first_reach = Pt{run, trial}(t, 2);
                reach_flag = 1;
            end
        end
        if session3.labels.type{1, run}(trial) == 1 && ...
                first_rest <= first_reach
            thresholded_correct2(run,1) = thresholded_correct2(run,1) + 1;
        end
        if session3.labels.type{1, run}(trial) == 2 && ...
                first_reach <= first_rest
            %thresholded_correct(run, 4) = thresholded_correct(run, 4) + 1;
            thresholded_correct2(run, 2) = thresholded_correct2(run, 2) + 1;
        end

    end
    % disp(correct/10);
end

% sample based accuracy per trial - for each data point, see if it meets the
% threshold (greater or equal to)
sample_results = zeros(4, 20, 1);
for run=1:4
    for trial=1:20
        if length(Pt{run, trial}) == 0
            continue;
        end
        label = session2.labels.type{1, run}(trial);
        if label == 1
            sample_results(run, trial) = sum(Pt{run, trial}(:, label) >= rest_thresh) ...
                / length(Pt{run, trial}(:, label));
        else 
            sample_results(run, trial) = sum(Pt{run, trial}(:, label) >= reach_thresh) ...
                / length(Pt{run, trial}(:, label));
        end
    end
end

sample_results2 = zeros(4, 20, 1);
for run=1:4
    for trial=1:20
        if length(Pt2{run, trial}) == 0
            continue;
        end
        label = session3.labels.type{1, run}(trial);
        if label == 1
            sample_results2(run, trial) = sum(Pt2{run, trial}(:, label) >= rest_thresh) ...
                / length(Pt2{run, trial}(:, label));
        else 
            sample_results2(run, trial) = sum(Pt2{run, trial}(:, label) >= reach_thresh) ...
                / length(Pt2{run, trial}(:, label));
        end
    end
end

%% ttest
for i=1:4
    [h, p,ci,stats] = ttest(sample_results(i, :), sample_results2(i, :));
    disp(strcat("h: ", num2str(h), "p: ", num2str(p)))
end

online1 = sum(thresholded_correct, 2)/20;
online2 = sum(thresholded_correct2, 2)/20;

[h, p,ci,stats] = ttest(finalCVAccuracy(1:4, 1), finalCVAccuracy(1:4, 2));
