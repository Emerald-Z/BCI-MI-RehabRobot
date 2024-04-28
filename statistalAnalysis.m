%% statistical analysis of results
% per sample - assume quadratic

%% session 2
sample_probabilities = cell(4, 20, 1);
for i=1:4
    idx = 1;
    for j=1:20
        len = length(online_labels{i, 1}{j, 1});
        sampleFeatures = online_feats{i, 1}(idx : idx + len - 1, :);
        % sampleFeatures
        [labels, scores] = predict(final_model, sampleFeatures); % scores: rest, reach
        disp(scores)
        sample_probabilities{i, j} = scores;
        idx = idx + len;
    end
end


%% evidence accumulation
alpha = 0.7;
Pt = cell(4, 20, 1);
for i=1:4
    for j=1:20
        class = session2.labels.type{1, i}(j);
        temp = zeros(size(sample_probabilities{i, j}));
        if length(sample_probabilities{i, j}) == 0
            continue
        end
        shape = size(sample_probabilities{i, j});
        for t = 1:shape(1)
            if t == 1
                temp(t, :) = sample_probabilities{i, j}(t, :); % For the first sample, use the original probability distribution
            else
                temp(t, :) = alpha * temp(t-1, :) + (1 - alpha) * sample_probabilities{i, j}(t, :); % Apply exponential smoothing
            end
        end
        Pt{i, j} = temp;

    end
end

% horizontal pars at thresholds = .55, .35 rest
% decision if sustained or not
% see if the plots are underneath the lines
%do it by run, and trial
% 0-160 time steps???
% how do you do it per sample
reach_thresh = 0.5;
rest_thresh = 0.5;
X = 1:length(Pt{1});
figure;
% plot(Pt{1}, 'r.')
scatter(X, Pt{1}, 20, 'filled');  % 100 is marker size, 'filled' fills markers
hold on
plot([1, length(Pt{1, 1})], [reach_thresh, reach_thresh], 'g--'); % 'r--' specifies red dashed line
hold on
plot([1, length(Pt{1, 1})], [rest_thresh, rest_thresh], 'o--'); % 'r--' specifies red dashed line

xlabel('X');
ylabel('Y');
title('Evidence Accumulation: Run 1')
legend({'Rest', 'Reach'}, 'Location', 'best');

% display random 4 from each session to get threshold
trials = randi([1, 10], 1, 4);

% for each session, go through each with the threshold and get an accuracy
% - trial based
% session 2
trial_results = zeros(4, 20, 2);
correct=zeros(4, 4); % rest, reach
for run=1:4
    for trial=1:20
        if length(Pt{run, trial}) == 0
            continue;
        end
        reach = Pt{run, trial}(end, 2) >= reach_thresh; %any();
        rest = Pt{run, trial}(end, 1) >= rest_thresh;
        trial_results(run, trial, 1) = reach; % if 1, it doesn't pass
        trial_results(run, trial, 2) = rest;           
        
        % ma
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

    end
    disp(correct/10);
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
            sample_results(i, j) = sum(Pt{run, trial}(:, label) >= rest_thresh) ...
                / length(Pt{run, trial}(:, label));
        else 
            sample_results(i, j) = sum(Pt{run, trial}(:, label) >= reach_thresh) ...
                / length(Pt{run, trial}(:, label));
        end
    end
end
%% ttest
%% randomness