%% epoching is done here too

%% offline sesion

% filtering - bpf
fc1 = 20; % first cutoff frequency in Hz 
fc2 = 38; % I got this from PSD
fs = 512;

% normalize the frequencies
%Wp = [fc1 fc2]*2/fs;
Wp = [fc1 fc2]/(fs/2);



% Specify which subject we're running on 
current_subject = 1;
subject_id = sprintf('Subject%d', current_subject);  % Dynamic subject ID based on current_subject
% How many runs per online session?
session2_run_num = 4; % Subj1: 4, Subj2: 3,
session3_run_num = 6; % Subj1: 6, Subj2: 6,

% Build a Butterworth bandpass filter of 4th order
% check the "butter" function in matlab
%N=4;
%[b,a]=butter(N, Wp, 'bandpass');
d = designfilt('bandpassiir', 'FilterOrder', 4, 'HalfPowerFrequency1', fc1, 'HalfPowerFrequency2', fc2, 'SampleRate', fs);


% artefact rejection - need to display the data to do this

% EOG removal - should we just remove it?

% split run data into task periods
offline = struct('rest', {cell(2, 1)}, 'runs', {cell(3, 1)});
runs = struct('eeg', {cell(3, 1)}, 'labels', {cell(3, 1)});


% Create logical index to select rows to keep
%rows_to_keep = setdiff(1:71, [13, 14, 18, 19, 65:71]);
rows_to_keep = setdiff(1:71, [13, 14, 18, 19, 54, 57, 65:67, 71]); % keep the EOG

% rest data processing
offline_rest_path = sprintf("%s/Offline/%s_Offline_s1Rest", subject_id, subject_id);
subject = load(offline_rest_path);
run = subject.(sprintf("%s_Offline_s1Rest", subject_id));

% Apply filtering to selected channels only
%run1 = filtfilt(d, run.signal(:, rows_to_keep)); %filtfilt(d,filtfilt(b, a, run.signal));
run1 = run.signal(:, rows_to_keep); % For calculating PSD

open = find(run.header.EVENT.TYP == 10);
close = find(run.header.EVENT.TYP == 11);
finish = find(run.header.EVENT.TYP == 55555);

% Extract data for open and close events from selected channels only
offline.rest{1} = run1(run.header.EVENT.POS(open) : run.header.EVENT.POS(close), :);% 10
offline.rest{2} = run1(run.header.EVENT.POS(close) : run.header.EVENT.POS(finish(2)), :);

% offline data
valuesStart=[7691, 7701];
valuesEnd=[7692, 7702];


%runs (task data)
for i=1:3
    labels = zeros(20, 1);
    data = cell(20, 1);
    runs.labels{i} = labels;
    runs.eeg{i} = data;

    % test
    offline_run_path = sprintf('%s/Offline/%s_Offline_s1r%d', subject_id, subject_id, i);
    subject = load(offline_run_path);
    run = subject.(sprintf('%s_Offline_s1r%d', subject_id, i));


    %run1= filtfilt(d, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
    run1 = run.signal;  % For calculating PSD
    run1_h_et=run.header.EVENT.TYP;
    run1_h_ep=run.header.EVENT.POS;

    run1_trials=[run1_h_ep(ismember(run1_h_et, valuesStart)) run1_h_ep(ismember(run1_h_et, valuesEnd))];
    r1_trialtype=[run1_h_et(ismember(run1_h_et, valuesStart)) run1_h_et(ismember(run1_h_et, valuesEnd))];
    
    for j=1:20
        % add samples
        runs.eeg{i}{j} = run1(run1_trials(j,1):run1_trials(j,2), rows_to_keep);
        % append task label
        if (floor(mod(r1_trialtype(j,1)/100, 10)) == 6)
            runs.labels{i}(j) = 1; % 6-rest 7-reach
        else
            runs.labels{i}(j) = 2; % 21 reach 
        end


    end

end

offline.runs = runs;

channel_labels = run.header.Label(rows_to_keep);


%% online session
valuesStart=[7691, 7701];
valuesEnd=[7692, 7693, 7702, 7703];
valuesSustain=[102, 103, 202, 203];

online = struct('session2', struct('eeg', {cell(4, 1)}, 'labels', struct('type', {cell(4, 1)}, 'end', {cell(4, 1)}, 'sustain', {cell(4, 1)})) ...
    , 'session3', struct('rest', {cell(2, 1)}, 'eeg', {cell(6, 1)}, 'labels', struct('type', {cell(6, 1)}, 'end', {cell(6, 1)}, 'sustain', {cell(6, 1)})));
% session2.labels = struct('type', {cell(4, 1)}, 'end', {cell(4, 1)}, 'sustain', {cell(4, 1)});
online = struct('session2', struct('eeg', {cell(session2_run_num, 1)}, 'labels', struct('type', {cell(session2_run_num, 1)}, 'end', {cell(session2_run_num, 1)}, 'sustain', {cell(session2_run_num, 1)})) ...
    , 'session3', struct('rest', {cell(2, 1)}, 'eeg', {cell(session3_run_num, 1)}, 'labels', struct('type', {cell(session3_run_num, 1)}, 'end', {cell(session3_run_num, 1)}, 'sustain', {cell(session3_run_num, 1)})));

% Session 2 (first online)
for i=1:session2_run_num
    types = zeros(20, 1);
    ends = zeros(20, 1);
    sustains = zeros(20, 1);
    sustain_idxs = zeros(20, 1);

    data = cell(20, 1);

    session2.labels.type{i} = types;
    session2.labels.end{i} = ends;
    session2.labels.sustain{i} = sustains;
    session2.labels.sustain_idx{i} = sustain_idxs;

    session2.eeg{i} = data;

    % test
    online_run_path = sprintf('%s/Online/%s_Online_s2r%d', subject_id, subject_id, i);
    subject = load(online_run_path);
    run = subject.(sprintf('%s_Online_s2r%d', subject_id, i));


    %run1= filtfilt(d, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
    run1 = run.signal;  % For calculating PSD
    run1_h_et=run.header.EVENT.TYP;
    run1_h_ep=run.header.EVENT.POS;

    run1_trials=[run1_h_ep(ismember(run1_h_et, valuesStart)) run1_h_ep(ismember(run1_h_et, valuesEnd))];
    r1_trialtype=[run1_h_et(ismember(run1_h_et, valuesStart)) run1_h_et(ismember(run1_h_et, valuesEnd))];
    run1_sustain=[run1_h_ep(ismember(run1_h_et, valuesSustain))];
    r1_sustaintype=[run1_h_et(ismember(run1_h_et, valuesSustain))];

    sustain_index = 1;
    for j=1:20
        % append task label
        % floor(r1_trialtype(j,1)/100)
        if (floor(r1_trialtype(j,1)/100) == 76)
            session2.labels.type{i}(j) = 1; % 76-rest 77-reach
        else
            session2.labels.type{i}(j) = 2;
        end

        % was the initial delivery correct? 
        if (mod(r1_trialtype(j, 2), 10) == 2)
            session2.labels.end{i}(j) = 1; % incorrect delivery
        else
            session2.labels.end{i}(j) = 2; % correct delivery
        end
        

        % Was delivery correct? Update sustain
        if (session2.labels.end{i}(j)== 1)
            % fail (sustain never reached)
            session2.labels.sustain{i}(j) = 0;
            session2.labels.sustain_idx{i}(j) = -1;
            % add samples
            session2.eeg{i}{j} = run1(run1_trials(j,1):run1_trials(j,2), rows_to_keep);
        else
            session2.labels.sustain_idx{i}(j) = run1_sustain(sustain_index) - run1_trials(j,1); % How long until sustain?

            % process sustain
            if (mod(r1_sustaintype(sustain_index), 100) == 2)
                session2.labels.sustain{i}(j) = 1; % failed to sustain
            else 
                session2.labels.sustain{i}(j) = 2; % sustain succeeded
            end
            session2.eeg{i}{j} = run1(run1_trials(j,1) : run1_sustain(sustain_index), rows_to_keep);
            sustain_index = sustain_index + 1;
        end
    end


end

online.session2 = session2;

% Session 3 (second online)
for i=1:session3_run_num % may have to manually change how many runs there were
    types = zeros(20, 1);
    ends = zeros(20, 1);
    sustains = zeros(20, 1);
    sustain_idxs = zeros(20, 1);
    data = cell(20, 1);

    session3.labels.type{i} = types;
    session3.labels.end{i} = ends;
    session3.labels.sustain{i} = sustains;
    session3.labels.sustain_idx{i} = sustain_idxs;


    session3.eeg{i} = data;

    % test
    online_run_path = sprintf('%s/Online/%s_Online_s3r%d', subject_id, subject_id, i);
    subject = load(online_run_path);
    run = subject.(sprintf('%s_Online_s3r%d', subject_id, i));


    %run1= filtfilt(d, run.signal); %filtfilt(d,filtfilt(b, a, run.signal));
    run1 = run.signal;  % For calculating PSD
    run1_h_et=run.header.EVENT.TYP;
    run1_h_ep=run.header.EVENT.POS;

    run1_trials=[run1_h_ep(ismember(run1_h_et, valuesStart)) run1_h_ep(ismember(run1_h_et, valuesEnd))];
    r1_trialtype=[run1_h_et(ismember(run1_h_et, valuesStart)) run1_h_et(ismember(run1_h_et, valuesEnd))];
    run1_sustain=[run1_h_ep(ismember(run1_h_et, valuesSustain))];
    r1_sustaintype=[run1_h_et(ismember(run1_h_et, valuesSustain))];

    sustain_index = 1;
    for j=1:20
        % append task label (rest vs reach)
        % floor(r1_trialtype(j,1)/100)
        if (floor(r1_trialtype(j,1)/100) == 76)
            % rest
            session3.labels.type{i}(j) = 1; % 76-rest 77-reach
        else
            % reach
            session3.labels.type{i}(j) = 2;
        end

        % was the initial command delivery correct? 
        if (mod(r1_trialtype(j, 2), 10) == 2)
            session3.labels.end{i}(j) = 1; % incorrect delivery
        else
            session3.labels.end{i}(j) = 2; % correct delivery
        end
        

        % Was delivery correct? Update sustain
        if (session3.labels.end{i}(j) == 1)
            % fail (sustain never reached)
            session3.labels.sustain{i}(j) = 0;
            session3.labels.sustain_idx{i}(j) = -1;

            % add samples
            session3.eeg{i}{j} = run1(run1_trials(j,1):run1_trials(j,2), rows_to_keep);
        else
            session3.labels.sustain_idx{i}(j) = run1_sustain(sustain_index) - run1_trials(j,1); % How long until sustain?
            % process sustain
            if (mod(r1_sustaintype(sustain_index), 100) == 2)
                session3.labels.sustain{i}(j) = 1; % failed to sustain
            else 
                session3.labels.sustain{i}(j) = 2; % sustain succeeded
            end
            session3.eeg{i}{j} = run1(run1_trials(j,1) : run1_sustain(sustain_index), rows_to_keep);
            sustain_index = sustain_index + 1;
        end
    end


end

% rest data processing for session 3
session3.rest = cell(2, 1);

online_rest_path = sprintf("%s/Online/%s_Online_s3Rest", subject_id, subject_id);
subject = load(online_rest_path);
run = subject.(sprintf("%s_Online_s3Rest", subject_id));

% Apply filtering to selected channels only
%run1 = filtfilt(d, run.signal(:, rows_to_keep)); %filtfilt(d,filtfilt(b, a, run.signal));
run1 = run.signal(:, rows_to_keep);  % For calculating PSD

open = find(run.header.EVENT.TYP == 10);
close = find(run.header.EVENT.TYP == 11);
finish = find(run.header.EVENT.TYP == 55555);

% Extract data for open and close events from selected channels only
session3.rest{1} = run1(run.header.EVENT.POS(open) : run.header.EVENT.POS(close), :);% 10
session3.rest{2} = run1(run.header.EVENT.POS(close) : run.header.EVENT.POS(finish(2)), :);


online.session3 = session3;

disp(sprintf("Data for Subject %d loaded", current_subject));

%% Fisher scores

num_channels = size(offline.rest{1}, 2);  % Total number of channels

non_eog_channels = setdiff(1:num_channels, [28, 59:61]);
non_eog_labels = channel_labels(non_eog_channels);

% Initialize rest and task data cells for each channel
data_rest = cell(length(non_eog_channels), 1);
data_task = cell(length(non_eog_channels), 1);

for i = 1:length(non_eog_channels)
    % Store rest data for the current channel
    data_rest{i} = offline.rest{1}(:, non_eog_channels(i));
    
    % Initialize an empty cell array to store task data for each run
    task_data_runs = cell(length(offline.runs.eeg), 1);
    
    for j = 1:3
        % Initialize an empty cell array to store task data for each trial within the run
        task_data_trials = cell(length(offline.runs.eeg{j}), 1);
        
        for k = 1:length(offline.runs.eeg{j})
            % Store task data for each trial within the run
            task_data_trials{k} = offline.runs.eeg{j}{k}(:, non_eog_channels(i));
        end
        
        % Average task data across trials within the run
        task_data_runs{j} = mean(cell2mat(task_data_trials), 2);
    end
    
    % Average task data across runs for the current channel
    data_task{i} = mean(cell2mat(task_data_runs), 2);
end

% Calculate Fisher scores for each channel
fisher_scores = zeros(length(non_eog_channels), 1);
for i = 1:length(non_eog_channels)
    mean_rest = mean(data_rest{i});
    mean_task = mean(data_task{i});
    var_rest = var(data_rest{i});
    var_task = var(data_task{i});
    fisher_scores(i) = (mean_task - mean_rest)^2 / (var_task + var_rest);
end

% Select the N most discriminable channels
N = 10; % Number of channels to select
[~, selected_channels] = maxk(fisher_scores, N);
selected_channel_labels = non_eog_labels(selected_channels);

% Plot selected channels
figure;
bar(fisher_scores(selected_channels));
xlabel('Selected Channel');
ylabel('Fisher Score');
title('Top 10 Fisher Scores and Associated Channels');
xticks(1:N);
xticklabels(selected_channel_labels);
xtickangle(45);
grid on;

disp(sprintf("Fisher scores calculated"));

%{
figure;

% Create a vector of zeros for all channels

% Set the values of selected channels to their respective Fisher scores
topoplot_data = fisher_scores;

chanloc = readlocs(run.header);

topoplot(topoplot_data, chanloc, 'electrodes', 'labelpoint', 'maplimits', 'minmax');

colorbar;
title('Fisher Scores of Selected Channels'); 
%}

%% PSD

% Do psd on just the most discrimiable channels? right now there's no good
% data from psd

%% Calculate Average PSDs Across All Channels
% Define PSD parameters
fs = 512;  % Sampling frequency
window = 4 * fs;  % Window size for PSD estimation (4 seconds)
noverlap = 2 * fs;  % Overlap between windows (2 seconds)
nfft = 2^nextpow2(window);  % Number of FFT points

% Calculate minimum samples across runs for each reach trial
min_samples = [inf, inf, inf];
runs = 3;
for run_idx = 1:runs
    for i = 1:length(offline.runs.eeg{run_idx})
        if offline.runs.labels{run_idx}(i) == 2  % Checking if it's a reach trial
            current_length = size(offline.runs.eeg{run_idx}{i}, 1);
            if current_length < min_samples(run_idx)
                min_samples(run_idx) = current_length;
            end
        end
    end
end

% Initialize PSD accumulators
psd_rest_total = zeros(nfft/2+1, 1);
psd_run_total = zeros(nfft/2+1, 1);

% Exclude EOG channels
num_channels = size(selected_channels, 1);  % Total number of channels
%non_eog_channels = size(non_eog_channels, 2);

%non_eog_channels = setdiff(1:num_channels, [28, 61:63]);
%num_channels = size(non_eog_channels, 2);  % Total number of channels


%{
% Calculate average PSD for resting state across all channels
for ch = non_eog_channels
    rest_avg = mean(offline.rest{1}(:, ch), 2);
    [psd_ch, freq_rest] = pwelch(rest_avg, window, noverlap, nfft, fs);
    psd_rest_total = psd_rest_total + psd_ch;
end
%}


% Calculate average PSD for "rest" task state across all channels
for ch = non_eog_channels
    for run_idx = 1:runs
        run_samples = arrayfun(@(x) offline.runs.eeg{run_idx}{x}(1:min_samples(run_idx)-1, ch), ...
                               find(offline.runs.labels{run_idx} == 1), 'UniformOutput', false);  % Only rest trials
        if ~isempty(run_samples)
            run_avg = mean(cat(3, run_samples{:}), 3);
            save_run_avg = run_avg;
            [psd_ch, freq_run] = pwelch(run_avg, window, noverlap, nfft, fs);
            psd_rest_total = psd_rest_total + psd_ch / runs;
        end
    end
end


% Calculate average PSD for "reach" task state across all channels
for ch = non_eog_channels
    for run_idx = 1:runs
        run_samples = arrayfun(@(x) offline.runs.eeg{run_idx}{x}(1:min_samples(run_idx)-1, ch), ...
                               find(offline.runs.labels{run_idx} == 2), 'UniformOutput', false);  % Only reach trials
        if ~isempty(run_samples)
            run_avg = mean(cat(3, run_samples{:}), 3);
            [psd_ch, freq_run] = pwelch(run_avg, window, noverlap, nfft, fs);
            psd_run_total = psd_run_total + psd_ch / runs;
        end
    end
end

% Average the PSDs across all channels
psd_rest_avg = (psd_rest_total / num_channels);
psd_run_avg = (psd_run_total / num_channels);

% Convert to dB
psd_rest_db = 10 * log10(psd_rest_avg);
psd_run_db = 10 * log10(psd_run_avg);

% Calculate PSD Difference
psd_diff_db = psd_run_db - psd_rest_db;

% Plot Comparison
figure;
plot(freq_run, abs(psd_diff_db), 'k', 'LineWidth', 2);
hold on;
plot(freq_run, psd_run_db, 'r', 'LineWidth', 2);
plot(freq_run, psd_rest_db, 'b', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('PSD Difference (dB/Hz)');
title('PSD Difference: Reach Task vs Resting State');
grid on;
%legend('Difference', 'Reach', 'Rest');
legend('Reach', 'Rest');
grid on;
xlim([0 50]);  % Limit frequency axis if needed

disp(sprintf("PSD calculated"));

%% Bandpass Filtering

fc1 = 12;
fc2 = 30;

% Design the individualized bandpass filter
Wp = [fc1 fc2] / (fs/2);
[b, a] = butter(4, Wp, 'bandpass');

d = designfilt('bandpassiir', 'FilterOrder', 4, 'HalfPowerFrequency1', fc1, 'HalfPowerFrequency2', fc2, 'SampleRate', fs);


% Plot filtered data for each selected channel
figure;
for i = 2
    %subplot(4, 3, i);
    channel_idx = selected_channels(i);
    % Get the data for the selected channel
    channel_data = offline.runs.eeg{1}{1}(:, channel_idx);
    
    % Filter the data
    filtered_data = filtfilt(d, channel_data);
    
    plot(channel_data - mean(channel_data), 'b', 'LineWidth', 1.5);
    hold on;
    plot(filtered_data, 'r', 'LineWidth', 1.5);
    xlabel('Sample');
    if mod(i, 3) == 1
        ylabel('Amplitude');
    end
    title(sprintf('Channel %d', channel_idx));
    grid on;
end
legend('Before Bandpass', 'After Bandpass');
annotation('textbox', [0.35, 0.95, 0.3, 0.05], 'String', 'Bandpass Filtering', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

% Apply the filter to offline and online data
offline.rest = cellfun(@(x) filtfilt(d, x), offline.rest, 'UniformOutput', false);
offline.runs.eeg = cellfun(@(x) cellfun(@(y) filtfilt(d, y), x, 'UniformOutput', false), offline.runs.eeg, 'UniformOutput', false);
online.session2.eeg = cellfun(@(x) cellfun(@(y) filtfilt(d, y), x, 'UniformOutput', false), online.session2.eeg, 'UniformOutput', false);
online.session3.rest = cellfun(@(x) filtfilt(d, x), online.session3.rest, 'UniformOutput', false);
online.session3.eeg = cellfun(@(x) cellfun(@(y) filtfilt(d, y), x, 'UniformOutput', false), online.session3.eeg, 'UniformOutput', false);

disp(sprintf("Filters applied"));

%% EOG Removal

% Calculate number of samples for N seconds
samples_to_plot = 10 * fs;

% Create the time vector for plotting
time_vector = (0:samples_to_plot - 1) / fs;

% Plot EOG channel
%% NOTE: change this if necessary. this works best for subject 1
eog_channel = 61;  % Adjust the EOG channel index if necessary


% Select the data for plotting (rest)
if size(offline.rest{1}, 1) >= samples_to_plot
    eog_data_to_plot = offline.rest{1}(1:samples_to_plot, eog_channel);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 647');
end

% Plot the EOG data
figure;
plot(time_vector, eog_data_to_plot);
hold on;
grid on;

channel_to_plot = selected_channels(1);
% Select the data for plotting (rest)
if size(offline.rest{1}, 1) >= samples_to_plot
    data_to_plot = offline.rest{1}(1:samples_to_plot, channel_to_plot);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 661');
end

% Plot the channel data
plot(time_vector, data_to_plot);
xlabel('Time (seconds)');
ylabel('Amplitude (\muV)');  % Adjust the units based on your data
title(sprintf('EOG Channel BEFORE EOG FILTER', eog_channel));
legend('EOG', 'Channel');
hold off;


% Remove EOG artifacts using polynomial regression followed by adaptive thresholding
polynomial_order = 3; % Specify the desired polynomial order
threshold_multiplier = 2.5; % Specify the desired threshold multiplier
window_size = 10; % Specify the window size for calculating the mean of surrounding values
offline.rest = cellfun(@(x) remove_eog_artifacts_regression(x, eog_channel, polynomial_order, threshold_multiplier, window_size), offline.rest, 'UniformOutput', false);
offline.runs.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order, threshold_multiplier, window_size), x, 'UniformOutput', false), offline.runs.eeg, 'UniformOutput', false);
online.session2.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order, threshold_multiplier, window_size), x, 'UniformOutput', false), online.session2.eeg, 'UniformOutput', false);
online.session3.rest = cellfun(@(x) remove_eog_artifacts_regression(x, eog_channel, polynomial_order, threshold_multiplier, window_size), online.session3.rest, 'UniformOutput', false);
online.session3.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order, threshold_multiplier, window_size), x, 'UniformOutput', false), online.session3.eeg, 'UniformOutput', false);


% Calculate number of samples for N seconds
samples_to_plot = 10 * fs;

% Create the time vector for plotting
time_vector = (0:samples_to_plot - 1) / fs;

% Select the data for plotting (rest)
if size(offline.rest{1}, 1) >= samples_to_plot
    eog_data_to_plot = offline.rest{1}(1:samples_to_plot, eog_channel);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 695');
end


% Plot the EOG data
figure;
plot(time_vector, eog_data_to_plot);
hold on;
grid on;

% Select the data for plotting (rest)
if size(offline.rest{1}, 1) >= samples_to_plot
    data_to_plot = offline.rest{1}(1:samples_to_plot, channel_to_plot);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 403');
end

% Plot the channel data
plot(time_vector, data_to_plot);
xlabel('Time (seconds)');
ylabel('Amplitude (\muV)');  % Adjust the units based on your data
title(sprintf('EOG Channel AFTER FILTER', eog_channel));
legend('EOG', 'Channel');
hold off;

disp(sprintf("EOG regressed"));

%% Plot eeg channel

channel_to_plot = 1;  % Adjust as necessary

% Calculate number of samples for N seconds
samples_to_plot = 10 * fs; 

%{
% Select the data for plotting
if size(offline.runs.eeg{1}{1}, 1) >= samples_to_plot
    data_to_plot = offline.runs.eeg{1}{1}(1:samples_to_plot, channel_to_plot);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate.');
end
%}

% Select the data for plotting (rest)
if size(offline.rest{1}, 1) >= samples_to_plot
    data_to_plot = offline.rest{1}(1:samples_to_plot, channel_to_plot);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 403');
end

% Create the time vector for plotting
time_vector = (0:samples_to_plot - 1) / fs;

% Plot the data
figure;
plot(time_vector, data_to_plot);
xlabel('Time (seconds)');
ylabel('Amplitude (\muV)');  % Adjust the units based on your data
title(sprintf('EEG Channel %d Over 15 Seconds', channel_to_plot));
grid on;


%% Calculate Mean/Median of Rest Data
% Session 1 Rest Mean
% Only consider open eye rest

s1_rest_mean = mean(offline.rest{1}, 1);

s3_rest_mean = mean(online.session3.rest{1}, 1);


%% Rest Regression

% Plot channels before and after rest regression
figure;
for i = 1:1
    %subplot(4, 3, i);
    channel_idx = selected_channels(i);
    % Get the data for the selected channel
    channel_data = offline.runs.eeg{1}{1}(:, channel_idx);
    
    % Remove rest artifacts using regression
    clean_channel_data = remove_rest_artifacts_regression(channel_data, offline.rest{1});
    
    plot(channel_data, 'b', 'LineWidth', 1.5);
    hold on;
    plot(clean_channel_data, 'r', 'LineWidth', 1.5);
    xlabel('Sample');
    if mod(i, 3) == 1
        ylabel('Amplitude');
    end
    title(sprintf('Channel %d', channel_idx));
    grid on;
end
legend('Before Rest Regression', 'After Rest Regression');
annotation('textbox', [0.35, 0.95, 0.3, 0.05], 'String', 'Rest Regression', 'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

% Perform regression for sessions 1 and 2
offline.runs.eeg = cellfun(@(x) cellfun(@(y) remove_rest_artifacts_regression(y, offline.rest{1}), x, 'UniformOutput', false), offline.runs.eeg, 'UniformOutput', false);
online.session2.eeg = cellfun(@(x) cellfun(@(y) remove_rest_artifacts_regression(y, offline.rest{1}), x, 'UniformOutput', false), online.session2.eeg, 'UniformOutput', false);
% Perform regression for session 3
online.session3.eeg = cellfun(@(x) cellfun(@(y) remove_rest_artifacts_regression(y, online.session3.rest{1}), x, 'UniformOutput', false), online.session3.eeg, 'UniformOutput', false);

disp(sprintf("Rest data regressed"));

%% Plot eeg channel

channel_to_plot = 1;  % Adjust as necessary

% Calculate number of samples for N seconds
samples_to_plot = 10 * fs; 

%{
% Select the data for plotting
if size(offline.runs.eeg{1}{1}, 1) >= samples_to_plot
    data_to_plot = offline.runs.eeg{1}{1}(1:samples_to_plot, channel_to_plot);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate.');
end
%}

% Select the data for plotting (rest)
if size(offline.rest{1}, 1) >= samples_to_plot
    data_to_plot = offline.rest{1}(1:samples_to_plot, channel_to_plot);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 403');
end

% Create the time vector for plotting
time_vector = (0:samples_to_plot - 1) / fs;

% Plot the data
figure;
plot(time_vector, data_to_plot);
xlabel('Time (seconds)');
ylabel('Amplitude (\muV)');  % Adjust the units based on your data
title(sprintf('EEG Channel %d Over 15 Seconds', channel_to_plot));
grid on;


%% make subject struct subject -> offline/online -> rest/active -> labels/eeg

% Isolate the selected channels found with the Fisher score
offline.rest = cellfun(@(x) x(:, selected_channels), offline.rest, 'UniformOutput', false);
offline.runs.eeg = cellfun(@(x) cellfun(@(y) y(:, selected_channels), x, 'UniformOutput', false), offline.runs.eeg, 'UniformOutput', false);
online.session2.eeg = cellfun(@(x) cellfun(@(y) y(:, selected_channels), x, 'UniformOutput', false), online.session2.eeg, 'UniformOutput', false);
online.session3.rest = cellfun(@(x) x(:, selected_channels), online.session3.rest, 'UniformOutput', false);
online.session3.eeg = cellfun(@(x) cellfun(@(y) y(:, selected_channels), x, 'UniformOutput', false), online.session3.eeg, 'UniformOutput', false);


% Subject1 = struct('offline', offline, 'online', online);
eval([subject_id ' = struct(''offline'', offline, ''online'', online);']);


disp(sprintf("Subject %d struct created", current_subject));

%% Functions

function threshold = calculate_eog_threshold(eog_data, multiplier)
    eog_channel = 32;
    threshold = multiplier * std(eog_data(:, eog_channel));
end

function clean_eeg = remove_eog_artifacts_regression(eeg_data, eog_channel, polynomial_order, threshold_multiplier, window_size)
    % Perform EOG artifact removal using polynomial regression and adaptive thresholding

    % Get the number of channels and samples
    num_channels = size(eeg_data, 2);
    num_samples = size(eeg_data, 1);

    % Initialize the cleaned EEG data
    clean_eeg = eeg_data;

    % Perform regression for each EEG channel
    for i = 1:num_channels
        if i ~= eog_channel
            % Create the polynomial terms for regression
            X = zeros(num_samples, polynomial_order + 1);
            for j = 0:polynomial_order
                X(:, j+1) = eeg_data(:, eog_channel) .^ j;
            end

            % Perform polynomial regression between EOG channel and current EEG channel
            beta = X \ eeg_data(:, i);

            % Estimate the EOG artifact using the polynomial regression coefficients
            eog_artifact_poly = X * beta;

            % Remove the EOG artifact estimated by polynomial regression
            eeg_data_poly = eeg_data(:, i) - eog_artifact_poly;

            % Perform linear regression between EOG channel and current EEG channel (after polynomial regression)
            X_linear = [ones(num_samples, 1), eeg_data(:, eog_channel)];
            beta_linear = X_linear \ eeg_data_poly;

            % Estimate the EOG artifact using the linear regression coefficients
            eog_artifact_linear = X_linear * beta_linear;

            % Calculate the residual after linear regression
            residual = eeg_data_poly - eog_artifact_linear;

            % Calculate the adaptive threshold
            threshold = threshold_multiplier * std(residual);

            % Find the indices of values above the threshold
            outlier_indices = find(abs(residual) > threshold);

            % Replace outliers with the mean of surrounding values
            for k = 1:length(outlier_indices)
                idx = outlier_indices(k);
                start_idx = max(1, idx - window_size);
                end_idx = min(num_samples, idx + window_size);
                eeg_data_poly(idx) = mean(eeg_data_poly(start_idx:end_idx));
            end

            clean_eeg(:, i) = eeg_data_poly;

        end
    end
end

function clean_eeg = remove_rest_artifacts_regression(eeg_data, rest_data)
    % Perform rest artifact removal using regression
    % Get the number of channels and samples
    num_channels = size(eeg_data, 2);
    num_samples = size(eeg_data, 1);
    
    % Initialize the cleaned EEG data
    clean_eeg = eeg_data;
    
    % Perform regression for each EEG channel
    for i = 1:num_channels
        % Create the regression matrix using rest data
        X = [ones(num_samples, 1), rest_data(1:size(ones(num_samples, 1)), i)];
    
        % Perform linear regression between rest data and current EEG channel
        beta = X \ eeg_data(:, i);
    
        % Estimate the rest artifact using the regression coefficients
        rest_artifact = X * beta;
    
        % Remove the rest artifact from the EEG channel
        clean_eeg(:, i) = eeg_data(:, i) - rest_artifact;
    end
end