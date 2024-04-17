%% epoching is done here too

%% offline sesion

% filtering - bpf
fc1 = 40; % first cutoff frequency in Hz 
fc2 = 55; % I got this from PSD
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
rows_to_keep = setdiff(1:71, [13, 14, 18, 19, 65:71]);

% rest data processing
offline_rest_path = sprintf("%s/Offline/%s_Offline_s1Rest", subject_id, subject_id);
subject = load(offline_rest_path);
run = subject.(sprintf("%s_Offline_s1Rest", subject_id));

% Apply filtering to selected channels only
run1 = filtfilt(d, run.signal(:, rows_to_keep)); %filtfilt(d,filtfilt(b, a, run.signal));
%run1 = run.signal(:, rows_to_keep); % For calculating PSD

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

    % subject = load(strcat("Subject1/Offline/Subject1_Offline_s1r", num2str(i)));
    % run = subject.(strcat("Subject1_Offline_s1r", num2str(i)));
    % filteredSignal = filtfilt(b, a, run.signal);

    run1= filtfilt(d, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
    %run1 = run.signal;  % For calculating PSD
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

    % subject = load(strcat("Subject1/Online/Subject1_Online_s2r", num2str(i)));
    % run = subject.(strcat("Subject1_Online_s2r", num2str(i)));
    % filteredSignal = filtfilt(b, a, run.signal);

    run1= filtfilt(d, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
    %run1 = run.signal;  % For calculating PSD
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

    % subject = load(strcat("Subject1/Online/Subject1_Online_s3r", num2str(i)));
    % run = subject.(strcat("Subject1_Online_s3r", num2str(i)));
    % filteredSignal = filtfilt(b, a, run.signal);

    run1= filtfilt(d, run.signal); %filtfilt(d,filtfilt(b, a, run.signal));
    %run1 = run.signal;  % For calculating PSD
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
run1 = filtfilt(d, run.signal(:, rows_to_keep)); %filtfilt(d,filtfilt(b, a, run.signal));
%run1 = run.signal(:, rows_to_keep);  % For calculating PSD

open = find(run.header.EVENT.TYP == 10);
close = find(run.header.EVENT.TYP == 11);
finish = find(run.header.EVENT.TYP == 55555);

% Extract data for open and close events from selected channels only
session3.rest{1} = run1(run.header.EVENT.POS(open) : run.header.EVENT.POS(close), :);% 10
session3.rest{2} = run1(run.header.EVENT.POS(close) : run.header.EVENT.POS(finish(2)), :);


online.session3 = session3;

%% PSD

%{
% Parameters
fs = 512; % Sampling frequency
window = 4 * fs; % Window size for PSD estimation (e.g., 4 seconds)
noverlap = 2 * fs; % Overlap between windows (e.g., 2 seconds)
nfft = 2^nextpow2(window); % Number of FFT points

% Select a channel for PSD analysis
channel_to_analyze = 1;

% Compute PSD for rest data
[psd_rest, freq_rest] = pwelch(offline.rest{1}(:, channel_to_analyze), window, noverlap, nfft, fs);
psd_rest_db = 10*log10(psd_rest);

% Compute PSD for task data (assuming offline.runs.eeg{1}{1} contains task data)
[psd_task, freq_task] = pwelch(offline.runs.eeg{1}{1}(:, channel_to_analyze), window, noverlap, nfft, fs);
psd_task_db = 10*log10(psd_task);

% Plot PSD for rest and task
figure;
plot(freq_rest, psd_rest_db, 'b', 'LineWidth', 1.5);
hold on;
plot(freq_task, psd_task_db, 'r', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('PSD (dB/Hz)');
title(sprintf('Power Spectral Density - Channel %d', channel_to_analyze));
legend('Rest', 'Task');
grid on;

% Set the frequency limit for the x-axis (e.g., 0 to 100 Hz)
xlim([0, 100]);
%}

%% EOG Removal

% Calculate number of samples for N seconds
samples_to_plot = 10 * fs;

% Create the time vector for plotting
time_vector = (0:samples_to_plot - 1) / fs;

% Plot EOG channel
eog_channel = 32;  % Adjust the EOG channel index if necessary


% Select the data for plotting (rest)
size(offline.rest{1}, 1)
if size(offline.rest{1}, 1) >= samples_to_plot
    eog_data_to_plot = offline.rest{1}(1:samples_to_plot, eog_channel);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 288');
end

% Plot the EOG data
figure;
plot(time_vector, eog_data_to_plot);
hold on;
grid on;

channel_to_plot = 1;
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
title(sprintf('EOG Channel BEFORE FILTER', eog_channel));
legend('EOG', 'Channel');
hold off;

%{
remove_samples = @(data, threshold, window_size) ...
    unique(cell2mat(arrayfun(@(i) max(1, i - window_size):min(size(data, 1), i + window_size), ...
                             find(abs(data(:, eog_channel)) > threshold), 'UniformOutput', false)'));

% IMPORTANT VARS
high_window_size = 50; % how much padding around the removed points are also removed?
stdmult = 3; % above and below this sd, points are removed

% Remove samples for offline.rest
for i = 1:length(offline.rest)
    high_threshold = calculate_eog_threshold(offline.rest{i}, stdmult);  % Adjust the multiplier as needed
    eog_high_spikes = remove_samples(offline.rest{i}, high_threshold, high_window_size);
    indices_to_remove = unique(eog_high_spikes);
    offline.rest{i}(indices_to_remove, :) = [];
end

% Remove samples for offline.runs.eeg
for i = 1:length(offline.runs.eeg)
    for j = 1:length(offline.runs.eeg{i})
        high_threshold = calculate_eog_threshold(offline.runs.eeg{i}{j}, stdmult);  % Adjust the multiplier as needed
        eog_high_spikes = remove_samples(offline.runs.eeg{i}{j}, high_threshold, high_window_size);
        indices_to_remove = unique(eog_high_spikes);
        offline.runs.eeg{i}{j}(indices_to_remove, :) = [];
    end
end

% Remove samples for online.session2.eeg
for i = 1:length(online.session2.eeg)
    for j = 1:length(online.session2.eeg{i})
        high_threshold = calculate_eog_threshold(online.session2.eeg{i}{j}, stdmult);  % Adjust the multiplier as needed
        eog_high_spikes = remove_samples(online.session2.eeg{i}{j}, high_threshold, high_window_size);
        indices_to_remove = unique(eog_high_spikes);
        online.session2.eeg{i}{j}(indices_to_remove, :) = [];
    end
end

% Remove samples for online.session3.rest
for i = 1:length(online.session3.rest)
    high_threshold = calculate_eog_threshold(online.session3.rest{i}, stdmult);  % Adjust the multiplier as needed
    eog_high_spikes = remove_samples(online.session3.rest{i}, high_threshold, high_window_size);
    indices_to_remove = unique(eog_high_spikes);
    online.session3.rest{i}(indices_to_remove, :) = [];
end

% Remove samples for online.session3.eeg
for i = 1:length(online.session3.eeg)
    for j = 1:length(online.session3.eeg{i})
        high_threshold = calculate_eog_threshold(online.session3.eeg{i}{j}, stdmult);  % Adjust the multiplier as needed
        eog_high_spikes = remove_samples(online.session3.eeg{i}{j}, high_threshold, high_window_size);
        indices_to_remove = unique(eog_high_spikes);
        online.session3.eeg{i}{j}(indices_to_remove, :) = [];
    end
end
%}


% Remove EOG artifacts using polynomial regression followed by adaptive thresholding
polynomial_order = 3; % Specify the desired polynomial order
threshold_multiplier = 3; % Specify the desired threshold multiplier
window_size = 10; % Specify the window size for calculating the mean of surrounding values
offline.rest = cellfun(@(x) remove_eog_artifacts_regression(x, eog_channel, polynomial_order, threshold_multiplier, window_size), offline.rest, 'UniformOutput', false);
offline.runs.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order, threshold_multiplier, window_size), x, 'UniformOutput', false), offline.runs.eeg, 'UniformOutput', false);
online.session2.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order, threshold_multiplier, window_size), x, 'UniformOutput', false), online.session2.eeg, 'UniformOutput', false);
online.session3.rest = cellfun(@(x) remove_eog_artifacts_regression(x, eog_channel, polynomial_order, threshold_multiplier, window_size), online.session3.rest, 'UniformOutput', false);
online.session3.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order, threshold_multiplier, window_size), x, 'UniformOutput', false), online.session3.eeg, 'UniformOutput', false);


%{
% Remove EOG artifacts using higher-order polynomial regression
polynomial_order = 3; % Specify the desired polynomial order
offline.rest = cellfun(@(x) remove_eog_artifacts_regression(x, eog_channel, polynomial_order), offline.rest, 'UniformOutput', false);
offline.runs.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order), x, 'UniformOutput', false), offline.runs.eeg, 'UniformOutput', false);
online.session2.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order), x, 'UniformOutput', false), online.session2.eeg, 'UniformOutput', false);
online.session3.rest = cellfun(@(x) remove_eog_artifacts_regression(x, eog_channel, polynomial_order), online.session3.rest, 'UniformOutput', false);
online.session3.eeg = cellfun(@(x) cellfun(@(y) remove_eog_artifacts_regression(y, eog_channel, polynomial_order), x, 'UniformOutput', false), online.session3.eeg, 'UniformOutput', false);
%}

% Calculate number of samples for N seconds
samples_to_plot = 10 * fs;

% Create the time vector for plotting
time_vector = (0:samples_to_plot - 1) / fs;

% Plot EOG channel
eog_channel = 32;  % Adjust the EOG channel index if necessary


% Select the data for plotting (rest)
size(offline.rest{1}, 1)
if size(offline.rest{1}, 1) >= samples_to_plot
    eog_data_to_plot = offline.rest{1}(1:samples_to_plot, eog_channel);
else
    error('Not enough samples to plot. Check data dimensions and sampling rate. 288');
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


%% Correct Task Data Using Rest Means
% Correct Session 1 and 2
for i = 1:length(offline.runs.eeg)
    for j = 1:length(offline.runs.eeg{i})
        % offline.runs.eeg{i}{j} = 0;
        offline.runs.eeg{i}{j} = bsxfun(@minus, offline.runs.eeg{i}{j}, s1_rest_mean);
    end
end

for i = 1:length(online.session2.eeg)
    for j = 1:length(online.session2.eeg{i})
        % online.session2.eeg{i}{j} = 0;
        online.session2.eeg{i}{j} = bsxfun(@minus, online.session2.eeg{i}{j}, s1_rest_mean);
    end
end

% Correct Session 3
for i = 1:length(online.session3.eeg)
    for j = 1:length(online.session3.eeg{i})
        % online.session3.eeg{i}{j} = 0;
        online.session3.eeg{i}{j} = bsxfun(@minus, online.session3.eeg{i}{j}, s3_rest_mean);
    end
end

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

% Subject1 = struct('offline', offline, 'online', online);
eval([subject_id ' = struct(''offline'', offline, ''online'', online);']);

%% Functions
function threshold = calculate_eog_threshold(eog_data, multiplier)
    eog_channel = 32;
    threshold = multiplier * std(eog_data(:, eog_channel));
end

%{
function clean_eeg = remove_eog_artifacts_regression(eeg_data, eog_channel)
    % Perform EOG artifact removal using regression
    
    % Get the number of channels and samples
    num_channels = size(eeg_data, 2);
    num_samples = size(eeg_data, 1);
    
    % Initialize the cleaned EEG data
    clean_eeg = eeg_data;
    
    % Perform regression for each EEG channel
    for i = 1:num_channels
        if i ~= eog_channel
            % Perform linear regression between EOG channel and current EEG channel
            X = [ones(num_samples, 1), eeg_data(:, eog_channel)];
            beta = X \ eeg_data(:, i);
            
            % Estimate the EOG artifact using the regression coefficients
            eog_artifact = X * beta;
            
            % Remove the EOG artifact from the current EEG channel
            clean_eeg(:, i) = eeg_data(:, i) - eog_artifact;
        end
    end
end
%}
%{
function clean_eeg = remove_eog_artifacts_regression(eeg_data, eog_channel, polynomial_order)
    % Perform EOG artifact removal using regression
    
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
            
            % Estimate the EOG artifact using the regression coefficients
            eog_artifact = X * beta;
            
            % Remove the EOG artifact from the current EEG channel
            clean_eeg(:, i) = eeg_data(:, i) - eog_artifact;
        end
    end
end
%}
%{
function clean_eeg = remove_eog_artifacts_regression(eeg_data, eog_channel, polynomial_order, threshold_multiplier)
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
            
            % Estimate the EOG artifact using the regression coefficients
            eog_artifact = X * beta;
            
            % Calculate the adaptive threshold for the EOG artifact
            threshold = threshold_multiplier * std(eog_artifact);
            
            % Remove extreme outliers from the EOG artifact
            eog_artifact(abs(eog_artifact) > threshold) = 0;
            
            % Remove the EOG artifact from the current EEG channel
            clean_eeg(:, i) = eeg_data(:, i) - eog_artifact;
        end
    end
end
%}

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