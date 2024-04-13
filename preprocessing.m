%% offline sesion
% filtering - bpf
fc1 = 50; % first cutoff frequency in Hz 
fc2 = 150; % second cutoff frequency in Hz 
fs = 512;
% normalize the frequencies
Wp = [fc1 fc2]*2/fs;

% Build a Butterworth bandpass filter of 4th order
% check the "butter" function in matlab
N=4;
[b,a]=butter(N, Wp, 'bandpass');

% artefact rejection - need to display the data to do this

% EOG removal - should we just remove it?

% split run data into task periods
offline = struct('rest', {cell(2, 1)}, 'runs', {cell(3, 1)});
runs = struct('eeg', {cell(3, 1)}, 'labels', {cell(3, 1)});

% rest data processing
subject = load("Subject1/Offline/Subject1_OfflineRest");
run = subject.("Subject1_OfflineRest");
run1= filtfilt(b, a, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
open = find(run.header.EVENT.TYP == 10);
close = find(run.header.EVENT.TYP == 11);
finish = find(run.header.EVENT.TYP == 55555);
offline.rest{1} = run1(run.header.EVENT.POS(open) : run.header.EVENT.POS(close));% 10 
offline.rest{2} = run1(run.header.EVENT.POS(close) : run.header.EVENT.POS(finish(2)));

% offline data
valuesStart=[7691, 7701];
valuesEnd=[7692, 7702];
for i=1:3
    labels = zeros(20, 1);
    data = cell(20, 1);
    runs.labels{i} = labels;
    runs.eeg{i} = data;
    subject = load(strcat("Subject1/Offline/Subject1_Offline_s1r", num2str(i)));
    run = subject.(strcat("Subject1_Offline_s1r", num2str(i)));
    % filteredSignal = filtfilt(b, a, run.signal);

    run1= filtfilt(b, a, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
    run1_h_et=run.header.EVENT.TYP;
    run1_h_ep=run.header.EVENT.POS;

    run1_trials=[run1_h_ep(ismember(run1_h_et, valuesStart)) run1_h_ep(ismember(run1_h_et, valuesEnd))];
    r1_trialtype=[run1_h_et(ismember(run1_h_et, valuesStart)) run1_h_et(ismember(run1_h_et, valuesEnd))];
    
    for j=1:20
        % add samples
        runs.eeg{i}{j} = run1(run1_trials(j,1):run1_trials(j,2));
        % append task label
        if (floor(mod(r1_trialtype(j,1)/100, 10)) == 6)
            runs.labels{j} = 1; % 6-rest 7-reach
        else
            runs.labels{j} = 2; % 21 reach 
        end


    end

end

offline.runs = runs;

%% online session
valuesStart=[7691, 7701];
valuesEnd=[7692, 7693, 7702, 7703];
sustain=[102, 103, 202, 203];

online = struct('eeg', {cell(3, 1)}, 'labels', struct('type', 'end', 'sustain'));

for i=1:3
    type = zeros(20, 1);
    ends = zeros(20, 1);
    sustain = zeros(20, 1);
    data = cell{20, 1};
    runs.labels{i} = labels;
    runs.eeg{i} = data;
    subject = load(strcat("Subject1/Online/Subject1_Online_s1r", num2str(i)));
    run = subject.(strcat("Subject1_Online_s1r", num2str(i)));
    % filteredSignal = filtfilt(b, a, run.signal);

    run1= filtfilt(b, a, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
    run1_h_et=run.header.EVENT.TYP;
    run1_h_ep=run.header.EVENT.POS;

    run1_trials=[run1_h_ep(ismember(run1_h_et, valuesStart)) run1_h_ep(ismember(run1_h_et, valuesEnd))];
    r1_trialtype=[run1_h_et(ismember(run1_h_et, valuesStart)) run1_h_et(ismember(run1_h_et, valuesEnd))];
    run1_sustain=[run1_h_ep(ismember(run1_h_et, sustain))];
    r1_sustaintype=[run1_h_et(ismember(run1_h_et, sustain))];

    sustain_idx = 1;
    for j=1:20
        % append task label
        if (floor(r1_trialtype(j,1)/100) == 6)
            runs.labels{j} = 1; % 6-rest 7-reach
        else
            runs.labels{j} = 2;
        end
        runs.end{j} = floor(r1_trialtype(j)/1000);

        if (floor(r1_trialtype(j,1)/1000) == 2)
            % fail
            runs.sustain{j} = 0;
            % add samples
            subject1Tasks{j} = run1(run1_trials(j,1):run1_trials(j,2));
        else
            % process sustain
            runs.sustain{j} = floor(r1_sustaintype(sustain_idx)/100);
            subject1Tasks{j} = run1(run1_trials(j,1): run1_sustain(sustain_idx));
        end

        sustain_idx = sustain_idx + 1;
    end

end

%% make subject struct subject -> offline/online -> rest/active -> labels/eeg
Subject1 = struct('offline', offline, 'online', online);