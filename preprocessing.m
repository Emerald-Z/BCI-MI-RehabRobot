%% epoching is done here too
%% offline sesion
% filtering - bpf
fc1 = 0.1; % first cutoff frequency in Hz 
fc2 = 45; % second cutoff frequency in Hz -> i got this from a paper
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
subject = load("Subject1/Offline/Subject1_Offline_s1Rest");
run = subject.("Subject1_Offline_s1Rest");
run1= filtfilt(b, a, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
open = find(run.header.EVENT.TYP == 10);
close = find(run.header.EVENT.TYP == 11);
finish = find(run.header.EVENT.TYP == 55555);
offline.rest{1} = run1(run.header.EVENT.POS(open) : run.header.EVENT.POS(close));% 10 
offline.rest{2} = run1(run.header.EVENT.POS(close) : run.header.EVENT.POS(finish(2)));

% offline data
valuesStart=[7691, 7701];
valuesEnd=[7692, 7702];

% Create logical index to select rows to keep
rows_to_keep = setdiff(1:71, [13, 14, 18, 19, 65:71]);

%runs (task data)
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
session2.labels = struct('type', {cell(4, 1)}, 'end', {cell(4, 1)}, 'sustain', {cell(4, 1)});
% Session 2 (first online)
for i=1:4 % may have to manually change how many runs there were
    type = zeros(20, 1);
    ends = zeros(20, 1);
    % sustain = zeros(20, 1); ???
    data = cell(20, 1);
    session2.labels.type{i} = labels;
    session2.eeg{i} = data;
    subject = load(strcat("Subject1/Online/Subject1_Online_s2r", num2str(i)));
    run = subject.(strcat("Subject1_Online_s2r", num2str(i)));
    % filteredSignal = filtfilt(b, a, run.signal);

    run1= filtfilt(b, a, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
    run1_h_et=run.header.EVENT.TYP;
    run1_h_ep=run.header.EVENT.POS;

    run1_trials=[run1_h_ep(ismember(run1_h_et, valuesStart)) run1_h_ep(ismember(run1_h_et, valuesEnd))];
    r1_trialtype=[run1_h_et(ismember(run1_h_et, valuesStart)) run1_h_et(ismember(run1_h_et, valuesEnd))];
    run1_sustain=[run1_h_ep(ismember(run1_h_et, valuesSustain))];
    r1_sustaintype=[run1_h_et(ismember(run1_h_et, valuesSustain))];

    sustain_idx = 1;
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
            % add samples
            session2.eeg{i}{j} = run1(run1_trials(j,1):run1_trials(j,2), rows_to_keep);
        else
            % process sustain
            if (mod(r1_sustaintype(sustain_idx), 100) == 2)
                session2.labels.sustain{i}(j) = 1; % failed to sustain
            else 
                session2.labels.sustain{i}(j) = 2; % sustain succeeded
            end
            session2.eeg{i}{j} = run1(run1_trials(j,1) : run1_sustain(sustain_idx), rows_to_keep);
            sustain_idx = sustain_idx + 1;
        end
    end


end

online.session2 = session2;

% Session 3 (second online)
for i=1:6 % may have to manually change how many runs there were
    type = zeros(20, 1);
    ends = zeros(20, 1);
    % sustain = zeros(20, 1); ???
    data = cell(20, 1);
    session3.labels.type{i} = labels;
    session3.eeg{i} = data;
    subject = load(strcat("Subject1/Online/Subject1_Online_s3r", num2str(i)));
    run = subject.(strcat("Subject1_Online_s3r", num2str(i)));
    % filteredSignal = filtfilt(b, a, run.signal);

    run1= filtfilt(b, a, run.signal); %filtfilt(d,filtfilt(b, a, run.signal));
    run1_h_et=run.header.EVENT.TYP;
    run1_h_ep=run.header.EVENT.POS;

    run1_trials=[run1_h_ep(ismember(run1_h_et, valuesStart)) run1_h_ep(ismember(run1_h_et, valuesEnd))];
    r1_trialtype=[run1_h_et(ismember(run1_h_et, valuesStart)) run1_h_et(ismember(run1_h_et, valuesEnd))];
    run1_sustain=[run1_h_ep(ismember(run1_h_et, valuesSustain))];
    r1_sustaintype=[run1_h_et(ismember(run1_h_et, valuesSustain))];

    sustain_idx = 1;
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
            % add samples
            session3.eeg{i}{j} = run1(run1_trials(j,1):run1_trials(j,2), rows_to_keep);
        else
            % process sustain
            if (mod(r1_sustaintype(sustain_idx), 100) == 2)
                session3.labels.sustain{i}(j) = 1; % failed to sustain
            else 
                session3.labels.sustain{i}(j) = 2; % sustain succeeded
            end
            session3.eeg{i}{j} = run1(run1_trials(j,1) : run1_sustain(sustain_idx), rows_to_keep);
            sustain_idx = sustain_idx + 1;
        end
    end


end

% rest data processing for session 3
session3.rest = cell(2, 1);
subject = load("Subject1/Online/Subject1_Online_s3Rest");
run = subject.("Subject1_Online_s3Rest");
run1= filtfilt(b, a, run.signal);%filtfilt(d,filtfilt(b, a, run.signal));
open = find(run.header.EVENT.TYP == 10);
close = find(run.header.EVENT.TYP == 11);
finish = find(run.header.EVENT.TYP == 55555);

session3.rest{1} = run1(run.header.EVENT.POS(open) : run.header.EVENT.POS(close));% 10
session3.rest{2} = run1(run.header.EVENT.POS(close) : run.header.EVENT.POS(finish(2)));


online.session3 = session3;

%% make subject struct subject -> offline/online -> rest/active -> labels/eeg
Subject1 = struct('offline', offline, 'online', online);