%% RUNS

% Init vars
subject = 1;
session = 1;
run = 3;    

% Load data
filename = sprintf('/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/Data/Subject1/Subject%d_Offline_s%dr%d.gdf', subject, session, run)
[signal, header] = sload(filename);

% Store header and signal into new struct
variableName = sprintf('Subject%d_Offline_s%dr%d', subject, session, run);
eval([variableName '.header = header'])
eval([variableName '.signal = signal'])

% Save struct to file
new_filename = sprintf('Subject%d_Offline_s%dr%d.mat', subject, session, run)
eval(['save(new_filename, ''' variableName ''');']);


%% REST

% Init vars
subject = 1;
session = 1

% Load data
filename = sprintf('/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/Data/Subject1/Subject%d_OfflineRest.gdf', subject)
[signal, header] = sload(filename);

% Store header and signal into new struct
variableName = sprintf('Subject%d_Offline_s%dRest', subject);
eval([variableName '.header = header'])
eval([variableName '.signal = signal'])

% Save struct to file
new_filename = sprintf('Subject%d_Offline_s%dRest.mat', subject, session)
eval(['save(new_filename, ''' variableName ''');']);

%% RUNS (ONLINE)

% Init vars
subject = 1;
session = 2;

% Load data
filename = sprintf('Subject%d/Online/sess3/Subject_211_HarmMI_Online__feedback_U_s003_r006_2024_03_19_182442.gdf', subject);
[signal, header] = sload(filename);

% Store header and signal into new struct    
variableName = sprintf('Subject%d_Online_s%dRest', subject, session);
eval([variableName '.header = header'])
eval([variableName '.signal = signal'])

% Save struct to file
new_filename = sprintf('Subject%d_Online_s%dRest.mat', subject, session)
eval(['save(new_filename, ''' variableName ''');']);

%% REST (ONLINE)

% Init vars
subject = 1;
session = 3;

% Load data
filename = sprintf('Subject%d/Online/sess3/restingState/Subject_211_HarmMI_OnlineCalib__s003_r000_2024_03_19_173249.gdf', subject);
[signal, header] = sload(filename);

% Store header and signal into new struct    
variableName = sprintf('Subject%d_Online_s%dRest', subject, session);
eval([variableName '.header = header'])
eval([variableName '.signal = signal'])

% Save struct to file
new_filename = sprintf('Subject%d_Online_s%dRest.mat', subject, session)
eval(['save(new_filename, ''' variableName ''');']);
