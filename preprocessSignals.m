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

% Load data
filename = sprintf('/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/Data/Subject1/Subject%d_OfflineRest.gdf', subject)
[signal, header] = sload(filename);

% Store header and signal into new struct
variableName = sprintf('Subject%d_OfflineRest', subject);
eval([variableName '.header = header'])
eval([variableName '.signal = signal'])

% Save struct to file
new_filename = sprintf('Subject%d_OfflineRest.mat', subject)
eval(['save(new_filename, ''' variableName ''');']);
