%% RUNS

% Init vars
subject = 2;
session = 3;
run = 6;    

% Load data
filename = sprintf('/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/git_project/Subject%d/Online/Subject_212_HarmMI_Online__feedback_U_s003_r006_2024_03_22_165209.gdf', subject);
[signal, header] = sload(filename);

% Store header and signal into new struct
variableName = sprintf('Subject%d_Online_s%dr%d', subject, session, run);
eval([variableName '.header = header']);
eval([variableName '.signal = signal']);

% Save struct to file
new_filename = sprintf('Subject%d_Online_s%dr%d.mat', subject, session, run);
eval(['save(new_filename, ''' variableName ''');']);


%% REST

% Init vars
subject = 2;
session = 3;

% Load data
filename = sprintf('/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/git_project/Subject%d/Online/Subject_212_HarmMI_OnlineCalib__s003_r000_2024_03_22_155916.gdf', subject);
[signal, header] = sload(filename);

% Store header and signal into new struct
variableName = sprintf('Subject%d_Online_s%dRest', subject, session);
eval([variableName '.header = header']);
eval([variableName '.signal = signal']);

% Save struct to file
new_filename = sprintf('Subject%d_Online_s%dRest.mat', subject, session);
eval(['save(new_filename, ''' variableName ''');']);

%% RUNS (OFFLINE)

% Init vars
subject = 1;
session = 3;

% Load data
filename = sprintf('/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/project_git/Subject%d/Online/Subject_211_HarmMI_OnlineCalib__s003_r000_2024_03_19_173249.gdf', subject);
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


%%
get_old_file = load('/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/project_git/Subject1/Offline/Subject1_Offline_s1Rest.mat');

Subject1_Offline_s1Rest = get_old_file.Subject1_OfflineRest;

save( '/Users/weavejul/Documents/Class_and_Lab/NeuEng_Spring_24/Project/project_git/Subject1/Offline/Subject1_Offline_s1Rest.mat', 'Subject1_Offline_s1Rest' );



