%% load gdf files
[signal, header] = sload("Subject1_OfflineRest.gdf");

% subject -> off/online -> session -> run resting/trials + header
% FeatureStore1 = struct('offline', {cell(6, 1)}, 'online', {cell(6, 1)});