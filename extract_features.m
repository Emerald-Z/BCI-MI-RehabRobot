function [MAV_feature, VAR_feature, ...
    RMS_feature, WL_feature, ZC_feature, ...
        SSC_feature, AR_feature, CSP_feature, featureLabels] = extract_features(window_size, overlap, fs, filteredSignal, label)
    WSize = floor(window_size*fs);	    % length of each data frame, 30ms
    nOlap = floor(overlap*WSize);  % overlap of successive frames, half of WSize
    hop = WSize-nOlap;	    % amount to advance for next data frame
    nx = length(filteredSignal);	            % length of input vector
    len = fix((nx - (WSize-hop))/hop);	%length of output vector = total frames
    
    % preallocate outputs for speed
    [MAV_feature, VAR_feature, RMS_feature, WL_feature, ZC_feature, ...
        SSC_feature, AR_feature, CSP_feature, featureLabels] = deal(zeros(1,len));
    
    Rise1 = gettrigger(label,0.5); % gets the starting points of stimulations
    Fall1 = gettrigger(-label,-0.5); % gets the ending points of stimulations
    
    for i = 1:len
        segment = filteredSignal(((i-1)*hop+1):((i-1)*hop+WSize));
        MAV_feature(i) = mean(abs(segment));   
        VAR_feature(i) = mean((segment-mean(segment)).^2);

        RMS_feature(i) = sqrt(mean(segment.^2));
        WL_feature(i) = sum(abs(diff(segment)));

        zero_crossings = 0;
    
        % Iterate through the signal
        for j = 2:length(signal)
            % Check for sign change
            if sign(signal(j)) ~= sign(signal(j-1))
                zero_crossings = zero_crossings + 1;
            end
        end
        ZC_feature(i) = zero_crossings;


        slope_sign_changes = 0;
   
        % Iterate through the signal to compute slopes
        for j = 2:length(signal)-1
            % Calculate slopes between consecutive points
            slope1 = signal(j) - signal(j-1);
            slope2 = signal(j+1) - signal(j);
            
            % Check for sign change in slopes
            if sign(slope1) ~= sign(slope2)
                slope_sign_changes = slope_sign_changes + 1;
            end
        end
        SSC_feature(i) = slope_sign_changes;
        %TOOO: idk if this is right

        ar_coeffs = [0.5, -0.3, 0.2, -0.1]; % Example autoregressive coefficients
        AR_feature(i) = calculate_autoregressive_feature(X, ar_coeffs); % only frequency domain

        %TOOO: idk if this is right
        % Generate example labels (replace with your actual class labels)
        labels = [ones(1, num_trials_per_class), 2*ones(1, num_trials_per_class)]; % Two classes
        
        % Compute CSP filters
        CSP_feature(i) = common_spatial_patterns(X, labels);
        % re-build the label vector to match it with the feature vector
        featureLabels(i) = sum(arrayfun(@(t) ((i-1)*hop+1) >= Rise1(t) && ((i-1)*hop+WSize) <= Fall1(t), 1:length(Rise1)));
    end
    
    % figure;

    % stem(find(featureLabels == 1), ones(1, length(find(featureLabels ==1))).*max(max(MAV_feature), max(VAR_feature)), 'Color', 'r', 'LineWidth', 0.2, 'DisplayName', 'Labels for stimulation');
    % hold on;
    % stem(find(featureLabels == 0), ones(1, length(find(featureLabels ==0))).*max(max(MAV_feature), max(VAR_feature)), 'Color', 'c', 'LineWidth', 0.2, 'DisplayName', 'Labels for rest');
    % hold on;
    % plot(1:length(MAV_feature), MAV_feature, 'Color', 'k', 'LineWidth', 0.2, 'DisplayName', strcat('MAV feature, WSize: ', num2str(window_size), 'Olap: ', num2str(overlap))); % , 'linestyle','none','marker','o'
    % grid on; grid minor;
    % set(gca, 'YScale', 'log');
    % xlabel('Frame count');
    % ylabel('Amplitude (uV)');
    % title('MAV and VAR Features: ', 'Flex');
    % lgd=legend('show');
    % lgd.FontSize=11;
    % set(gca, 'FontSize', 15)

end