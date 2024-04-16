function [MAV_feature, VAR_feature, ...
    RMS_feature, WL_feature, ZC_feature, ...
        SSC_feature, AR_feature, featureLabels] = extract_features(window_size, overlap, fs, filteredSignal, label) %add CSP_feature back once working
    WSize = floor(window_size*fs);	    % length of each data frame, 30ms
    nOlap = floor(overlap*WSize);  % overlap of successive frames, half of WSize
    hop = WSize-nOlap;        % amount to advance for next data frame
    nx = length(filteredSignal);                % length of input vector
    len = fix((nx - (WSize-hop))/hop);    %length of output vector = total frames
    
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
    
        % Iterate through the signal -- why would we iterate through the entire signal for this, wouldnt it just be the segment?
        for j = 2:length(segment)
            % Check for sign change
            if sign(segment(j)) ~= sign(segment(j-1))
                zero_crossings = zero_crossings + 1;
            end
        end
        ZC_feature(i) = zero_crossings;

        slope_sign_changes = 0;
   
        % Iterate through the signal to compute slopes
        for j = 2:length(segment)-1
            % Calculate slopes between consecutive points
            slope1 = segment(j) - segment(j-1);
            slope2 = segment(j+1) - segment(j);
            
            % Check for sign change in slopes
            if sign(slope1) ~= sign(slope2)
                slope_sign_changes = slope_sign_changes + 1;
            end
        end
        SSC_feature(i) = slope_sign_changes;
        %TOOO: idk if this is right

        %ar_coeffs = [0.5, -0.3, 0.2, -0.1]; % Example autoregressive coefficients
        [ar_coeffs, ~] = arburg(segment,4);

        % Generate white noise with the same length as the segment
        % Scale the white noise by 4
        white_noise = 4 * randn(size(segment));
    
        % Initialize the output signal array
        Xn = zeros(size(segment));
    
        % Assume initial conditions for the model (could use zeros or segment start)
        if length(segment) > p
            Xn(1:p) = segment(1:p);
        end
    
        % Apply the AR model formula to compute the signal including white noise
        for n = p+1:length(segment)
            Xn(n) = -sum(ar_coeffs(2:end) .* segment(n-p:n-1)) + white_noise(n);
        end

        % Sum the computed outputs to get a single feature
        AR_feature(i) = sum(Xn);

        %TOOO: idk if this is right
        % Generate example labels (replace with your actual class labels)
        %labels = [ones(1, num_trials_per_class), 2*ones(1, num_trials_per_class)]; % Two classes
        
        % Compute CSP filters
        %CSP_feature(i) = common_spatial_patterns(X, labels);
        % re-build the label vector to match it with the feature vector
        featureLabels(i) = sum(arrayfun(@(t) ((i-1)*hop+1) >= Rise1(t) && ((i-1)*hop+WSize) <= Fall1(t), 1:length(Rise1)));
    end

end
    