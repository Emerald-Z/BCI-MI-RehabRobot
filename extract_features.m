function [MAV_feature, VAR_feature, ...
    RMS_feature, WL_feature, ZC_feature, ...
        SSC_feature, AR_feature, EN_feature, FRAC_feature, CWT_feature, segments, featureLabels] = extract_features(window_size, overlap, fs, filteredSignal, label)
    WSize = floor(window_size*fs);	    % length of each data frame, 30ms
    nOlap = floor(overlap*WSize);  % overlap of successive frames, half of WSize
    hop = WSize-nOlap;        % amount to advance for next data frame
    nx = length(filteredSignal);                % length of input vector
    len = fix((nx - (WSize-hop))/hop);    %length of output vector = total frames
    
    % preallocate outputs for speed
    [MAV_feature, VAR_feature, RMS_feature, WL_feature, ZC_feature, ...
        SSC_feature, AR_feature, EN_feature, PSD_feature, FRAC_feature, CWT_feature, segments, featureLabels] = deal(zeros(1,len * 10));
    
    Rise1 = gettrigger(label,0.5); % gets the starting points of stimulations
    Fall1 = gettrigger(-label,-0.5); % gets the ending points of stimulations

    % Define wavelet parameters
    waveletName = 'morl'; % Morlet wavelet
    scales = 1:128; % Define scales for the CWT
    
    for i = 1:len
        for j = 1:10
            segment = filteredSignal(((i-1)*hop+1):((i-1)*hop+WSize), j);
            idx = (i-1) * 10 + j;
            % PSD_feature(i) = pwelch(segment, length(segment), 0, [12 30]);

            %segments(idx) = segment;

            MAV_feature(idx) = mean(abs(segment));   
            VAR_feature(idx) = mean((segment-mean(segment)).^2);
    
            RMS_feature(idx) = sqrt(mean(segment.^2));
            WL_feature(idx) = sum(abs(diff(segment)));
    
            zero_crossings = 0;
        
            % Iterate through the segment
            for j = 2:length(segment)
                % Check for sign change
                if sign(segment(j)) ~= sign(segment(j-1))
                    zero_crossings = zero_crossings + 1;
                end
            end
            ZC_feature(idx) = zero_crossings;
    
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
            SSC_feature(idx) = slope_sign_changes;
    
            p=4;
    
            %ar_coeffs = [0.5, -0.3, 0.2, -0.1]; % Example autoregressive coefficients
            [ar_coeffs, ~] = arburg(segment,p);
    
            % Generate white noise with the same length as the segment
            % Scale the white noise by 4
            white_noise = p * randn(size(segment));
        
            % Initialize the output signal array
            Xn = zeros(size(segment));
        
            % Assume initial conditions for the model (could use zeros or segment start)
            if length(segment) > p
                Xn(1:p) = segment(1:p);
            end
        
            % Apply the AR model formula to compute the signal including white noise
            for n = p+1:length(segment)
                Xn(n) = -sum(ar_coeffs(2:end)' .* segment(n-p:n-1)) + white_noise(n);
    
            end
    
            % Sum the computed outputs to get a single feature
            AR_feature(idx) = sum(Xn);

            % ENTROPY FEATURE
            num_bins = 50;
            epsilon = 1e-10;

            % bin_numbers = [25, 50, 100, 150, 200]; <-- used this to find
            % optimal bin size
            % 
            % figure;  % Create a new figure for plotting
            % 
            % for i = 1:length(bin_numbers)
            %     num_bins = bin_numbers(i);
            % 
            %     % Compute the histogram
            %     [signal_counts, bin_edges] = histcounts(segment, num_bins);
            %     bin_centers = bin_edges(1:end-1) + diff(bin_edges)/2;
            % 
            %     % Plot the histogram
            %     subplot(length(bin_numbers), 1, i);
            %     bar(bin_centers, signal_counts);
            %     title(['Histogram with ', num2str(num_bins), ' bins']);
            %     xlabel('Signal Value');
            %     ylabel('Count');
            % end
            % 
            % sgtitle('Signal Histograms with Different Bin Numbers');
            
            % Discretize the signal into bins
            [signal_counts, ~] = histcounts(segment, num_bins);
            %disp(signal_counts)
            probabilities = (signal_counts+epsilon) / sum(signal_counts+epsilon);
            %disp(probabilities)
            entropy_feat = -sum(probabilities .* log2(probabilities));
            EN_feature(idx) = entropy_feat;

            %Power Spectral Density Feature
            %PSD_feature(idx) = ;
            %Fracral Dimension Feature
            maxK = 5;
            N = length(segment);
            L = 1:maxK;
            FD = zeros(size(L));
            for k = L
                Lm = zeros(1, k);
                for m = 1:k
                    x = zeros(1, floor((N - m) / k));
                    for j = 1:floor((N - m) / k)
                        x(j) = segment((m - 1) + j * k);
                    end
                    Lm(m) = sum(abs(diff(x)));
                end
                FD(k) = log(sum(Lm) * (N - 1) / ((N - m) * k));
            end
            FD = polyfit(log(L), log(FD), 1);
            FD = FD(1);
            %disp(FD);
            FRAC_feature(idx) = FD;
            %disp(entropy_feat)
            %entropy_feat = entropy(segment);
    
            %TOOO: idk if this is right
            % Generate example labels (replace with your actual class labels)
            %labels = [ones(1, num_trials_per_class), 2*ones(1, num_trials_per_class)]; % Two classes
            
            % Compute CSP filters
            %CSP_feature(i) = common_spatial_patterns(X, labels);
            % re-build the label vector to match it with the feature vector

            % Compute Continuous Wavelet Transform (CWT) using cwtft
            cwtStruct = cwtft({segment,1/fs},'wavelet',waveletName,'scales',scales);

            % Extract the magnitude of the CWT coefficients at the target scale
            cwtMagnitude = abs(cwtStruct.cfs(64,:));

            % Calculate feature from the CWT (e.g., maximum magnitude at target scale)
            CWT_feature(idx) = max(cwtMagnitude);
    
            featureLabels(i) = sum(arrayfun(@(t) ((i-1)*hop+1) >= Rise1(t) && ((i-1)*hop+WSize) <= Fall1(t), 1:length(Rise1)));
        end

    end

end
    