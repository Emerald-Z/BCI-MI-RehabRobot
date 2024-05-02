function [avg_MAV_feature, avg_VAR_feature, ...
    avg_RMS_feature, avg_WL_feature, avg_ZC_feature, ...
        avg_SSC_feature, avg_AR_feature, featureLabels] = extract_avg_features(window_size, overlap, fs, filteredSignal, label)
    
    WSize = floor(window_size*fs);	    % length of each data frame, 30ms
    nOlap = floor(overlap*WSize);  % overlap of successive frames, half of WSize
    hop = WSize-nOlap;        % amount to advance for next data frame
    nx = length(filteredSignal);                % length of input vector
    len = fix((nx - (WSize-hop))/hop);    %length of output vector = total frames
    
    [avg_MAV_feature, avg_VAR_feature, avg_RMS_feature, avg_WL_feature, avg_ZC_feature, ...
        avg_SSC_feature, avg_AR_feature, avg_EN_feature, avg_PSD_feature, avg_FRAC_feature, avg_GFP_feature, avg_GD_feature, featureLabels] = deal(zeros(1,len * 10));
    disp("AVERAGE FEATURES!")
    disp(size(filteredSignal))
    %disp(size(avg_segments))

    avg_segments = cell(1, len*10);
    for i = 1:numel(avg_segments)
        avg_segments{i} = zeros(WSize, 1);
    end

    disp(size(avg_segments))
    disp("SIZE")
    disp(size(avg_MAV_feature))
    disp(len*10)

    for channel=1:10
        %fix channel signal here too :/
        %channel_signal = filteredSignal(:, channel);
        channel_signal = filteredSignal;
        idx = 0;
        for i = 1:len
            for j = 1:10
                segment = channel_signal(((i-1)*hop+1):((i-1)*hop+WSize), j);
                idx = (i-1) * 10 + j;
                %disp(avg_segments(idx))
                % if isempty(avg_segments{idx})
                %     disp("HI")
                %     disp(size(segment))
                %     avg_segments{idx} = segment;
                % else
                %disp("BRUH")
                %disp(size(segment))
                %disp(size(avg_segments{idx}))
                disp(idx)
                disp(i)
                disp(j)
                disp(channel)
                %avg_MAV_feature(idx) = 0;
                avg_segments{idx} = avg_segments{idx} + segment;
                %end
            end
        end
    end


    
    for i = 1:numel(avg_segments)
        avg_segments{i} = avg_segments{i} / 10;
    end
    disp(size(avg_segments))

    for seg_idx = 1:len*10
        % Extract the segment data for the current segment
        segment_data = avg_segments{seg_idx};
        
        % Calculate GFP for the current segment
        avg_GFP_feature(seg_idx) = sqrt(mean(var(segment_data, [], 2)));
        gfp_values_per_channel = sqrt(mean(var(segment_data, [], 2)));

        % Calculate mean GFP across channels within the segment
        mean_gfp_per_segment = mean(gfp_values_per_channel);
        
        % Calculate standard deviation of GFP values across channels within the segment
        std_dev_gfp_per_segment = std(gfp_values_per_channel);
        
        % Calculate global dissimilarity for the current segment
        avg_GD_feature(seg_idx) = std_dev_gfp_per_segment / mean_gfp_per_segment;
    end


    for channel=1:10
        %TODO: fix how I am getting the channel's signal here -- something is wrong :/
        disp(channel)
        channel_signal = filteredSignal(:, channel);
        [MAV_feature, VAR_feature,RMS_feature, WL_feature, ZC_feature, SSC_feature, AR_feature, EN_feature, FRAC_feature, CWT_feature, segments, featureLabels] = extract_features(window_size, overlap, fs, channel_signal, label);
        avg_MAV_feature = avg_MAV_feature + MAV_feature;
        avg_VAR_feature = avg_VAR_feature + VAR_feature;
        avg_RMS_feature = avg_RMS_feature + RMS_feature;
        avg_ZC_feature = avg_ZC_feature + ZC_feature;
        avg_WL_feature = avg_WL_feature + WL_feature;
        avg_SSC_feature = avg_SSC_feature + SSC_feature;
        avg_AR_feature = avg_AR_feature + AR_feature;
        avg_EN_feature = avg_EN_feature + EN_feature;
        avg_FRAC_feature = avg_FRAC_feature + FRAC_feature;
        avg_CWT_feature = avg_CWT_feature + CWT_feature;
    end

    

    avg_MAV_feature = avg_MAV_feature / 10;

    disp("DONE")
    
end