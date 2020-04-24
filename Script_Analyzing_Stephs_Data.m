% Analysing Steph's data
clear all 
close all

load 'responseMats.mat'

%Convert to ms
rt_midd12=1000*rt_midd12;
rt_post12=1000*rt_post12;

%% Plotting RTs by Distance - Phase 1
phase_1_xCond_cell = {[],[],[],[],[],[]};
phase_1_within_cell = {[],[],[],[],[]};

for sub=1:34
    for i=1:12
        for j=1:12
            if and(j<7,i>6)
                for k=1:6
                    if abs(j-(i-6))==k-1
                        phase_1_xCond_cell{k}(1,end+1) = rt_midd12(sub,i,j);
                    end
                end
            end
            if and(j>6,i<7)
                for k=1:6
                    if abs(i-(j-6))==k-1
                        phase_1_xCond_cell{k}(1,end+1) = rt_midd12(sub,i,j);
                    end
                end
            end
            if and(i<7,j<7) || and(i>6,j>6)
                for k=1:5
                    if abs(i-j)==k
                        phase_1_within_cell{k}(1,end+1) = rt_midd12(sub,i,j);
                    end
                end
            end            
        end    
    end
end

phase_1_xCond_RT = zeros(1,6);
phase_1_xCond_RT_StdError = zeros(1,6);
phase_1_within_RT = zeros(1,5);
phase_1_within_RT_StdError = zeros(1,5);
for i=1:6
    phase_1_xCond_RT(i) = nanmean(phase_1_xCond_cell{i});
    phase_1_xCond_RT_StdError(i) = nanstd(phase_1_xCond_cell{i})/sqrt(length(phase_1_xCond_cell{i}));
end
for i = 1:5
    phase_1_within_RT(i) = nanmean(phase_1_within_cell{i});
    phase_1_within_RT_StdError(i) = nanstd(phase_1_within_cell{i})/sqrt(length(phase_1_within_cell{i}));    
end

%% Plotting RTs by Distance - Phase 2
phase_2_xCond_cell = {[],[],[],[],[],[],[],[],[],[],[]};
phase_2_within_cell = {[],[],[],[],[]};
for sub=1:34
    for i=1:12
        for j=1:12
            if xor(i<7,j<7)
                for k=1:11
                    if abs(i-j)==k
                        phase_2_xCond_cell{k}(1,end+1) = rt_post12(sub,i,j);
                    end
                end
            end
            if and(i<7,j<7) || and(i>6,j>6)
                for k=1:5
                    if abs(i-j)==k
                        phase_2_within_cell{k}(1,end+1) =rt_post12(sub,i,j);
                    end
                end
            end            
        end    
    end
end
phase_2_xCond_RT = zeros(1,11);
phase_2_xCond_RT_StdError = zeros(1,11);
phase_2_within_RT = zeros(1,5);
phase_2_within_RT_StdError = zeros(1,5);
for i=1:11
    phase_2_xCond_RT(i) = nanmean(phase_2_xCond_cell{i});
    phase_2_xCond_RT_StdError(i) = nanstd(phase_2_xCond_cell{i})/sqrt(length(phase_2_xCond_cell{i}));
end
for i = 1:5
    phase_2_within_RT(i) = nanmean(phase_2_within_cell{i});
    phase_2_within_RT_StdError(i) = nanstd(phase_2_within_cell{i})/sqrt(length(phase_2_within_cell{i}));    
end

%% Plotting Phase 1 RTs

figure
hold on
errorbar([1:5],phase_1_within_RT,phase_1_within_RT_StdError, 'Linewidth',1.5)
errorbar([0:5],phase_1_xCond_RT,phase_1_xCond_RT_StdError, 'Linewidth',1.5)
xlabel('Distance', 'Fontsize', 15)
ylabel('RT (ms)', 'Fontsize', 15)
xlim([0,5])
xticks([0:1:5])
legend('Within', 'xCond')
hold off

%% Plotting Phase 2 RTs
figure
hold on
errorbar([1:5],phase_2_within_RT,phase_2_within_RT_StdError, 'Linewidth',1.5)
errorbar([1:11],phase_2_xCond_RT,phase_2_xCond_RT_StdError, 'Linewidth',1.5)
xlabel('Distance', 'Fontsize', 15)
ylabel('RT (ms)', 'Fontsize', 15)
xlim([1,11])
legend('Within', 'xCond')
hold off

%% Plotting mean Choice - Phase 1
figure
resp_midd12_normalized = zeros(34,12,12);
max_val_array_midd = [];
for sub=1:34
    max_val = max(resp_midd12(sub,:,:),[],'all');
    resp_midd12_normalized(sub,:,:) = resp_midd12(sub,:,:)/ceil(max_val);
    max_val_array_midd(end+1) = max_val;
end
midd_mean_choice = squeeze(nanmean(resp_midd12_normalized,1));
imagesc(midd_mean_choice)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)
%% Plotting mean Choice - Phase 2
figure
resp_post12_normalized = zeros(34,12,12);
max_val_array_post = [];
for sub=1:34
    max_val = max(resp_midd12(sub,:,:),[],'all');
    resp_post12_normalized(sub,:,:) = resp_post12(sub,:,:)/ceil(max_val);
    max_val_array_post(end+1) = max_val;
end
midd_mean_choice = squeeze(nanmean(resp_post12_normalized,1));
imagesc(midd_mean_choice)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

%% Median Split
resp_post12_normalized = zeros(34,12,12);
max_val_array_post = [];
for sub=1:34
    max_val = max(resp_midd12(sub,:,:),[],'all');
    resp_post12_normalized(sub,:,:) = resp_post12(sub,:,:)/ceil(max_val);
    max_val_array_post(end+1) = max_val;
end

post_perf_score = zeros(1,34);
for sub=1:34
    post_perf_score(sub) = perf(7,sub);
end
post_median = median(post_perf_score);
low_bool = zeros(1,34);
high_bool = zeros(1,34);
for sub=1:34
    if post_perf_score(sub) < post_median
        low_bool(sub)=1;
    end
    if post_perf_score(sub) >= post_median
        high_bool(sub)=1;
    end
end
sum(low_bool)
sum(high_bool)
low_median_split = squeeze(nanmean(resp_post12_normalized(low_bool==1,:,:),1));
high_median_split = squeeze(nanmean(resp_post12_normalized(high_bool==1,:,:),1));

figure
imagesc(low_median_split)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

figure
imagesc(high_median_split)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)