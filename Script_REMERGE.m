% REMERGE RSA
clear all
close all

%% Parameters

tau = 0.25;
C = 1;
beta = 0.3;
w = 1.7;
lambda = 0.2;
estr = 0.5;             %Scaling factor of external input

num_obj = 6;
num_cycles = 500;
num_patterns = (2*num_obj)^2;


%Create weight matrix between feature and conjunctive layer
weight_vector_diag = w*ones(1,2*num_obj);
weight_vector_off_diag = w*ones(1,2*num_obj-1);
W1_phase1 = diag(weight_vector_diag)+diag(weight_vector_off_diag,1);
W1_phase1(2*num_obj,:) = [];                             %Delete num_obj'th row     
%Set row corresponding to the [6,7] conjunctive unit (i.e row num_obj) to zero
W1_phase1(num_obj,:) = zeros(1, 2*num_obj);

%Create weight matrix between conjunctive and response layer
W2_phase1 = diag(weight_vector_diag) - diag(weight_vector_off_diag,1);
W2_phase1(:,1) = [];                                      %Delete first column
%Set column corresponding to the [6,7] conjunctive unit (i.e column num_obj) to zero
W2_phase1(:,num_obj) = zeros( 2*num_obj, 1); 

test_input = zeros(num_patterns, 2);
for i=1:2*num_obj
    for j=1:2*num_obj
        test_input(j+(i-1)*2*num_obj,:) = [i,j];
    end
end

phase_1_response = zeros(num_patterns, 2*num_obj);
phase_1_hidden = zeros(num_patterns, 4*num_obj-1); 
phase_1_rt = zeros(num_patterns,1);

%% Post Phase 1 testing
response_act_1 = zeros(num_patterns, num_cycles,2*num_obj);
for p=1:num_patterns
    %zero the feature and conjunctive net inputs
    feat_net= zeros(num_cycles,2*num_obj);
    feat_act = zeros(num_cycles,2*num_obj);
    conj_net = zeros(num_cycles, 2*num_obj-1);
    conj_act = zeros(num_cycles, 2*num_obj-1);
    response_net= zeros(num_cycles,2*num_obj);
    
    i = test_input(p,1);
    j = test_input(p,2);
    ext_input = zeros(1,2*num_obj);
    ext_input([i,j])= estr;
    feat_net(1, :) = feat_net(1,:) + ext_input;
    feat_act(1,:) = scaled_sigmoid_func(feat_net(1,:), tau);

    for t = 2:num_cycles
        conj_net(t,:) = lambda*(W1_phase1*(feat_act(t-1,:)'))' + (1-lambda)*conj_net(t-1,:);
        conj_act(t,:) = hedged_softmax_func(conj_net(t,:), tau, C);
        
        feat_net(t,:) = lambda*(W1_phase1'*(conj_act(t-1,:)'))' + ext_input + (1-lambda)*feat_net(t-1,:);
        feat_act(t,:) = Rectified_scaled_sigmoid_func(feat_net(t,:), tau);     
        
        response_net(t,:) = (W2_phase1*(conj_act(t,:)'))'; 
        response_net(t,:) = sigmoid_func(response_net(t,:), tau);
        response_act_1(p,t,[i,j]) = hedged_softmax_func(response_net(t,[i,j]), beta, 0);
    end   
    %calculate RTs
    [val, idx] = max(response_act_1(p,end,:));
    rt = num_cycles;
    for t=num_cycles:-1:2        
        if abs(response_act_1(p,t,idx)-val)>0.05*val
            break
        end
        rt=rt-1;
    end
    phase_1_rt(p,1)=rt;
    
    phase_1_hidden(p,:) = cat(2, feat_act(end,:), conj_act(end,:));
    phase_1_response(p,:) = response_act_1(p,end,:);
end

%% Phase 2 Model

W1_phase2 = diag(weight_vector_diag)+diag(weight_vector_off_diag,1);
W1_phase2(2*num_obj,:) = [];                             %Delete 2*num_obj'th row     

%Create weight matrix between conjunctive and response layer
W2_phase2 = diag(weight_vector_diag) - diag(weight_vector_off_diag,1);
W2_phase2(:,1) = [];                                      %Delete first column

phase_2_response = zeros(num_patterns,2*num_obj);
phase_2_hidden = zeros(num_patterns, 4*num_obj-1); 
phase_2_rt = zeros(num_patterns, 1);


 response_act_2 = zeros(num_patterns,num_cycles,2*num_obj);
for p=1:num_patterns
    %zero the feature and conjunctive net inputs
    feat_net= zeros(num_cycles,2*num_obj);
    feat_act = zeros(num_cycles,2*num_obj);
    conj_net = zeros(num_cycles, 2*num_obj-1);
    conj_act = zeros(num_cycles, 2*num_obj-1);
    response_net= zeros(num_cycles,2*num_obj);

    i = test_input(p,1);
    j = test_input(p,2);
   
    ext_input = zeros(1,2*num_obj);
    ext_input([i,j])= estr;
    feat_net(1, :) = feat_net(1,:) + ext_input;
    feat_act(1,:) = scaled_sigmoid_func(feat_net(1,:), tau);

    for t = 2:num_cycles
        conj_net(t,:) = lambda*(W1_phase2*(feat_act(t-1,:)'))' + (1-lambda)*conj_net(t-1,:);
        conj_act(t,:) = hedged_softmax_func(conj_net(t,:), tau, C);
        
        feat_net(t,:) = lambda*(W1_phase2'*(conj_act(t-1,:)'))' + ext_input + (1-lambda)*feat_net(t-1,:);
        feat_act(t,:) = Rectified_scaled_sigmoid_func(feat_net(t,:), tau); 
        
        response_net(t,:) = W2_phase2*(conj_act(t,:)'); 
        response_net(t,:) = sigmoid_func(response_net(t,:), tau);
        response_act_2(p, t,[i,j]) = hedged_softmax_func(response_net(t,[i,j]), beta, 0);               
    end
    
    %calculate RTs
    [val, idx] = max(response_act_2(p,end,:));
    rt = num_cycles;
    for t=num_cycles:-1:2
        if abs(response_act_2(p,t,idx)-val)>0.05*val
            break
        end
        rt=rt-1;     
    end
    phase_2_rt(p,1)=rt;
    
    phase_2_hidden(p,:) = cat(2, feat_act(end,:), conj_act(end,:));
    phase_2_response(p,:) = response_act_2(p,end,:);
end

%% Phase 3 Model - Ambiguous Linking 

W1_phase3 = W1_phase1;
W1_phase3([6,6],[5,8]) = w;                             %Delete 2*num_obj'th row     

%Create weight matrix between conjunctive and response layer
W2_phase3 = W2_phase1;
W2_phase3(5,6) = -w;
W2_phase3(8,6) = w;

phase_3_response = zeros(num_patterns,2*num_obj);
phase_3_hidden = zeros(num_patterns, 4*num_obj-1); 
phase_3_rt = zeros(num_patterns, 1);


 response_act_3 = zeros(num_patterns,num_cycles,2*num_obj);
for p=1:num_patterns
    %zero the feature and conjunctive net inputs
    feat_net= zeros(num_cycles,2*num_obj);
    feat_act = zeros(num_cycles,2*num_obj);
    conj_net = zeros(num_cycles, 2*num_obj-1);
    conj_act = zeros(num_cycles, 2*num_obj-1);
    response_net= zeros(num_cycles,2*num_obj);

    i = test_input(p,1);
    j = test_input(p,2);
   
    ext_input = zeros(1,2*num_obj);
    ext_input([i,j])= estr;
    feat_net(1, :) = feat_net(1,:) + ext_input;
    feat_act(1,:) = scaled_sigmoid_func(feat_net(1,:), tau);

    for t = 2:num_cycles
        conj_net(t,:) = lambda*(W1_phase3*(feat_act(t-1,:)'))' + (1-lambda)*conj_net(t-1,:);
        conj_act(t,:) = hedged_softmax_func(conj_net(t,:), tau, C);
        
        feat_net(t,:) = lambda*(W1_phase3'*(conj_act(t-1,:)'))' + ext_input + (1-lambda)*feat_net(t-1,:);
        feat_act(t,:) = Rectified_scaled_sigmoid_func(feat_net(t,:), tau); 
        
        response_net(t,:) = W2_phase3*(conj_act(t,:)'); 
        response_net(t,:) = sigmoid_func(response_net(t,:), tau);
        response_act_3(p, t,[i,j]) = hedged_softmax_func(response_net(t,[i,j]), beta, 0);               
    end
    
    %calculate RTs
    [val, idx] = max(response_act_3(p,end,:));
    rt = num_cycles;
    for t=num_cycles:-1:2
        if abs(response_act_3(p,t,idx)-val)>0.05*val
            break
        end
        rt=rt-1;     
    end
    phase_3_rt(p,1)=rt;
    
    phase_3_hidden(p,:) = cat(2, feat_act(end,:), conj_act(end,:));
    phase_3_response(p,:) = response_act_3(p,end,:);
end

%% Calculating RDM

phase_1_mean_hidden = zeros(12, 4*num_obj-1);
phase_2_mean_hidden = zeros(12, 4*num_obj-1);
phase_3_mean_hidden = zeros(12, 4*num_obj-1);

for i=1:12
   phase_1_mean_hidden(i, :) =  mean(phase_1_hidden(((i-1)*12+1:i*12),:));
   phase_2_mean_hidden(i, :) =  mean(phase_2_hidden(((i-1)*12+1:i*12),:));
   phase_3_mean_hidden(i, :) =  mean(phase_3_hidden(((i-1)*12+1:i*12),:));
end

phase_1_RDM_euclid = zeros(12,12);
phase_2_RDM_euclid = zeros(12,12);
phase_3_RDM_euclid = zeros(12,12);

for i=1:12
    for j = 1:12
        phase_1_RDM_euclid(i,j) = norm(phase_1_mean_hidden(i,:)-phase_1_mean_hidden(j,:));
        phase_2_RDM_euclid(i,j) = norm(phase_2_mean_hidden(i,:)-phase_2_mean_hidden(j,:));
        phase_3_RDM_euclid(i,j) = norm(phase_3_mean_hidden(i,:)-phase_3_mean_hidden(j,:));
       
    end
end

%% Calculating the Decision Matrices and RT Matrices
data_1 = zeros(2*num_obj,2*num_obj);
data_2 = zeros(2*num_obj,2*num_obj);
data_3 = zeros(2*num_obj,2*num_obj);
rt_1 = zeros(2*num_obj,2*num_obj);
rt_2 = zeros(2*num_obj,2*num_obj);
rt_3 = zeros(2*num_obj,2*num_obj);


for i=1:2*num_obj
    for j=1:2*num_obj
        data_1(j, i) = phase_1_response(j+(i-1)*2*num_obj, i);
        data_2(j, i) = phase_2_response(j+(i-1)*2*num_obj, i);
        data_3(j, i) = phase_3_response(j+(i-1)*2*num_obj, i);

    end
end

for i=1:2*num_obj
    rt_1(:,i) = phase_1_rt((i-1)*2*num_obj+1:i*2*num_obj);
    rt_2(:,i) = phase_2_rt((i-1)*2*num_obj+1:i*2*num_obj);
    rt_3(:,i) = phase_3_rt((i-1)*2*num_obj+1:i*2*num_obj);
end

%% Plotting RTs by Distance - Phase 1
phase_1_xCond_cell = {[],[],[],[],[],[]};
phase_1_within_cell = {[],[],[],[],[]};

for i=1:12
    for j=1:12
        if and(j<7,i>6)
            for k=1:6
                if abs(j-(i-6))==k-1
                    phase_1_xCond_cell{k}(1,end+1) = rt_1(i,j);
                end
            end
        end
        if and(j>6,i<7)
            for k=1:6
                if abs(i-(j-6))==k-1
                    phase_1_xCond_cell{k}(1,end+1) = rt_1(i,j);
                end
            end
        end
        if and(i<7,j<7) || and(i>6,j>6)
            for k=1:5
                if abs(i-j)==k
                    phase_1_within_cell{k}(1,end+1) = rt_1(i,j);
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
    phase_1_xCond_RT(i) = mean(phase_1_xCond_cell{i});
    phase_1_xCond_RT_StdError(i) = std(phase_1_xCond_cell{i})/sqrt(length(phase_1_xCond_cell{i}));
end
for i = 1:5
    phase_1_within_RT(i) = mean(phase_1_within_cell{i});
    phase_1_within_RT_StdError(i) = std(phase_1_within_cell{i})/sqrt(length(phase_1_within_cell{i}));    
end

%% Plotting RTs by Distance - Phase 2
phase_2_xCond_cell = {[],[],[],[],[],[],[],[],[],[],[]};
phase_2_within_cell = {[],[],[],[],[]};

for i=1:12
    for j=1:12
        if xor(i<7,j<7)
            for k=1:11
                if abs(i-j)==k
                    phase_2_xCond_cell{k}(1,end+1) = rt_2(i,j);
                end
            end
        end
        if and(i<7,j<7) || and(i>6,j>6)
            for k=1:5
                if abs(i-j)==k
                    phase_2_within_cell{k}(1,end+1) = rt_2(i,j);
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
    phase_2_xCond_RT(i) = mean(phase_2_xCond_cell{i});
    phase_2_xCond_RT_StdError(i) = std(phase_2_xCond_cell{i})/sqrt(length(phase_2_xCond_cell{i}));
end
for i = 1:5
    phase_2_within_RT(i) = mean(phase_2_within_cell{i});
    phase_2_within_RT_StdError(i) = std(phase_2_within_cell{i})/sqrt(length(phase_2_within_cell{i}));    
end


%% Plotting Figures
figure
imagesc(data_1, [0 1])
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

figure
imagesc(data_2,[0 1])
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

figure
imagesc(data_3, [0 1])
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

figure
imagesc(phase_1_RDM_euclid)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
colormap(bluewhitered(256))
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Dissimilarity', 'Fontsize', 15)

figure
imagesc(phase_2_RDM_euclid)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
colormap(bluewhitered(256))
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Dissimilarity', 'Fontsize', 15)

figure
imagesc(phase_3_RDM_euclid)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
colormap(bluewhitered(256))
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Dissimilarity', 'Fontsize', 15)

figure
phase_3_rtplot_5 = rt_3(:,5);
phase_3_rtplot_5(5) = NaN;
plot([1:12], phase_3_rtplot_5, '.-', 'Linewidth', 1.5,'MarkerSize',20 )
xlim([1 12])
xlabel('Object', 'Fontsize', 15)
ylabel('Reaction Time (Cycles)', 'Fontsize', 15)

figure
phase_3_rtplot_8 = rt_3(:,8);
phase_3_rtplot_8(8) = NaN;
plot([1:12], phase_3_rtplot_8, '.-', 'Linewidth', 1.5,'MarkerSize',20 )
xlim([1 12])
xlabel('Object', 'Fontsize', 15)
ylabel('Reaction Time (Cycles)', 'Fontsize', 15)

figure
hold on
errorbar([1:5],phase_1_within_RT,phase_1_within_RT_StdError, 'Linewidth',1.5)
errorbar([0:5],phase_1_xCond_RT,phase_1_xCond_RT_StdError, 'Linewidth',1.5)
xlabel('Distance', 'Fontsize', 15)
ylabel('Reaction Time (Cycles)', 'Fontsize', 15)
xlim([0,5])
legend('Within', 'xCond')
hold off

figure
hold on
errorbar([1:5],phase_2_within_RT,phase_2_within_RT_StdError, 'Linewidth',1.5)
errorbar([1:11],phase_2_xCond_RT,phase_2_xCond_RT_StdError, 'Linewidth',1.5)
xlabel('Distance', 'Fontsize', 15)
ylabel('Reaction Time (Cycles)', 'Fontsize', 15)
xlim([1,11])
legend('Within', 'xCond')
hold off

%%
figure
kk= 7;
yyaxis left

phase_3_plot = data_3(:,kk);
phase_3_plot(kk) = NaN;
plot([1:12], phase_3_plot, '.-', 'Linewidth', 1.5,'MarkerSize',20)
xlim([1 12])
ylim([0 0.52])
xlabel('Object', 'Fontsize', 15)
ylabel('Probability', 'Fontsize', 15)

yyaxis right
phase_3_rtplot = rt_3(:,kk);
phase_3_rtplot(kk) = NaN;
plot([1:12], phase_3_rtplot, '.-', 'Linewidth', 1.5,'MarkerSize',20 )
xlim([1 12])
ylim([0 46])
ylabel('Reaction Time (Cycles)', 'Fontsize', 15)
legend('Choice Probability', 'Reaction Time')
