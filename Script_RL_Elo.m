% RL-Model for Transitive Inference
% Two rules model
clear all
close all
%% Initializing Parameters
obj_num = 6;                          %Number of objects in each of our two groups

init_scale = 0.1;                     %Scale factor of our initilization of object values
lambda = 0.1;                         %Learning parameter
asym_scale = 1;                       %Scale factor for aysmmetric learning. 1= perfectly symmetric, 0= perfectly asymmetric
tau = 8;                              %Temperature parameter for RPE function
gamma = 0;                            %Associative updating parameter
gamma_2= 0.9;
beta = 0.95;                           %Temperature parameter for softmax function
alpha_1 = 0;                           %PostPhase2 bais towards rule 1
alpha_2 = 0;                           %PostPhase2 bias towards rule 1 involving 6 or 7.
alpha_3 = 0;                           %PostPhase2 bias towards rule 1 involving 6 AND 7.

v1 = init_scale*randn(1, obj_num);     %Create two vectors that store the value/ranking of our objects
v2 = init_scale*randn(1, obj_num);

t1 = 200;                         %Trial numbers for phase 1
t2 = 100;                          %Trial numbers for phase 2


%phase_1_teach: [a,b,c] - a and b stand for which objects to compare, and c=1 if a>b and c=2 if a<b.
phase_1_teach = cat(2,[[2:obj_num] ; [1:obj_num-1]; ones(1,obj_num-1)], [[1:obj_num-1] ; [2:obj_num]; 2*ones(1,obj_num-1)])';

%% Phase 1 training
for t=1:t1
    teach_data = phase_1_teach(randperm(size(phase_1_teach,1)),:);
    for p=1:length(phase_1_teach)
        [p_x1, p_x2] = softmax_func([v1(teach_data(p,1)),v1(teach_data(p,2))], beta);
        Delta = RPE_sigmoid([p_x1, p_x2], teach_data(p,3), tau);
       if teach_data(p,3) == 1
           v1(v1> v1(teach_data(p,1)))= v1(v1> v1(teach_data(p,1))) + lambda*Delta*gamma;   %Associative value updating
           v1(v1< v1(teach_data(p,2)))= v1(v1< v1(teach_data(p,2))) - lambda*Delta*gamma;
           
           v1(teach_data(p,1)) = v1(teach_data(p,1)) + lambda*Delta;                            %Compaired value updating
           v1(teach_data(p,2)) = v1(teach_data(p,2)) - asym_scale*lambda*Delta;
       end
       if teach_data(p,3) == 2
           v1(v1< v1(teach_data(p,1)))= v1(v1< v1(teach_data(p,1))) - lambda*Delta*gamma;
           v1(v1> v1(teach_data(p,2)))= v1(v1> v1(teach_data(p,2))) + lambda*Delta*gamma;  
           
           v1(teach_data(p,1)) = v1(teach_data(p,1)) - asym_scale*lambda*Delta;          
           v1(teach_data(p,2)) = v1(teach_data(p,2)) + lambda*Delta;
       end
    end
       
end
%Phase 1 training group 2
for t=1:t1
    teach_data = phase_1_teach(randperm(size(phase_1_teach,1)),:);
    for p=1:length(phase_1_teach)
        [p_x1, p_x2] = softmax_func([v2(teach_data(p,1)),v2(teach_data(p,2))], beta);
        Delta = RPE_sigmoid([p_x1, p_x2], teach_data(p,3), tau);
       if teach_data(p,3) == 1
           v2(v2> v2(teach_data(p,1)))= v2(v2> v2(teach_data(p,1))) + lambda*Delta*gamma;   %Associative value updating
           v2(v2< v2(teach_data(p,2)))= v2(v2< v2(teach_data(p,2))) - lambda*Delta*gamma;
           
           v2(teach_data(p,1)) = v2(teach_data(p,1)) + lambda*Delta;                            %Compaired value updating
           v2(teach_data(p,2)) = v2(teach_data(p,2)) - asym_scale*lambda*Delta;
       end
       if teach_data(p,3) == 2
           v2(v2< v2(teach_data(p,1)))= v2(v2< v2(teach_data(p,1))) - lambda*Delta*gamma;
           v2(v2> v2(teach_data(p,2)))= v2(v2> v2(teach_data(p,2))) + lambda*Delta*gamma;  
           
           v2(teach_data(p,1)) = v2(teach_data(p,1)) - asym_scale*lambda*Delta;          
           v2(teach_data(p,2)) = v2(teach_data(p,2)) + lambda*Delta;
       end
    end
       
end

phase_1_v1 = v1;
phase_1_v2 = v2;
v3_rule1 = cat(2,phase_1_v1 , phase_1_v2);
phase_1_decision_matrix = zeros(obj_num*2, obj_num*2);
for i=1:obj_num*2
    for j=1:obj_num*2
        [p_x1, p_x2] = softmax_func([v3_rule1(i),v3_rule1(j)], beta);
        phase_1_decision_matrix(i, j) = p_x2;
    end
end

figure(1)
imagesc(phase_1_decision_matrix)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

%% Phase 2 boundary training 
phase_2_teach = [[obj_num+1, obj_num, 1]; [obj_num, obj_num+1, 2]];
%phase_2_teach = [[obj_num, 1, 2]; [1, obj_num, 1]];
%v3_rule2 = init_scale*randn(2*obj_num,1);
v3_rule2 = gamma_2*v3_rule1;
v3_rule2([obj_num, obj_num+1]) = [v3_rule1(obj_num), v3_rule1(obj_num+1)]

for t=1:t2
    teach_data = phase_2_teach(randperm(size(phase_2_teach,1)),:);
    for p=1:2
            if teach_data(p,3) == 1
               [p_x1, p_x2] = softmax_func([v3_rule2(teach_data(p,1)),v3_rule2(teach_data(p,2))], beta);
               Delta = RPE_sigmoid([p_x1, p_x2], teach_data(p,3), tau);
               
               v3_rule2(obj_num+2:2*obj_num)= v3_rule2(obj_num+2:2*obj_num) + lambda*Delta*gamma_2;   %Associative value updating
               v3_rule2(1:obj_num-1)= v3_rule2(1:obj_num-1) - lambda*Delta*gamma_2;

               v3_rule2(teach_data(p,1)) = v3_rule2(teach_data(p,1)) + lambda*Delta;                            %Compaired value updating
               v3_rule2(teach_data(p,2)) = v3_rule2(teach_data(p,2)) - lambda*Delta;
            end
            if teach_data(p,3) == 2
               [p_x1, p_x2] = softmax_func([v3_rule2(teach_data(p,1)),v3_rule2(teach_data(p,2))], beta);
               Delta = RPE_sigmoid([p_x1, p_x2], teach_data(p,3), tau);
               
               v3_rule2(obj_num+2:2*obj_num)= v3_rule2(obj_num+2:2*obj_num) + lambda*Delta*gamma_2;   %Associative value updating
               v3_rule2(1:obj_num-1)= v3_rule2(1:obj_num-1) - lambda*Delta*gamma_2; 

               v3_rule2(teach_data(p,1)) = v3_rule2(teach_data(p,1)) - lambda*Delta;          
               v3_rule2(teach_data(p,2)) = v3_rule2(teach_data(p,2)) + lambda*Delta;
            end
    end
end

phase_2_decision_matrix = zeros(obj_num*2, obj_num*2);
for i=1:obj_num*2
    for j=1:obj_num*2
        [p_x1_rule1, p_x2_rule1] = softmax_func([v3_rule1(i),v3_rule1(j)], beta);
        [p_x1_rule2, p_x2_rule2] = softmax_func([v3_rule2(i),v3_rule2(j)], beta);
        %All patterns involving 6 and 7
        phase_2_decision_matrix(i, j) = alpha_1*p_x2_rule1+ (1-alpha_1)*p_x2_rule2;
        if i==obj_num || j==(obj_num)
            phase_2_decision_matrix(i, j) = alpha_2*p_x2_rule1+ (1-alpha_2)*p_x2_rule2;
        end
        if i==obj_num+1 || j==(obj_num+1)
            phase_2_decision_matrix(i, j) = alpha_2*p_x2_rule1+ (1-alpha_2)*p_x2_rule2;
        end
       %(6,7)
        if i==obj_num && j==(obj_num+1) 
            phase_2_decision_matrix(i, j) = alpha_3*p_x2_rule1+ (1-alpha_3)*p_x2_rule2;
        end
       %(7,6)
        if i==obj_num+1 && j==(obj_num) 
            phase_2_decision_matrix(i, j) = alpha_3*p_x2_rule1+ (1-alpha_3)*p_x2_rule2;
        end      
    end
    
end

figure(2)
imagesc(phase_2_decision_matrix)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

%% Phase 3 boundary training 
phase_3_teach = [[obj_num+2, obj_num-1, 1]; [obj_num-1, obj_num+2, 2]];
v3_rule3 = gamma_2*v3_rule1;
v3_rule3([obj_num, obj_num+1]) = [v3_rule1(obj_num), v3_rule1(obj_num+1)];

for t=1:t2
    teach_data = phase_3_teach(randperm(size(phase_3_teach,1)),:);
    for p=1:2
            if teach_data(p,3) == 1
               [p_x1, p_x2] = softmax_func([v3_rule3(teach_data(p,1)),v3_rule3(teach_data(p,2))], beta);
               Delta = RPE_sigmoid([p_x1, p_x2], teach_data(p,3), tau);
               
               v3_rule3(obj_num+3:2*obj_num)= v3_rule3(obj_num+3:2*obj_num) + lambda*Delta*gamma_2;   %Associative value updating
               v3_rule3(obj_num+1)= v3_rule3(obj_num+1) + lambda*Delta*gamma_2;
               v3_rule3(1:obj_num-2)= v3_rule3(1:obj_num-2) - lambda*Delta*gamma_2;
               v3_rule3(obj_num)= v3_rule3(obj_num) - lambda*Delta*gamma_2;

               v3_rule3(teach_data(p,1)) = v3_rule3(teach_data(p,1)) + lambda*Delta;                            %Compaired value updating
               v3_rule3(teach_data(p,2)) = v3_rule3(teach_data(p,2)) - lambda*Delta;
            end
            if teach_data(p,3) == 2
               [p_x1, p_x2] = softmax_func([v3_rule3(teach_data(p,1)),v3_rule3(teach_data(p,2))], beta);
               Delta = RPE_sigmoid([p_x1, p_x2], teach_data(p,3), tau);
               
               v3_rule3(obj_num+3:2*obj_num)= v3_rule3(obj_num+3:2*obj_num) + lambda*Delta*gamma_2;   %Associative value updating
               v3_rule3(obj_num+1)= v3_rule3(obj_num+1) + lambda*Delta*gamma_2;
               v3_rule3(1:obj_num-2)= v3_rule3(1:obj_num-2) - lambda*Delta*gamma_2;
               v3_rule3(obj_num)= v3_rule3(obj_num) - lambda*Delta*gamma_2; 

               v3_rule3(teach_data(p,1)) = v3_rule3(teach_data(p,1)) - lambda*Delta;          
               v3_rule3(teach_data(p,2)) = v3_rule3(teach_data(p,2)) + lambda*Delta;
            end
    end
end

phase_3_decision_matrix = zeros(obj_num*2, obj_num*2);
for i=1:obj_num*2
    for j=1:obj_num*2
        [p_x1_rule1, p_x2_rule1] = softmax_func([v3_rule1(i),v3_rule1(j)], beta);
        [p_x1_rule3, p_x2_rule3] = softmax_func([v3_rule3(i),v3_rule3(j)], beta);
        %All patterns involving 6 and 7
        phase_3_decision_matrix(i, j) = alpha_1*p_x2_rule1+ (1-alpha_1)*p_x2_rule3;
        if i==obj_num || j==(obj_num)
            phase_3_decision_matrix(i, j) = alpha_2*p_x2_rule1+ (1-alpha_2)*p_x2_rule3;
        end
        if i==obj_num+1 || j==(obj_num+1)
            phase_3_decision_matrix(i, j) = alpha_2*p_x2_rule1+ (1-alpha_2)*p_x2_rule3;
        end
       %(6,7)
        if i==obj_num && j==(obj_num+1) 
            phase_3_decision_matrix(i, j) = alpha_3*p_x2_rule1+ (1-alpha_3)*p_x2_rule3;
        end
       %(7,6)
        if i==obj_num+1 && j==(obj_num) 
            phase_3_decision_matrix(i, j) = alpha_3*p_x2_rule1+ (1-alpha_3)*p_x2_rule3;
        end      
    end
    
end
figure(3)
imagesc(phase_3_decision_matrix)
xlabel('Object', 'Fontsize', 15)
ylabel('Object', 'Fontsize', 15)
pbaspect([1 1 1])
xticks([1:12])
yticks([1:12])
h = colorbar;
ylabel(h, 'Probability', 'Fontsize', 15)

%%
figure(4)
kk= 7;
phase_3_plot = phase_3_decision_matrix(:,kk);
phase_3_plot(kk) = NaN;
plot([1:12], phase_3_plot, '.-', 'Linewidth', 1.5,'MarkerSize',20 )
xlim([1 12])
xlabel('Object', 'Fontsize', 15)
ylabel('Reaction Time (Cycles)', 'Fontsize', 15)