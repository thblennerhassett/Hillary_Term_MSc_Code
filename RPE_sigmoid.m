function [out] = RPE_sigmoid(X_prob,y, tau)
%Function calculating a measure of the RPE, or 'surprise' at the outcome.
%X_prob: 1x2 vector with model choice probabilites for each object.
%Y: is a 1x2 vector containing the teaching signal for each object (e.g [1,0])
%Tau is scaling parameter of sigmoid function
x = X_prob(y);
max_f = 1-1./(1+exp(-tau*(0-0.5)));
min_f = 1-1./(1+exp(-tau*(1-0.5)));
out = (1-1./(1+exp(-tau*(x-0.5)))-min_f)/(max_f-min_f);
end

