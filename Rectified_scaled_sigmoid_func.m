function [out] = Rectified_scaled_sigmoid_func(X, tau)
%Function calculating a measure of the RPE, or 'surprise' at the outcome.
%X_prob: 1x2 vector with model choice probabilites for each object.
%Y: is a 1x2 vector containing the teaching signal for each object (e.g [1,0])
%Tau is scaling parameter of sigmoid function
max_f = 1-1./(1+exp(-(0-0.5)/tau));
min_f = 1-1./(1+exp(-(1-0.5)/tau));
out = (1./(1+exp(-(X-0.5)./tau))-min_f)/(max_f-min_f);
out(out<0)=0;
out(out>1)=1;

end