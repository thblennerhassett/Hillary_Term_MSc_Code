function [out] = sigmoid_func(X, tau)
%Function calculating a measure of the RPE, or 'surprise' at the outcome.
%X_prob: 1x2 vector with model choice probabilites for each object.
%Y: is a 1x2 vector containing the teaching signal for each object (e.g [1,0])
%Tau is scaling parameter of sigmoid function
out = 1./(1+exp(-X./tau));
end