function [out] = RPE_linear(X_prob,Y, tau)
%Function calculating a measure of the RPE, or 'surprise' at the outcome.
%X_prob: 1x2 vector with model choice probabilites for each object.
%Y: is a 1x2 vector containing the teaching signal for each object (e.g [1,0])
[M,I] = max(Y);                       
x = X_prob(I);
out = 1-x;
end

