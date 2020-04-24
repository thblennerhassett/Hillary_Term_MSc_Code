function [out] = hedged_softmax_func(X, tau, C)
%Use C=0 for standard softmax function
denominator = C^(1/tau);
for i=1:length(X)
    denominator = denominator + exp(X(i)/tau);
end
out = exp(X/tau)./denominator;
end