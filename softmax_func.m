function [out1, out2] = softmax_func(X, beta)
out1 = exp(beta*X(1))/(exp(beta*X(1))+exp(beta*X(2)));
out2 = exp(beta*X(2))/(exp(beta*X(1))+exp(beta*X(2)));
end

