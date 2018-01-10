function [beta] = BackwardProcedure(O,A,B,scale_alpha)
[N M] = size(B);
T = length(O);
beta = zeros(N,T);
beta(:,T) = ones(N,1);                           %eq:24
for t = T-1:-1:1
    beta(:,t) = A*(B(:,O(t+1)).*beta(:,t+1));    %eq:25
    beta(:,t) = beta(:,t) * scale_alpha(t);
end