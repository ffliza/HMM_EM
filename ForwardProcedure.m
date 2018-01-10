function [alpha,scale_alpha] = ForwardProcedure(O,A,B,P)
[N M] = size(B);
T = length(O);
alpha = zeros(N,T);
scale_alpha = zeros(1,T);
alpha(:,1) = P.*B(:,O(1));                      % Initialization eq:19
scale_alpha(1) = 1./sum(alpha(:,1));
for t = 2:T
    alpha(:,t) = A'*alpha(:,t-1).*B(:,O(t));    % Induction eq:20
    scale_alpha(t) = 1./sum(alpha(:,t));
    alpha(:,t) = alpha(:,t) * scale_alpha(t);
end