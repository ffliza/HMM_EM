function [P,A,B] = HMM_PL_EM(Observation,P,A,B)

% EM for HMM parameter estimation
% Developed by Farhana Ferdousi Liza
% Version:1 
% based on Ranibar Tutorial Paper 
% P : initial probability
% A : transition probability
% B : emmision probability


%% some initializations of the parameter 

max_limit = 1000;                    % setting the max iteration number
log_likelihood = zeros(max_limit,1); % log liklihood of each iterration for convergence check
seqnum = length(Observation);        % number of observation, currently based on one long sequence
conv_limit = 1e-4;                   % setting the error tolerance of the convergence

[N,M] = size(B);                     % retriving dimention

%% the EM algorithm
for i = 1:max_limit
    temp_P = zeros(size(P));
    temp_A = zeros(size(A));
    temp_B = zeros(size(B));
    for seq_id = 1:seqnum
        O = Observation{seq_id};
        T = length(O);
        [alpha,scale_alpha] = ForwardProcedure(O,A,B,P);           %forward
        [beta] = BackwardProcedure(O,A,B,scale_alpha);             %backward
        % compute posterior probabilities--------------------------(E-step)
        [gamma] = compGamma(alpha,beta);                           %eq:27
        % compute averaged joint posterior (q_t=S_i,q_(t+1)=S_j|O)
        shi = zeros(N);
        for t = 1:T-1
            shi_temp = (alpha(:,t) * (beta(:,t+1).*B(:,O(t+1)))') .* A;
            shi = shi + shi_temp / sum(sum(shi_temp));              %eq:37
        end
        % update parameters ----------------------------------------(M-step)
        temp_P = temp_P + gamma(:,1);                               %eq:40a
        temp_A = temp_A + shi;                                      %eq:40b
        for k = 1:M
            temp_B(:,k) = temp_B(:,k) + sum(gamma(:,O==k),2);       %eq:40c
        end
        log_likelihood(i) = log_likelihood(i) - sum(log(scale_alpha));
    end
    P = temp_P / seqnum;
    A = bsxfun(@times,temp_A,1./sum(temp_A,2));
    B = bsxfun(@times,temp_B,1./sum(temp_B,2));
    if i > 2
        log_likelihood_del = abs(1-log_likelihood(i-1)/log_likelihood(i));
        if log_likelihood_del < conv_limit
            break;
        end
    end
end