function [gamma] = compGamma(alpha,beta)
gamma = alpha.*beta;
gamma = bsxfun(@times,gamma,1./sum(gamma));   %eq:27