function [P,A,B] = Random_Initialization(N,M)
P = rand(N,1)+eps;                %pi-prior belief
P = P/sum(P);       
A = rand(N,N)+(2*eps);            %Transition Matrix : NxN matrix
A = bsxfun(@times,A,1./sum(A,2)); %Element by element binary operation
B = rand(N,M)+(2*eps);            %Emmision Matrix : NxM matrix 
B = bsxfun(@times,B,1./sum(B,2));