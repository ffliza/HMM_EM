clear all;
%% generate training sequence using matlab function utilizing prior knowledge of the process

T_m = [0.95 0.05;
       0.05 0.95];
 
E_m = [1/6 1/6 1/6 1/6 1/6 1/6;
       1/10 1/10 1/10 1/10 1/10 1/2];

[seq_m,states_m] = hmmgenerate(1000,T_m,E_m);

%% training data

os = seq_m;           %hmmgenerate sequence   
s = num2cell(os,2);   %utilizing cell so that extension can be easier ex : seq_test = s{1}
n = 2;                %No hidden state 
m = 6;                %Max num observed in the sequence
[P_ini,T_ini,E_ini] = Random_Initialization(n,m);   %Random initialization of the transition matrix and emmision matrix

%% using implemented functions with random initialization (no prior belief)

[P_MF,T_MF_est_r,E_MF_est_r] = HMM_PL_EM(s,P_ini,T_ini,E_ini);
T_MF_est_r

%% using implemented functions (with prior belief)

[P_ini,T_ini,E_ini] = Random_Initialization(n,m);
[P_MF,T_MF_est,E_MF_est] = HMM_PL_EM(s,P_ini,T_m, E_m);
T_MF_est

%% using matlab functions with random initialization (no prior belief)

[T_MatLab_r, E_MatLab_r] = hmmtrain(s, T_ini,E_ini);
T_MatLab_r

%% using matlab functions (with prior belief)
[T_MatLab, E_MatLab] = hmmtrain(s, T_m, E_m);
T_MatLab

%% result dependence on initialization
for i = 1: 10
    [P_ini,T_ini,E_ini] = Random_Initialization(n,m); 
    [P_MF,T_MF_est_r,E_MF_est_r] = HMM_PL_EM(s,P_ini,T_ini,E_ini);
    [T_MatLab_r, E_MatLab_r] = hmmtrain(s, T_ini,E_ini);
    t_p(i) = T_MF_est_r(1,1);
    t_p_mat(i) = T_MatLab_r(1,1);
end

x = 0.1:0.1:1;
plot(x, t_p,'g', x, t_p_mat, 'r-')