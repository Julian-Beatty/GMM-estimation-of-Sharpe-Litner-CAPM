%%------------------------------------------------------------Objective----------------------------------------------------------------------------%%%%%
%%In this file I estimate the Sharpe Litner CAPM using Generalized Method of Moments, by hand.
%Compute the number of observations in the time, and asset dimensions. For our data this is 956, and 25 respectively.
global z_i z_m omega T N
returns_matrix=returns_matrix(1:100,:)
market_return=market_return(1:100)
RF=RF(1:100)

T=size(returns_matrix,1);
N=size(returns_matrix,2);
%For convenience I compute excess returns on the market portfolio, and test portfolios. Denoted z_m and z_i as in Kris slides.
for i=1:N
    risk_free(:,i)=RF;
end
z_i=returns_matrix-risk_free;
z_m=market_return
u_m=mean(z_m)
var_m=var(z_m)

results=zeros(25,2)
var_cov_matrices=zeros(2,2,25)

for p=1:25  %%loops the GMM procedure for each portfolio%%
z_i=returns_matrix(:,p)-risk_free(:,p)





%%%----------------------------------------------------First stage GMM---------------------------------------------------------
[first_stage_gmm_estimates]=fminunc(@norm_moments,[1;1])  %%returns first stage, consistent estimates of alpha and beta, using 
%identity matrix as W

%%%----------------------------------------------------Computes the Gradient of the moment matrix-----------------------------%%%
gradients = zeros(2,2,T)
%%Thankfully since our model is linear, I have taken partial derivatives/gradient of each moment condition. 
%%the first column in gradients is the (column) gradient of moment 1, starting with alpha in row 1 and beta in row 2. Second column is for the 2nd moment
for i=1:T
    gradients(1,1,i)=1
    gradients(2,1,i)=z_m(i,1)
    gradients(1,2,i)=z_m(i,1)
    gradients(2,2,i)=(z_m(i,1).^2)
end
G=(1/T)*sum(gradients,3)  %%%G matrix

%%---------------------------------------------------Computes the Spectral density matrix (Omega or S)----------------------------------%%
for i=1:T
    moment_matrix(1,1)=z_i(i,1)-first_stage_gmm_estimates(1,1)-z_m(i,1)*first_stage_gmm_estimates(2,1)
    moment_matrix(2,1)=(z_i(i,1)-first_stage_gmm_estimates(1,1)-z_m(i,1)*first_stage_gmm_estimates(2,1))*z_m(i,1)
    omega_stack(:,:,i)=moment_matrix*moment_matrix'
end

omega=sum(omega_stack,3)*(1/T)
%%--------------------------------------------------Computing standard i.i.d errors-------------------------------------------------------------------
%%using simple OLS formula
residual_vector=z_i-first_stage_gmm_estimates(1,1)-z_m*first_stage_gmm_estimates(2,1)

s_square=(residual_vector'*residual_vector)/(T-2)
X=[ones(T,1) z_m]
iid_variance_covariance_matrix=s_square*inv(X'*X)
alpha_iid_error(p,1)=sqrt(iid_variance_covariance_matrix(1,1))
beta_iid_error(p,1)=sqrt(iid_variance_covariance_matrix(2,2))

%%--------------------------------------------------Stores variance, and estimates of first stage estimators (i.i.d error) for each portfolio------------------------------------
%first_stage_var_cov_matrix(:,:,p)=inv(G'*eye(2)*G)*G'*eye(2)*omega*eye(2)*G*inv(G'*eye(2)*G)
%first_stage_iid_error_alpha(p,1)=sqrt(first_stage_var_cov_matrix(1,1,p))/sqrt(956)
%first_stage_iid_error_beta(p,1)=sqrt(first_stage_var_cov_matrix(2,2,p))/sqrt(956)

first_stage_gmm_parameters(p,1:2)=transpose(first_stage_gmm_estimates)
first_stage_gmm_alpha(p,1)=first_stage_gmm_estimates(1,1)
first_stage_gmm_beta(p,1)=first_stage_gmm_estimates(2,1)


%%--------------------------------------------------Second Stage----------------------------------------------------------%%%
%% performs minimization of quadratic form of moment conditions, but with inverse spectral density/omega matrix as the W
[second_stage_gmm_estimates]=fminunc(@second_norm_moments,[1;1])  


%%-----------------------------------------------Stores variance, robust errors, and estimates in 2nd stage GMM for each portfolio%%%
second_stage_var_cov_matrix(:,:,p)=inv(G'*(inv(omega))*G)
second_stage_robust_error_alpha(p,1)=sqrt(second_stage_var_cov_matrix(1,1,p))/sqrt(T-2)
second_stage_robust_error_beta(p,1)=sqrt(second_stage_var_cov_matrix(2,2,p))/sqrt(T-2)


second_stage_gmm_parameters(p,:)=transpose(second_stage_gmm_estimates)
second_stage_gmm_alpha(p,1)=second_stage_gmm_estimates(1,1)
second_stage_gmm_beta(p,1)=second_stage_gmm_estimates(2,1)

end
%%-------------------------------------------Computes the wald statistic to test if alphas are jointly zero--------------------------%%
u=transpose(returns_matrix-risk_free);

for i=1:T
    second_gmm_residual=u(:,i)-second_stage_gmm_alpha-second_stage_gmm_beta*z_m(i)
    second_gmm_cov_stack_residual(:,:)=second_gmm_residual*second_gmm_residual'
end
second_gmm_cov_residual=(1/T)*sum(second_gmm_cov_stack_residual,3)
gmm_chi_squared=T*inv((1+(u_m/var_m)^2))*second_stage_gmm_alpha'*second_gmm_cov_residual*second_stage_gmm_alpha

%%Large Chi square stat, P value==0, we reject Sharpe Litner. CAPM







function m=norm_moments(theta)
    global z_m z_i T
    e=z_i(:,1)-ones(T,1)*theta(1,1)-z_m(:,1)*theta(2,1);
    g(1,1)=mean(e);
    g(2,1)=mean(e.*z_m(:,1));
    m = g'*eye(2)*g
end

function m=second_norm_moments(theta)
    global z_m z_i omega T
    e=z_i(:,1)-ones(T,1)*theta(1,1)-z_m(:,1)*theta(2,1);
    g(1,1)=mean(e);
    g(2,1)=mean(e.*z_m(:,1));
    m = g'*(inv(omega))*g
end