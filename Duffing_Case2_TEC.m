% This code produces the TEC-twin response of a SDOF Duffing system with
% increasing input amplitude.
%
% Physics-based model: ABC
% Data-based model: NARX-NN

clear all
clc

% Sampling parameters
omega=1;                % Input frequency (rad/sec)
Fs=1000*omega/2/pi;     % Sampling frequency, (samples/cycle*rad/second*cycle/rad = samples/sec)            
T=1/Fs;                 % Sampling period (sec) 
L=1000*10;              % Length of signal (samples/cycle*no.cycles = samples)
time=(0:L-1)*T;         % Time vector (sec)

% Input and initial conditions
A=50+0.15*time.^2;
F=A.*cos(omega*time);
y_initial=[0;0];

% Simulate Duffing system response data
m_d=10;
c1_d=15;
k1_d=10;
k3_d=0.1*ones(L,1);
[t,y]=ode45(@(t,y) duffing(m_d,c1_d,k1_d,k3_d,time,F,t,y),time,y_initial);
X_D=y(:,1);        
V_D=gradient(X_D)/T;  
A_D=gradient(V_D)/T;  
    
% Training and validation sets (maintain even number of points in segments)
ind_tr=1:2:L;
ind_val=2:2:L;
F_tr=F(ind_tr);
F_val=F(ind_val);
X_D_tr=X_D(ind_tr); 
X_D_val=X_D(ind_val);
V_D_tr=V_D(ind_tr); 
V_D_val=V_D(ind_val);
A_D_tr=A_D(ind_tr); 
A_D_val=A_D(ind_val);
time_trval=T*(ind_tr-1);

% Number of segments
sec=8;
sec_length=L/sec;       % length of segment
translate=sec_length;   % how much segment "translates" by

% ABC parameters
N=1000;                     % number of particles
prior_min=[10 15 10 0 0;    % mean (a) of normal (gamma) prior
           10 15 10 1 0;
           10 15 10 0 1];       
prior_max=[1 1 1 0 0;       % var (b) of normal (gamma) prior
           1 1 1 0.1 0;
           1 1 1 0 0.1]; 
mods=size(prior_min,1);                 % number of model options
num_pars=size(prior_min,2);             % number of parameters
population=zeros(N,num_pars,sec);
models=zeros(N,sec);

% Create vectors/matrices for storing values
plot_rho_val=zeros(L,1);
plot_rho_val_new=zeros(L,1);
plot_rho_tr=zeros(L,1);
plot_rho_tr_new=zeros(L,1);
plot_entropy_tr=zeros(L,1);
plot_entropy_val=zeros(L,1);
plot_entropy_PBM_tr=zeros(L,1);
plot_entropy_PBM_val=zeros(L,1);
plot_entropy_DBM_tr=zeros(L,1);
plot_entropy_DBM_val=zeros(L,1);
plot_entropy_DT_tr=zeros(L,1);
plot_entropy_DT_val=zeros(L,1);
mod_probs=zeros(sec,3);
PBMparameters=zeros(5,length(1:translate:L-sec_length+1));
PBM_MSE_tr=zeros(length(1:translate:L-sec_length+1),1);
DBM_MSE_tr=zeros(length(1:translate:L-sec_length+1),1);
DT_MSE_tr=zeros(length(1:translate:L-sec_length+1),1);
PBM_MSE_val=zeros(length(1:translate:L-sec_length+1),1);
DBM_MSE_val=zeros(length(1:translate:L-sec_length+1),1);
DT_MSE_val=zeros(length(1:translate:L-sec_length+1),1);

for i=1:translate:L-sec_length+1
    
    % Find end of current segment
    if L-(i+sec_length-1)<translate
        sec_end=L;
    else
        sec_end=i+sec_length-1;
    end
    
    % Find current segment of data
    x_d_tr=X_D_tr((i-1)/2+1:sec_end/2,:);
    v_d_tr=V_D_tr((i-1)/2+1:sec_end/2,:); 
    a_d_tr=A_D_tr((i-1)/2+1:sec_end/2,:); 
    f_tr=F_tr(:,(i-1)/2+1:sec_end/2); 
    x_d_val=X_D_val((i-1)/2+1:sec_end/2,:);
    v_d_val=V_D_val((i-1)/2+1:sec_end/2,:); 
    a_d_val=A_D_val((i-1)/2+1:sec_end/2,:); 
    f_val=F_val(:,(i-1)/2+1:sec_end/2); 

    % PHYSICS BASED MODEL 
    %=====================================================================
    data=x_d_tr;
    epsilon=0.01*((i-1)/sec_length+1)^2;    % ABC threshold
    theta=zeros(num_pars,1);
    w=zeros(N,1);
    count=0;
    progress=N/10:N/10:N;           % for showing progress of algorithm
    while count<N
        
        % sample model and parameters from priors
        mod_opts=1:mods;
        model=mod_opts(randsample(length(mod_opts),1));
        for j=1:num_pars
                if j<4
                    theta(j)=normrnd(prior_min(model,j),sqrt(prior_max(model,j)));
                else
                    theta(j)=gamrnd(prior_min(model,j),prior_max(model,j));
                end
        end
        
        % simulate sampled model
        try
            [t,y]=ode45(@(t,y) nonlinear(theta(1),theta(2),0,0,theta(3),theta(4),theta(5),time_trval(1:sec_end/2),F_tr(1:sec_end/2),t,y),time_trval(1:sec_end/2),y_initial);
            x_sim=y(round(i/2):sec_end/2,1);
            MSE=sum((x_d_tr-x_sim).^2)/length(x_sim);
            if MSE<epsilon
                if isempty(intersect(count,progress))==0
                    fprintf('%d percent done...',intersect(count,progress)/N*100)
                end
                count=count+1;
                models(count,(i+translate-1)/translate)=model;
                population(count,:,(i+translate-1)/translate)=theta;
            end
        catch
        end
    end
    
    % Find posterior model probabilities
    for k=1:mods
        mod_probs((i+translate-1)/translate,k)=length(find(models(:,(i+translate-1)/translate)==k))/N; 
    end
    figure
    bar([1:mods],mod_probs((i+translate-1)/translate,:)) 
    
    % Find most probable model
    best_model=mod_opts(find(mod_probs((i+translate-1)/translate,:)==max(mod_probs((i+translate-1)/translate,:))));
    
    % Plot a histogram of accepted particles
    for k=1:mods
        
        % Find the parameters used in model k
        ind=1:num_pars;
        null_pars=intersect(find(prior_min(k,:)==0),find(prior_max(k,:)==0));
        if isempty(null_pars)==0
            for l=1:length(null_pars)
                ind=ind(ind~=null_pars(l));
            end
        end
        
        % Save most probable parameters of best model as PBM parameters
        if k==best_model
            PBM=zeros(num_pars,1);
            for j=ind
                if j<4
                    h=figure;
                    histfit(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),40)
                    title(['Model ',num2str(k)])
                    pd = fitdist(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),'Normal');
                    title(sprintf('Mean = %.4f, Std = %.4f',pd.mu,pd.sigma),'fontsize',30)
                    ylabel('Frequency','fontsize',30)
                    if j==1
                        xlabel('Value (kg)','fontsize',30)
                    elseif j==2
                        xlabel('Value (Ns/m)','fontsize',30)
                    else
                        xlabel('Value (N/m)','fontsize',30)
                    end
                    a = get(gca,'XTickLabel');
                    set(gca,'XTickLabel',a,'fontsize',30);
                    PBM(j)=pd.mu;
                else
                    h=figure;
                    histfit(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),40,'Gamma')
                    title(['Model ',num2str(k)])
                    pd = fitdist(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),'gamma');
                    title(sprintf('a = %.4f, b = %.4f',pd.a,pd.b),'fontsize',30)
                    ylabel('Frequency','fontsize',30)
                    if k==2
                        xlabel('Value (N/m^2)','fontsize',30)
                    else
                        xlabel('Value (N/m^3)','fontsize',30)
                    end
                    a = get(gca,'XTickLabel');
                    set(gca,'XTickLabel',a,'fontsize',30);
                    PBM(j)=(pd.a-1)*pd.b;
                end
            end
        end
    end
    PBMparameters(:,(i+translate-1)/translate)=PBM;
    
    % Find responses of PBM
    
    % Training
    [t,y]=ode45(@(t,y) nonlinear(PBM(1),PBM(2),0,0,PBM(3),PBM(4),PBM(5),time_trval,F_tr,t,y),time_trval,y_initial);
    x_p_tr=y(round(i/2):round(sec_end/2),1);
    
    % Validation
    [t,y]=ode45(@(t,y) nonlinear(PBM(1),PBM(2),0,0,PBM(3),PBM(4),PBM(5),time_trval,F_val,t,y),time_trval,y_initial);
    x_p_val=y(round(i/2):round(sec_end/2),1);
    
    % Store responses of PBM
    if i==1
        X_P_tr=x_p_tr(1:translate/2);
        X_P_val=x_p_val(1:translate/2);
    else
        if L-(i+sec_length-1)<translate                
            X_P_tr=[X_P_tr; x_p_tr(1:end)];
            X_P_val=[X_P_val; x_p_val(1:end)];
        else     
            X_P_tr=[X_P_tr; x_p_tr(1:translate/2)];
            X_P_val=[X_P_val; x_p_val(1:translate/2)];
        end
    end
    %=====================================================================
    
    % DATA BASED MODEL
    %=====================================================================
    f_tr=f_tr';
    f_val=f_val';
    lags=0;             % number of max lags in NARX model
    NMSE_DBM_val=2;     % Set to > NMSE limit
    
    % Increase the lags by 1 for new model
    while NMSE_DBM_val>1    % NMSE limit
        lags=lags+1;
        
        % Training inputs/outputs for neural network:
        response_mat=x_d_tr(1:length(x_d_tr)-lags);
        for j=2:lags
            response_mat=[x_d_tr(j:length(x_d_tr)-lags+j-1) response_mat]; 
        end
        input_mat=f_tr(1:length(f_tr)-lags);
        for j=2:lags+1
            input_mat=[f_tr(j:length(f_tr)-lags+j-1) input_mat]; 
        end
        X_tr=[response_mat input_mat];
        Y_tr=[x_d_tr(lags+1:length(x_d_tr))]; 
        
        % Scale training data
        mins=zeros(1,size(X_tr,2));
        maxs=zeros(1,size(X_tr,2));
        for j=1:size(X_tr,2)
            mins(j)=min(X_tr(:,j));
            maxs(j)=max(X_tr(:,j));
            X_tr(:,j)=(X_tr(:,j)-min(X_tr(:,j)))/(max(X_tr(:,j))-min(X_tr(:,j)));
        end
        
        % Train neural network
        net.performParam.regularization = 0;
        net.layers{1}.transferFcn='tansig';
        net.layers{2}.transferFcn='purelin';
        net=feedforwardnet([10]);
        [net,tr]=train(net,X_tr',Y_tr');
        
        % Validation inputs/outputs
        response_mat=x_d_val(1:length(x_d_val)-lags);
        for j=2:lags
            response_mat=[x_d_val(j:length(x_d_val)-lags+j-1) response_mat]; 
        end
        input_mat=f_val(1:length(f_val)-lags);
        for j=2:lags+1
            input_mat=[f_val(j:length(f_val)-lags+j-1) input_mat]; 
        end
        X_val=[response_mat input_mat];
        Y_val=[x_d_val(lags+1:length(x_d_val))]; 
        
        % Scale validation data
        for j=1:size(X_val,2)
            X_val(:,j)=(X_val(:,j)-mins(j))./(maxs(j)-mins(j));
        end
        
        % One Step Ahead
        % Training
        MSE_OSA_tr=sum((net(X_tr')-Y_tr').^2)/(length(x_d_tr)-lags);
        NMSE_OSA_tr=sum((net(X_tr')-Y_tr').^2)*100/var(Y_tr)/(length(x_d_tr)-lags);       
        
        % Validation
        MSE_OSA_val=sum((net(X_val')-Y_val').^2)/(length(x_d_val)-lags);
        NMSE_OSA_val=sum((net(X_val')-Y_val').^2)*100/var(Y_val)/(length(x_d_val)-lags);
        
        % Model Predicted Output
        % Training
        Yhat_tr=zeros(length(x_d_tr),1);
        Yhat_tr(1:lags)=x_d_tr(1:lags);
        for j=lags+1:length(x_d_tr)
            pred_matrix=[Yhat_tr(j-1)];
            for k=2:lags
                pred_matrix=[pred_matrix Yhat_tr(j-k)];
            end
            for k=1:lags+1
                pred_matrix=[pred_matrix f_tr(j-k+1)]; 
            end
            pred_matrix_scaled=(pred_matrix-mins)./(maxs-mins);
            Yhat_tr(j)=net(pred_matrix_scaled');
        end
        MSE_MPO_tr=sum((Yhat_tr(lags+1:length(x_d_tr))-Y_tr).^2)/(length(x_d_tr)-lags);
        NMSE_MPO_tr=sum((Yhat_tr(lags+1:length(x_d_tr))-Y_tr).^2)*100/var(Y_tr)/(length(x_d_tr)-lags);
        
        % Validation
        Yhat_val=zeros(length(x_d_val),1);
        Yhat_val(1:lags)=x_d_val(1:lags);
        for j=lags+1:length(x_d_val)
            pred_matrix=[Yhat_val(j-1)];
            for k=2:lags
                pred_matrix=[pred_matrix Yhat_val(j-k)];
            end
            for k=1:lags+1
                pred_matrix=[pred_matrix f_val(j-k+1)]; 
            end
            pred_matrix_scaled=(pred_matrix-mins)./(maxs-mins);
            Yhat_val(j)=net(pred_matrix_scaled');
        end
        MSE_MPO_val=sum((Yhat_val(lags+1:length(x_d_val))-Y_val).^2)/(length(x_d_val)-lags);
        NMSE_MPO_val=sum((Yhat_val(lags+1:length(x_d_val))-Y_val).^2)*100/var(Y_val)/(length(x_d_val)-lags);
        
        % Choose OSA or MPO prediction
        xd_tr=[x_d_tr(1:lags-1);net(X_tr')'];
        xd_val=[x_d_val(1:lags-1);net(X_val')'];
        NMSE_DBM_tr=NMSE_OSA_tr;
        NMSE_DBM_val=NMSE_OSA_val;
        % Comment out if using OSA predictions:
        xd_tr=Yhat_tr;
        xd_val=Yhat_val;
        NMSE_DBM_tr=NMSE_MPO_tr;
        NMSE_DBM_val=NMSE_MPO_val;
    end
    
    % Save DBM responses
    if i==1
        XD_tr=xd_tr(1:translate/2);
        XD_val=xd_val(1:translate/2);
    else
        if L-(i+sec_length-1)<translate                
            XD_tr=[XD_tr; xd_tr(1:end)];
            XD_val=[XD_val; xd_val(1:end)];
        else    
            XD_tr=[XD_tr; xd_tr(1:translate/2)];
            XD_val=[XD_val; xd_val(1:translate/2)];
        end
    end
    %=====================================================================
    
    % TEC-TWIN RESPONSE
    %=====================================================================
    % Training:
    J1sum=0;
    J2sum=0;
    for j=1:length(f_tr)
        J1sum=J1sum+(x_p_tr(j)-x_d_tr(j))^2;
        J2sum=J2sum+(xd_tr(j)-x_d_tr(j))^2;   
    end
    PBM_MSE_tr(i:sec_end,1)=J1sum/length(f_tr);
    DBM_MSE_tr(i:sec_end,1)=J2sum/length(f_tr);
    beta=sqrt(J2sum/J1sum);
    rho_tr=beta/(beta+1);
    plot_rho_tr(i:sec_end,1)=rho_tr;
    x_dt_tr=rho_tr*x_p_tr+(1-rho_tr)*xd_tr;
    J3sum=0;
    for c=1:length(f_tr)
       J3sum=J3sum+(x_dt_tr(c)-x_d_tr(c))^2;
    end
    
    % Update rho and TEC-twin response if needed
    if J3sum>J1sum || J3sum>J2sum
        if J1sum<J2sum
            rho_tr=1;
        else
            rho_tr=0;
        end 
        x_dt_tr=rho_tr*x_p_tr+(1-rho_tr)*xd_tr;
    end
    plot_rho_tr_new(i:sec_end,1)=rho_tr;   
    
    % Validation:
    J1sum=0;
    J2sum=0;
    totalJsum=0;
    for j=1:length(f_val)
        J1sum=J1sum+(x_p_val(j)-x_d_val(j))^2;
        J2sum=J2sum+(xd_val(j)-x_d_val(j))^2; 
    end
    PBM_MSE_val(i:sec_end,1)=J1sum/length(f_val);
    DBM_MSE_val(i:sec_end,1)=J2sum/length(f_val);
    beta=sqrt(J2sum/J1sum);
    rho_val=beta/(beta+1);
    plot_rho_val(i:sec_end,1)=rho_val;
    x_dt_val=rho_val*x_p_val+(1-rho_val)*xd_val;
    J3sum=0;
    for c=1:length(f_val)
       J3sum=J3sum+(x_dt_val(c)-x_d_val(c))^2;
    end
    
    % Update rho and TEC-twin response if needed
    if J3sum>J1sum || J3sum>J2sum
        if J1sum<J2sum
            rho_val=1;
        else
            rho_val=0;
        end 
        x_dt_val=rho_val*x_p_val+(1-rho_val)*xd_val;
    end
    plot_rho_val_new(i:sec_end,1)=rho_val;   
    
    % Save TEC-twin response
    if i==1
        X_DT_tr=x_dt_tr(1:translate/2);
        X_DT_val=x_dt_val(1:translate/2);
    else
        if L-(i+sec_length-1)<translate                
            X_DT_tr=[X_DT_tr; x_dt_tr(1:end)];
            X_DT_val=[X_DT_val; x_dt_val(1:end)];
        else    
            X_DT_tr=[X_DT_tr; x_dt_tr(1:translate/2)];
            X_DT_val=[X_DT_val; x_dt_val(1:translate/2)];
        end
    end
    
    % Calculate TEC-twin error
    J3sum=0;
    for c=1:length(f_tr)
       J3sum=J3sum+(x_dt_tr(c)-x_d_tr(c))^2;
    end
    DT_MSE_tr(i:sec_end,1)=J3sum/length(f_tr);
    J3sum=0;
    for c=1:length(f_val)
       J3sum=J3sum+(x_dt_val(c)-x_d_val(c))^2;
    end
    DT_MSE_val(i:sec_end,1)=J3sum/length(f_val);
    %=====================================================================
    
    % Entropies
    [n m]=hist(x_d_tr,1000000);
    p=n/sum(n);
    ent_t=-sum(p(p~=0).*log2(p(p~=0)));
    [n m]=hist(x_d_val,1000000);
    p=n/sum(n);
    ent_v=-sum(p(p~=0).*log2(p(p~=0)));
    [n m]=hist(x_p_tr,1000000);
    p=n/sum(n);
    ent_PBM_t=-sum(p(p~=0).*log2(p(p~=0)));
    [n m]=hist(x_p_val,1000000);
    p=n/sum(n);
    ent_PBM_v=-sum(p(p~=0).*log2(p(p~=0)));
    [n m]=hist(xd_tr,1000000);
    p=n/sum(n);
    ent_DBM_t=-sum(p(p~=0).*log2(p(p~=0)));
    [n m]=hist(xd_val,1000000);
    p=n/sum(n);
    ent_DBM_v=-sum(p(p~=0).*log2(p(p~=0)));
    [n m]=hist(x_dt_tr,1000000);
    p=n/sum(n);
    ent_DT_t=-sum(p(p~=0).*log2(p(p~=0)));
    [n m]=hist(x_dt_val,1000000);
    p=n/sum(n);
    ent_DT_v=-sum(p(p~=0).*log2(p(p~=0)));
    
    plot_entropy_tr(i:sec_end)=ent_t;
    plot_entropy_val(i:sec_end)=ent_v;
    plot_entropy_PBM_tr(i:sec_end)=ent_PBM_t;
    plot_entropy_PBM_val(i:sec_end)=ent_PBM_v;
    plot_entropy_DBM_tr(i:sec_end)=ent_DBM_t;
    plot_entropy_DBM_val(i:sec_end)=ent_DBM_v;
    plot_entropy_DT_tr(i:sec_end)=ent_DT_t;
    plot_entropy_DT_val(i:sec_end)=ent_DT_v;
end
last_point=i+sec_length-1;

% PLOT RESULTS
%=========================================================================
% For plotting start and end of segments:
section_y=[-1000 1000];
if sec_length==translate
    section_x=zeros(sec,2);
    for i=1:sec
        section_x(i,:)=[time(L)/sec*i time(L)/sec*i];
    end
else
    section_x=zeros(floor((L-sec_length)/translate)+1,2);
    for i=1:floor((L-sec_length)/translate)
        section_x(i,:)=[time(translate*i) time(translate*i)];
    end
    section_x(end,:)=[time(last_point) time(last_point)];
end

% Plot DBM model probabilties
for i=1:4
    h=figure;
    bar([1:mods],mod_probs(i,:)) 
    title('Marginal Posterior Model Probabilities','fontsize',50)
    xlabel('Model','fontsize',50)
    ylabel('Probability','fontsize',50)
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',50)
    ylim([0 1])
end

% Plot model responses (training):
% figure
% plot(time_trval,X_D_tr,'LineWidth',4.5)
% hold on
% plot(time_trval,X_P_tr,time_trval,XD_tr,time_trval,X_DT_tr,'--','LineWidth',2.5)
% for j=1:size(section_x,1)
%     plot(section_x(j,:),section_y,'k','LineWidth',4.5)
% end
% hold off
% legend('Data','PBM','DBM','TEC-Twin','fontsize',30)
% title('Training Data: Predicted System Responses','fontsize',30)
% xlabel('Time (s)','fontsize',30)
% ylabel('Displacement (m)','fontsize',30)
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',30)
% ylim([-25 25])
% xlim([0 time(end)])

% Plot model responses (validation):
h=figure;
plot(time_trval,X_D_val,'LineWidth',4.5)
hold on
plot(time_trval,X_P_val,time_trval,XD_val,time_trval,X_DT_val,'--','LineWidth',2.5)
for j=1:size(section_x,1)
    plot(section_x(j,:),section_y,'k','LineWidth',4.5)
end
hold off
legend('Data','PBM','DBM','TEC-Twin','fontsize',30)
title('Validation Data: Predicted System Responses','fontsize',30)
xlabel('Time (s)','fontsize',30)
ylabel('Displacement (m)','fontsize',30)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',30)
ylim([-25 25])
xlim([0 time(end)])

% Plot model fractions:
% % Training
% figure
% plot(time,plot_rho_tr,time,plot_rho_tr_new,'--','LineWidth',4.5)
% hold on
% for j=1:size(section_x,1)
%     plot(section_x(j,:),[0 1],'k','LineWidth',4.5)
% end
% hold off
% legend('Original Value','Updated Value')
% title('Training Data: Physics-to-Data Model Fraction vs. Time','fontsize',30)
% xlabel('Time (s)','fontsize',30)
% ylabel('Model Fraction','fontsize',30)
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',30);
% xlim([0 time(end)])

% Validation
h=figure;
plot(time,plot_rho_val,time,plot_rho_val_new,'--','LineWidth',4.5)
hold on
for j=1:size(section_x,1)
    plot(section_x(j,:),[0 1],'k','LineWidth',4.5)
end
hold off
legend('Original Value','Updated Value')
title('Validation Data: Physics-to-Data Model Fraction vs. Time','fontsize',30)
xlabel('Time (s)','fontsize',30)
ylabel('Model Fraction','fontsize',30)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',30);
xlim([0 time(end)])

% Plot nonlinear stiffness of PBM and true value:
h=figure;
plot_k3=zeros(1,L);
for i=1:translate:L-sec_length+1
    plot_k3(:,i:i+translate-1)=[PBMparameters(5,(i+translate-1)/translate)*ones(1,translate)];
end
plot(time,k3_d,time,plot_k3,'LineWidth',4.5)
hold on
for j=1:size(section_x,1)
    plot(section_x(j,:),[0 500],'k','LineWidth',4.5)
end
hold off
legend('True Value','PBM')
title('Cubic Stiffness Parameter vs. Time','fontsize',30)
xlabel('Time (s)','fontsize',30)
ylabel('Cubic Stiffness Parameter (N/m^3)','fontsize',30)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',30);
xlim([0 time(end)])
ylim([0 0.12])

% Plot MSE of models:
% %Training:
% h=figure;
% g=plot([10000000],[10000000])
% g.Annotation.LegendInformation.IconDisplayStyle = 'off';
% hold on
% plot(time,PBM_MSE_tr,time,DBM_MSE_tr,time,DT_MSE_tr,'--','LineWidth',4.5)
% for j=1:size(section_x,1)
%     plot(section_x(j,:),[0 1.6],'k','LineWidth',4.5)
% end
% hold off
% legend('PBM','DBM','TEC-Twin')
% title('Training Data: Model MSE vs. Time','fontsize',30)
% xlabel('Time (s)','fontsize',30)
% ylabel('MSE (m^2)','fontsize',30)
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',30);
% ylim([0 0.07])
% xlim([0 time(end)])

% Validation:
h=figure;
g=plot([10000000],[10000000]);
g.Annotation.LegendInformation.IconDisplayStyle = 'off';
hold on
plot(time,PBM_MSE_val,time,DBM_MSE_val,time,DT_MSE_val,'--','LineWidth',4.5)
for j=1:size(section_x,1)
    plot(section_x(j,:),[0 1.6],'k','LineWidth',4.5)
end
hold off
legend('PBM','DBM','TEC-Twin')
title('Validation Data: Model MSE vs. Time','fontsize',30)
xlabel('Time (s)','fontsize',30)
ylabel('MSE (m^2)','fontsize',30)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',30);
ylim([0 0.07])
xlim([0 time(end)])

h=figure;
f=plot([10000000],[10000000]);
f.Annotation.LegendInformation.IconDisplayStyle = 'off';
hold on
g=plot([10000000],[10000000])
g.Annotation.LegendInformation.IconDisplayStyle = 'off';
plot(time,DBM_MSE_val,time,DT_MSE_val,'--','LineWidth',4.5)
for j=1:size(section_x,1)
    plot(section_x(j,:),[0 1.6],'k','LineWidth',4.5)
end
hold off
legend('DBM','TEC-Twin')
title('Validation Data: Model MSE vs. Time','fontsize',30)
xlabel('Time (s)','fontsize',30)
ylabel('MSE (m^2)','fontsize',30)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',30);
ylim([0 0.0000065])
xlim([0 time(end)])

% Plot input signal:
h=figure;
plot(time,F,'LineWidth',4.5)
% hold on
% for j=1:size(section_x,1)
%     plot(section_x(j,:),[0 1],'k','LineWidth',2)
% end
% hold off
title('Simulated System Input vs. Time','fontsize',30)
xlabel('Time (s)','fontsize',30)
ylabel('Input (N)','fontsize',30)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',30);
xlim([0 time(end)])

% Plot entropies
% % Training
% figure
% plot(time,plot_entropy_tr,'LineWidth',8)
% hold on
% plot(time,plot_entropy_PBM_tr,time,plot_entropy_DBM_tr,time,plot_entropy_DT_tr,'--','LineWidth',4.5)
% for j=1:size(section_x,1)
%     plot(section_x(j,:),[0 10],'k','LineWidth',4.5)
% end
% hold off
% title('Training Data: Shannon Entropy of Predicted Responses vs. Time','fontsize',30)
% xlabel('Time (s)','fontsize',30)
% ylabel('Shannon Entropy (bits)','fontsize',30)
% legend('Data','PBM','DBM','TEC-Twin')
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',30);
% xlim([0 time(end)])
% ylim([9.276 9.294])

% Validation
h=figure;
plot(time,plot_entropy_val,'LineWidth',8)
hold on
plot(time,plot_entropy_PBM_val,time,plot_entropy_DBM_val,time,plot_entropy_DT_val,'--','LineWidth',4.5)
for j=1:size(section_x,1)
    plot(section_x(j,:),[0 10],'k','LineWidth',4.5)
end
hold off
title('Validation Data: Shannon Entropy of Predicted Responses vs. Time','fontsize',30)
xlabel('Time (s)','fontsize',30)
ylabel('Shannon Entropy (bits)','fontsize',30)
legend('Data','PBM','DBM','TEC-Twin')
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',30);
xlim([0 time(end)])
ylim([9.276 9.294])

% For replotting ABC histograms from saved results:
for i=1:translate:L-sec_length+1
    best_model=mod_opts(find(mod_probs((i+translate-1)/translate,:)==max(mod_probs((i+translate-1)/translate,:))));
    k=best_model;
    ind=1:num_pars;
    null_pars=intersect(find(prior_min(k,:)==0),find(prior_max(k,:)==0));
    if isempty(null_pars)==0
        for l=1:length(null_pars)
            ind=ind(ind~=null_pars(l));
        end
    end
    for j=ind
        if j<4
            h=figure;
            histfit(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),40)
            pd = fitdist(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),'Normal');
            title(sprintf('Mean = %.4f, Std = %.4f',pd.mu,pd.sigma),'fontsize',50)
            ylabel('Frequency','fontsize',50)
            if j==1
                xlabel('Value (kg)','fontsize',50)
            elseif j==2
                xlabel('Value (Ns/m)','fontsize',50)
            else
                xlabel('Value (N/m)','fontsize',50)
            end
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'fontsize',50);
        else
            h=figure;
            histfit(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),40,'Gamma')
            pd = fitdist(population(find(models(:,(i+translate-1)/translate)==k),j,(i+translate-1)/translate),'gamma');
            title(sprintf('a = %.4f, b = %.4f',pd.a,pd.b),'fontsize',50)
            ylabel('Frequency','fontsize',50)
            if k==2
                xlabel('Value (N/m^2)','fontsize',50)
            else
                xlabel('Value (N/m^3)','fontsize',50)
            end
            a = get(gca,'XTickLabel');
            set(gca,'XTickLabel',a,'fontsize',50);
        end
    end
end
%=========================================================================