% DIDI project

% N_orig 2961
% N_orig_tr 2369
% N_orig_ts 592

X=importdata('../data/data_chao/didi_train_data.csv');
Y=importdata('../data/data_chao/didi_train_label.csv');
data=[X Y];
N_orig=size(data,1);
data=data(randperm(N_orig),:);
X=data(:,1:199);
Y=data(:,200:end);

%%
ts_X_SM=zeros(592,199,66);
ts_lb_SM=zeros(592,1,66);

Division=6;
tRange=linspace(0,20,Division);
numNoZero=ones(1,Division,66);
weights=ones(199,Division,66);
for j=1:66
    index=find(Y(:,j)); 
    X1=X(index,:);
    Y1=Y(index,j);
    N=length(index);
    
    N_tr=round(N*0.9);
    N_ts=N-N_tr;

    % train
    tr_X=X1(1:N_tr,1:199);
    tr_lb=Y1(1:N_tr);
    % test
    ts_X=X1(N_tr+1:end,1:199);
    ts_lb=Y1(N_tr+1:end);
    % save test
    ts_X_ext=[ts_X;zeros(592-size(ts_X,1),199)];
    ts_lb_ext=[ts_lb;zeros(592-length(ts_lb),1)];  
    ts_X_SM(:,:,j)=ts_X_ext;
    ts_lb_SM(:,:,j)=ts_lb_ext;
    
    for i=1:Division
        tau=tRange(i);
        cvx_begin quiet
        variable w(199)
        minimize( norm( (tr_lb - tr_X*w)./tr_lb ) + tau*norm(w,1) )
        cvx_end

        [WIndx]=find(abs(w) > 1e-5);
        numNoZero(:,i,j)=length(WIndx);
        weights(:,i,j)=w;
    end 
end
%% find best weights
index_weights=ones(1,66);
best_weights=ones(199,66);
costV_tr=zeros(Division,66);
for j=1:66
    for i=1:Division
        len_ts=length(find(ts_lb_SM(:,:,j)));

        ts_X=ts_X_SM(1:len_ts,:,j);
        ts_lb=ts_lb_SM(1:len_ts,:,j);
        residual=(ts_lb - ts_X * weights(:,i,j))./ts_lb;
        costV_tr(i,j)=norm(residual,1)/len_ts;
    end
    index2=find(costV_tr(:,j)==min(costV_tr(:,j)));
    index_weights(j)=index2;
    best_weights(:,j)=weights(:,index2,j);
end
%% cost
costV_ts=zeros(1,66);
for j=1:66
    len_ts=length(find(ts_lb_SM(:,:,j)));

    ts_X=ts_X_SM(1:len_ts,:,j);
    ts_lb=ts_lb_SM(1:len_ts,:,j);
    residual=(ts_lb - ts_X * best_weights(:,j))./ts_lb;
    costV_ts(j)=norm(residual,1)/len_ts;
end
cost=sum(costV_ts)/66;





