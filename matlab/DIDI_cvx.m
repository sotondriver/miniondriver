% DIDI project

% train: 0.7 valid: 0.2 test: 0.1
% N_orig 2961
% N_orig_tr 2073
% N_orig_vl 592
% N_orig_ts 296
%% parameter
dim_X=475;
dim_lb=66;
rate_train=0.7;
rate_valid=0.2;
%% 
X=importdata('../data/data_xuan/didi_train_data_w_t.csv');
Y=importdata('../data/data_chao/didi_train_label.csv');
data=[X Y];
N_orig=size(data,1);
N_orig_tr = round(N_orig*rate_train);
N_orig_vl = round(N_orig*rate_valid);
N_orig_ts = N_orig-N_orig_tr-N_orig_vl;
data=data(randperm(N_orig),:);
X=data(:,1:dim_X);
Y=data(:,dim_X+1:end);

%% train part
vl_X_SM=zeros(N_orig_vl,dim_X,66);
vl_lb_SM=zeros(N_orig_vl,1,66);
ts_X_SM=zeros(N_orig_ts,dim_X,66);
ts_lb_SM=zeros(N_orig_ts,1,66);

Division=3;
tRange=linspace(0,20,Division);
numNoZero=ones(1,Division,66);
weights=ones(dim_X,Division,66);
for j=1:66
    % find nonzero
    index=find(Y(:,j)); 
    X1=X(index,:);
    Y1=Y(index,j);
    N=length(index);
    
    % separate train and test
    N_tr=round(N*rate_train);
    N_vl=round(N*rate_valid);
    N_ts=N-N_tr-N_vl;

    % train
    tr_X=X1(1:N_tr,1:dim_X);
    tr_lb=Y1(1:N_tr);
    
    % valid
    vl_X=X1(N_tr+1:N_tr+N_vl,1:dim_X);
    vl_lb=Y1(N_tr+1:N_tr+N_vl);  
    % save valid
    vl_X_ext=[vl_X;zeros(N_orig_vl-size(vl_X,1),dim_X)];
    vl_lb_ext=[vl_lb;zeros(N_orig_vl-length(vl_lb),1)];  
    vl_X_SM(:,:,j)=vl_X_ext;
    vl_lb_SM(:,:,j)=vl_lb_ext;
    
    % test
    ts_X=X1(N_tr+N_vl+1:end,1:dim_X);
    ts_lb=Y1(N_tr+N_vl+1:end);
    % save test
    ts_X_ext=[ts_X;zeros(N_orig_ts-size(ts_X,1),dim_X)];
    ts_lb_ext=[ts_lb;zeros(N_orig_ts-length(ts_lb),1)];  
    ts_X_SM(:,:,j)=ts_X_ext;
    ts_lb_SM(:,:,j)=ts_lb_ext;
    
    for i=1:Division
        tau=tRange(i);
        cvx_begin quiet
        variable w(dim_X)
        minimize( norm( (tr_lb - tr_X*w)./tr_lb , 1 ) + tau*norm(w,1) )
        cvx_end
        
        [WIndx]=find(abs(w) > 1e-5);
        numNoZero(:,i,j)=length(WIndx);
        weights(:,i,j)=w;
    end 
end
%% find best weights
index_weights=ones(1,66);
best_weights=ones(dim_X,66);
costV_vl=zeros(Division,66);
for j=1:66
    for i=1:Division
        len_vl=length(find(vl_lb_SM(:,:,j)));

        vl_X=vl_X_SM(1:len_vl,:,j);
        vl_lb=vl_lb_SM(1:len_vl,:,j);
        residual=(vl_lb - vl_X * weights(:,i,j))./vl_lb;
        costV_vl(i,j)=norm(residual,1)/len_vl;
    end
    index2=find(costV_vl(:,j)==min(costV_vl(:,j)));
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
cost_didi=cost*0.66;




