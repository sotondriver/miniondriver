% prediction part


ts_X=importdata('../data/data_chao/didi_test_data.csv');
results=ones(43,66);
%% predict
for j=1:66 
    results(:,j)=ts_X * best_weights(:,j);
end
results=results';

results_V=[];
for j=1:43
    results_V=[results_V;results(:,j)];
end
%% write csv and save
test_file=importdata('../data/test.csv');
% test_file=importdata('test.csv');
test_file.data=results_V;
test_table=table(test_file.textdata(:,1),...
    test_file.textdata(:,2),...
    test_file.data);
writetable(test_table, 'test.csv');
%% save best weights
best_weights_table=table(best_weights);
writetable(best_weights_table, 'best_weights.csv');






