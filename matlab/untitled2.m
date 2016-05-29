path = 'season_1/training_data/order_data/order_data_2016-01-01';
tableFormat = ['order_id','driver_id','passenger_id','start_district_hash',...
    'dest_district_hash','Price','Time'];
formatSpec = '%s%s%s%s%s%d%{yyyy-MM-dd HH:mm:ss}D';
a = cell(0);
a(1) = {tableFormat};
% a = readtable(path,'Delimiter','\t','Format',formatSpec);