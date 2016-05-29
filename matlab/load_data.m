PATH = 'season_1/training_data/';

list1 = dir(PATH);
for i = (3:length(list1))
    if (list1(i).isdir())
        list2 = dir([PATH, list1(i).name]);
        eval([list1(i).name,'= {}']);
        for j = (3:length(list2))
            if (~list2(j).isdir())
                s = list2(j).name;
                
            end
        end
    end
end

function  data = load_data(path)
    
end