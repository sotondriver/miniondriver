db.order_data.find({'start_district_hash':{$ne:NaN}}).forEach( 
    function (doc) {
        temp_hash = doc.start_district_hash
        id = db.cluster_map.findOne({"cluster_hash":temp_hash})["id"]
        print(id)
        doc.start_district_hash = id; 
        db.order_data.save(doc); 
});

db.order_data.find({'Time':{$ne:NaN}}).forEach( 
    function (doc) {
        temp_str = doc.Time
        temp_str = temp_str.replace("-", ":")
        temp_str = temp_str.replace("-", ":")
        temp_str = temp_str.replace(" ", ":")
        array = temp_str.split(":")
        print(array)
        temp1 = NumberInt(array[2])
        t1 = NumberInt(array[3])
        t2 = NumberInt(array[4])
        temp2 = (t1 * 6 + Math.floor(t2 / 10)) + 1 
        doc.date = temp1
        doc.time_slot = temp2
        db.order_data.save(doc)
});