import os
data_path='/home/cwx/extra/covid_project_data'
seg_path='/home/cwx/extra/covid_project_segs'
with open('3cls_test.list','w') as f:
    for set_name in os.listdir(data_path):
        if set_name=='cap_qqhr' or set_name=='lidc' or set_name=='ild' or set_name=='cap_zs':
            continue
        for name in os.listdir(os.path.join(data_path,set_name)):
            set_id = int(name.split('_')[0])
            person_id = name.split('_')[1].split('-')[0]
            if set_name == 'healthy':
               # continue
                if set_id <= 6:  # 1-6 train, 7-13 test
                    continue
            if set_name == 'cap' :
                continue
            if set_name == 'covid':
                if set_id <= 5 or set_id>=8:
                    continue
            if set_name=='cap2':
                fix='cap'
            else:
                fix=set_name
            f.writelines(os.path.join(data_path,set_name,name)+','+os.path.join(seg_path,'lungs',fix,set_name+'_'+name)+'\n')