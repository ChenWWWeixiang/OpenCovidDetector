import os
data_path='/home/cwx/extra/covid_project_data'
with open('4cls_test.list','w') as f:
    for set_name in os.listdir(data_path):
        if set_name=='cap_qqhr' or set_name=='lidc':
            continue
        for name in os.listdir(os.path.join(data_path,set_name)):
            set_id = int(name.split('_')[0])
            person_id = name.split('_')[1].split('-')[0]
            if set_name == 'healthy':
                if set_id <= 6:  # 1-6 train, 7-13 test
                    continue
            if set_name == 'cap' and int(person_id) <= 100:
                continue
            if set_name == 'ild' and int(person_id) <= 84:
                continue
            if set_name == 'covid' and set_id <= 3:
                continue
            f.writelines(os.path.join(data_path,set_name,name)+'\n')