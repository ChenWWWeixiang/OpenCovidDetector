import os,glob
data_raw=open('lists/reader_HN_vs_covid.list','r').readlines()
data_com=open('3cls_train_HN.list','r').readlines()
data_new=[]
with open('lists/reader_influenza_vs_covid.list','w') as f:
    for item in data_raw:
        if 'HxNx' in item:
            '/home/cwx/extra/covid_project_data/HxNx/45_0_20180212_72_F.nii,/home/cwx/extra/covid_project_segs/lungs/HxNx/HxNx_45_0_20180212_72_F.nii'
            item=item.replace('HxNx','AB-in')
            first=item.split(',')[0]
            id=first.split('AB-in/')[-1].split('_')
            new_first=first.split('AB-in/')[0]+'AB-in/1_'+id[0]+'_'+id[2]+'_'+id[3]+'_'+id[4]
            if not os.path.exists(new_first):
                try:
                    new_first=glob.glob(first.split('AB-in/')[0]+'AB-in/1_'+id[0]+'_*.nii')[0]
                except:
                    try:
                        new_first = glob.glob(first.split('AB-in/')[0] + 'AB-in/1_*' + id[2] + '_*.nii')[0]
                    except:
                        new_first = glob.glob(first.split('AB-in/')[0] + 'AB-in/1_*.nii')[0]
            new_second =new_first.replace('AB-in/','AB-in/AB-in_').replace('_data/','_segs/lungs/')
            data_new.append(new_first+','+new_second)
            f.writelines(new_first+','+new_second+'\n')
        else:
            f.writelines(item)
            data_new.append(item)
