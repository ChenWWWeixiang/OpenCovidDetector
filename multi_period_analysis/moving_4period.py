import os,shutil
in_root='/mnt/data7/NCP_mp_CTs/reg/lesions/NCP_ill'
out_root='/mnt/data9/mp_NCPs/lesions/'
os.makedirs(out_root,exist_ok=True)
for i in range(1,8):
    for items in os.listdir(in_root+str(i)):
        if int(items[-1])>1:
            shutil.copytree(os.path.join(in_root + str(i), items.split('_')[0]+'_1'),
                            os.path.join(out_root, str(i) + '_' + items.split('_')[0]))
            shutil.copytree(os.path.join(in_root+str(i),items),os.path.join(out_root,str(i)+'_'+items.split('_')[0]))