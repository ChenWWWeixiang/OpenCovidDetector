import os,random,glob
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-p", "--path", help="A list of paths to jpgs for seperate",
                    type=str,
                    default=[#'/mnt/data9/covid_detector_jpgs/pos_abnormal',
                              #'/mnt/data9/covid_detector_jpgs/neg_abnormal',
                              #'/mnt/data7/resampled_jpgs/masked_train_lidc',
                              #'/mnt/data9/covid_detector_jpgs/masked_ild',
                             '/mnt/data9/covid_detector_jpgs/crop_masked_healthy',
                             '/mnt/data9/covid_detector_jpgs/crop_masked_covid',
                             '/mnt/data9/covid_detector_jpgs/crop_masked_cap',
                             #'/mnt/data9/covid_detector_jpgs/masked_cap_zs'
                             ])
parser.add_argument("-t", "--train_txt",
                    help="train list output path",
                    type=str,                    
                    default='txt/train3cls_c.txt')
parser.add_argument("-v", "--val_txt",
                    help="validation list output path",
                    type=str,
                    default='txt/test3cls_c.txt')

args = parser.parse_args()
if isinstance(args.path,str):
    path=eval(args.path)
else:
    path=args.path
#path=['/mnt/data7/slice_test_seg/jpgs2']
f1 = open(args.train_txt, 'w')
f2 = open(args.val_txt, 'w')

train_count=500000
c=0
for ipath in path:
    cnt = 0
    files=os.listdir(ipath)
    names_id=[file.split('_')[1]+'_'+file.split('_')[2] for file in files]
    names_id=list(set(names_id))
    set_name=files[0].split('_')[0]
    random.shuffle(names_id)
    train=names_id[:-len(names_id)//4]
    #val=names_id[len(names_id)//2:-len(names_id)//4]
    test=names_id[-len(names_id)//4:]
    for _,i in enumerate(train):
        if cnt >= train_count:
            break
        names=glob.glob(ipath+'/'+set_name+'_'+i+'_*')
        for name in names:
            if cnt>=train_count:
                break
            cnt+=1
            c+=1
            f1.writelines(name+'\n')
#print(c)
    for i in test:
        names = glob.glob(ipath + '/' + set_name + '_' + i + '_*')
        for name in names:
            f2.writelines(name+'\n')