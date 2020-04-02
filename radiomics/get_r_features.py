from radiomics import featureextractor
import os,csv,six
import numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-o", "--outputfile", help="output file's name", type=str,
                    default='withfake_features.csv')
parser.add_argument("-m", "--inputmask", help="input mask root", type=str,
                    default='mask/')
parser.add_argument("-r", "--inputimgs", help="input data root", type=str,
                    default='img/')
args = parser.parse_args()

o_img_nii = args.inputimgs
o_msk_nii = args.inputmask
extractor = featureextractor.RadiomicsFeatureExtractor('RadiomicsParams.yaml')



with open(args.outputfile, 'w', newline='') as f:
    writer = csv.writer(f)
    for i,name in enumerate(os.listdir(o_img_nii)):
        #print(name)
        if not name[0]=='c' and i%4>=1:
            continue
        row = ['id', 'label']
        row_next = [name, 1-(name[0]=='c')]
        if name[0]=='c':
            listall=os.listdir(o_msk_nii)
            listall=[item for item in listall if not item[0]=='c']
            n_fake=listall[np.random.randint(0,len(listall))]
        else:
            n_fake=name
        imageName=os.path.join(o_img_nii,name)
        maskName=os.path.join(o_msk_nii,n_fake)
        try:
            result = extractor.execute(imageName, maskName)
            for idx, (key, val) in enumerate(six.iteritems(result)):
                if idx<14:
                    continue
                if not isinstance(val,(float,int,np.ndarray)):
                    continue
                if np.isnan(val):
                    val=0
                   # print(val)
                row.append(key)
                row_next.append(val)
            if i ==0:
                writer.writerow(row)

            writer.writerow(row_next)
        except:
            print(imageName)
            os.remove(imageName)
            os.remove(maskName)