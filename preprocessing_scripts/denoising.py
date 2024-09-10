import bm3d
import scipy.io

import os

data_path = os.path.expandvars('$TMPDIR/FO04_raw.mat')
mat_data = scipy.io.loadmat(data_path)

z = mat_data['FO04_raw']

#Denoising

sigma = 20

dn = bm3d.bm3d(z, sigma, profile = 'np')


#


output_path = os.path.expandvars('$TMPDIR/FO04_bm3d_dn_20.mat')
data_to_save = {'FO04_bm3d_dn_20': dn}

scipy.io.savemat(output_path, data_to_save)

print('done')
