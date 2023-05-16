import cv2
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import pathlib
#from util.util import check_makedirs

colors = ['dimgrey', 'lightcoral', 'aqua', 'gold', 'darkgreen', 'red', 'palegreen', 'dodgerblue', 'gold', 'navy', 'darkmagenta', 'lightgreen', 'darkred', 'olive', 'indigo', 'tan']
count = 0


left_feats = glob.glob("/work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/**/**/left_feats.npy")
right_feats = glob.glob("/work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/**/**/left_feats.npy")
left_input = glob.glob("/work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/**/**/left_feats.npy")
right_input = glob.glob("/work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/**/**/left_feats.npy")

left_input=left_input[0]
right_input=right_input[0]

left_i = np.load(left_input)
right_i = np.load(right_input)

save_path = "/work/ws-tmp/sa058646-segment2/stereo-transformer/run/sceneflow/freq_comparison_norm.png"

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30,15))

#ax1.plot(left_i[0], left_i[1], label='input', color='k', linestyle='-.', linewidth=4.0)
#ax2.plot(right_i[0], right_i[1], label='input', color='k', linestyle='-.', linewidth=4.0)
left_i[1]=left_i[1]/left_i[1][0]
right_i[1]=right_i[1]/right_i[1][0]
ax1.plot(left_i[0], left_i[1], label='input', color='k', linewidth=4.0)
ax2.plot(right_i[0], right_i[1], label='input', color='k', linewidth=4.0)

#import ipdb;ipdb.set_trace()

for left, right in zip(left_feats, right_feats):
    #import ipdb;ipdb.set_trace()
    p = pathlib.Path(left)
    name = p.parent.parent.name
    left_loaded = np.load(left)
    left_kvals = left_loaded[0]
    left_Abins = left_loaded[1]/left_loaded[1][0]
    right_loaded = np.load(right)
    right_kvals = right_loaded[0]
    right_Abins = right_loaded[1]/right_loaded[1][0]
    
    if 'v' in name:
        ax1.plot(left_kvals, left_Abins, label=name, color='gold')
        ax2.plot(right_kvals, right_Abins, label=name, color='gold')
    else:
        ax1.plot(left_kvals, left_Abins, label=name)
        ax2.plot(right_kvals, right_Abins, label=name)

ax1.set_title('LEFT')
ax2.set_title('RIGHT')

fig.suptitle('Comparing Freqeuncies after feature extraction')

#plt.plot(gt_kvals, gt_Abins, label='ground_truth', color='k', linestyle='-.')
ax1.set_xlabel("$k$")
ax1.set_ylabel("$P(k)$")
ax2.set_xlabel("$k$")
ax2.set_ylabel("$P(k)$")
ax1_y1, ax1_y2 = ax1.get_ylim()
ax1_x1, ax1_x2 = ax1.get_xlim()
ax2_y1, ax2_y2 = ax2.get_ylim()
ax2_x1, ax2_x2 = ax2.get_xlim()
#ax1.set_ylim(y1, 10**7)
#ax1.set_xlim(20,x2)
#ax2.set_ylim(y1, 10**7)
#ax2.set_xlim(20,x2)

plt.tight_layout()
plt.legend(bbox_to_anchor=(1.04, 1), loc="best")

plt.savefig(save_path, dpi = 500, bbox_inches = "tight")








"""
curr_dir = os.getcwd()
img = cv2.imread(curr_dir+'/temp.png',0)
print( img.shape )

# Fourier Transform along the first axis

# Round up the size along this axis to an even number
n = int( math.ceil(img.shape[0] / 2.) * 2 )

# We use rfft since we are processing real values
a = np.fft.rfft(img,n, axis=0)

# Sum power along the second axis
a = a.real*a.real + a.imag*a.imag
a = a.sum(axis=1)/a.shape[1]

# Generate a list of frequencies
f = np.fft.rfftfreq(n)

# Graph it
plt.plot(f[1:],a[1:], label = 'sum of amplitudes over y vs f_x')

# Fourier Transform along the second axis

# Same steps as above
n = int( math.ceil(img.shape[1] / 2.) * 2 )

a = np.fft.rfft(img,n,axis=1)

a = a.real*a.real + a.imag*a.imag
a = a.sum(axis=0)/a.shape[0]

f = np.fft.rfftfreq(n)

plt.plot(f[1:],a[1:],  label ='sum of amplitudes over x vs f_y')

plt.ylabel( 'amplitude' )
plt.xlabel( 'frequency' )
plt.yscale( 'log' )

plt.legend()

plt.savefig( 'test_rfft.png' , transparent=True )
#plt.show()



        mag_path = os.path.join(freq_folder, image_name + '_mag.png')
        freq_path = os.path.join(freq_folder, image_name + '_freq.png')
        post_path = os.path.join(freq_folder, image_name + '_map.png')
        new_col, new_row, rows, cols=True, True, 0,0
        for it in range(posterior.shape[-1]):
            maps=posterior[:,:,it]
            if new_col:
                cols=maps
                new_col=False
            else:                
                cols=np.concatenate((cols,maps), axis=1)
            if (it+1)%7==0:
                cols.shape
                if new_row:
                    rows=cols
                    new_row=False
                else:
                    rows=np.vstack([rows, cols])
                new_col=True
        #import ipdb;ipdb.set_trace()
        #for i,j in zip(range(rows.shape[0]), range(rows.shape[1])):            
        #    if rows[i,j] < 1e-7:
        #        rows[i,j] = 255
            #    item=0
        #import ipdb;ipdb.set_trace()        
        (thresh, blackAndWhiteImage) = cv2.threshold(rows, 0.1, 255, cv2.THRESH_BINARY)        
        cv2.imwrite(post_path, blackAndWhiteImage)
        cv2.imwrite(mag_path, magnitude_specturm)
        cv2.imwrite(freq_path, img_back)
        gray_posterior= np.uint8(posterior)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_specturm = np.abs(fshift)
        row, col = gray.shape
        win_row, win_col = row//2, col//2        
        fshift[win_row-30:win_row+30, win_col-30:win_col+30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)


"""