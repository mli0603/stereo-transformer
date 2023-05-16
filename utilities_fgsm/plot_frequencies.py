import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.stats as stats
import torch.nn.functional as F
import torchvision.transforms as T
import torch
import torch.fft


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_plotting_info(data, avg_pool=False):
    first = True
    first_amp = True
    count = 0
    amplitudes = []
    torch.cuda.empty_cache()
    #d3 = torch.device('cuda:3')
    #d2 = torch.device('cuda:2')

    for split_data in torch.squeeze(torch.stack(data), dim=1):
        for img in split_data:            
            if avg_pool:
                img = T.Resize((img.shape[0]//4, img.shape[1]//4))(torch.unsqueeze(img,0))[0]
            if first:
                freq_domain = np.expand_dims(torch.fft.fftn(img.cuda()).detach().cpu().numpy(), axis=0)
                first = False      
            else:
                freq_domain = np.vstack((freq_domain, np.expand_dims(torch.fft.fftn(img.cuda()).detach().cpu().numpy(), axis=0)))
        if first_amp:
            amplitudes = np.expand_dims(np.average(np.abs(freq_domain)**2, axis=0), axis =0)
            first_amp = False
        else:
            amplitudes = np.vstack((amplitudes, np.expand_dims(np.average(np.abs(freq_domain)**2, axis=0), axis =0)))
        first=True
    
    
    amplitude = np.average(np.asarray(amplitudes), axis=0)
   
    npix = amplitude.shape[0]
    npix2 = amplitude.shape[1]
    kfreq = np.fft.fftfreq(npix) *npix
    kfreq2 = np.fft.fftfreq(npix2) *npix2
    #import ipdb;ipdb.set_trace()



    kfreq2D = np.meshgrid(kfreq, kfreq2)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = amplitude.flatten()

    kbins = np.arange(0.5, npix2//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins

def plot_frequencies(all_data, output_path, logger):
    #import ipdb;ipdb.set_trace()
    left_feats = get_plotting_info(all_data[0][::3])
    right_feats = get_plotting_info(all_data[1][::3])
    #left_input = get_plotting_info(all_data[2][::3], avg_pool=True)
    #right_input = get_plotting_info(all_data[3][::3], avg_pool=True)

    
    lf_save_name = output_path+ '/left_feats.npy'
    rf_save_name = output_path+ '/right_feats.npy'
    np.save(lf_save_name, left_feats)
    np.save(rf_save_name, right_feats)
    logger.info('\n\n\n\n\nOUTPUT_PATH={}\n\n\n\n'.format(lf_save_name))
    #li_save_name = output_path+ '/left_inputs.npy'
    #ri_save_name = output_path+ '/right_inputs.npy'

    #torch.save(torch.tensor([left_feats], device='cpu'), lf_save_name)
    #torch.save(torch.tensor([right_feats], device='cpu'), rf_save_name)
    #torch.save(torch.tensor([left_input], device='cpu'), li_save_name)
    #torch.save(torch.tensor([right_input], device='cpu'), ri_save_name)
    
    #np.save(li_save_name, left_input)
    #np.save(ri_save_name, right_input)
    

    


