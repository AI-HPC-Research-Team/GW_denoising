import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import gwsurrogate
import os
import h5py
import functools
from pathlib import Path

class GW_Dataset(object):
    def __init__(self,psd_dir = '.',load_dataset = False):
        param_idx = dict(mass_1=0, mass_2=1)
        nparams = 2
        self.param_idx = param_idx
        self.nparams = nparams
        
        #parameter range
        self.par_range = dict(mass_1=[5.0, 75.0],  mass_2=[5.0, 75.0]) # solar masses
        
        self.q_max = 10 
        
        self.f_min = 15  # Hertz
        self.sampling_rate = 4096.0
        self.time_duration = 1.0 
        self.psd_path = os.path.join(psd_dir,'aLIGODesign.txt')
        self.psd = None
        self.SNR = [9.0]
        self.train_par = []
        self.train_waveform = {'clean':[],'noisy':[]}
        self.test_par = []
        self.test_waveform ={'clean':[],'noisy':[]}
        
        #load NRSur model
        #download model
        fname = os.path.join(gwsurrogate.catalog.download_path(),'NRHybSur3dq8.h5')
        if not gwsurrogate.catalog.is_file_recent(fname):
            gwsurrogate.catalog.pull('NRHybSur3dq8')
        if not load_dataset:
            self.NRSur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
        
        
    def tukey(self,M,alpha=0.5):
        """
        Tukey window code copied from scipy
        """
        n = np.arange(0, M)
        width = int(np.floor(alpha*(M-1)/2.0))
        n1 = n[0:width+1]
        n2 = n[width+1:M-width-1]
        n3 = n[M-width-1:]

        w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
        w2 = np.ones(n2.shape)
        w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
        w = np.concatenate((w1, w2, w3))

        return np.array(w[:M])
    
    @property
    def f_max(self):
        """Set the maximum frequency to half the sampling rate."""
        return self.sampling_rate / 2.0

    @f_max.setter
    def f_max(self, f_max):
        self.sampling_rate = 2.0 * f_max

    @property
    def delta_t(self):
        return 1.0 / self.sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t):
        self.sampling_rate = 1.0 / delta_t

    @property
    def delta_f(self):
        return 1.0 / self.time_duration

    @delta_f.setter
    def delta_f(self, delta_f):
        self.time_duration = 1.0 / delta_f

    @property
    def Nt(self):
        return int(self.time_duration * self.sampling_rate)

    @property
    def Nf(self):
        return int(self.f_max / self.delta_f) + 1
    
    @property
    def sample_times(self):
        """Array of times at which waveforms are sampled."""
        return np.linspace(0.0, self.time_duration,
                           num=self.Nt,
                           endpoint=False,
                           dtype=np.float32)

    @property
    @functools.lru_cache()
    def sample_frequencies(self):
        return np.linspace(0.0, self.f_max,
                           num=self.Nf, endpoint=True,
                           dtype=np.float32)


    def get_psd(self):
        f, psd = np.loadtxt(self.psd_path, unpack=True)
        f_min = min(f)
        
        if f_min != 0:
            psd = np.insert(psd,0,[psd[0]])
            f   = np.insert(f,0,[0])
        psd_new = np.interp(self.sample_frequencies,f,psd)
        
        return psd_new
    
    
    def gen_signal(self,m1,m2,chiA = [0,0,0],chiB = [0,0,0]):
        q = m1/m2   #m1>m2
        #chiA = [0, 0, 0.5]
        #chiB = [0, 0, -0.7]
        M = m1 + m2             # Total masss in solar masses
        dist_mpc = 100     # distance in megaparsecs
        #dt = 1./8192       # step size in seconds
        #f_low = 20         # initial frequency in Hz
        t, h, dyn = self.NRSur(q, chiA, chiB, dt=self.delta_t, f_low=5, mode_list=[(2,2)], M=M, dist_mpc=dist_mpc, units='mks')
        n = self.sampling_rate * self.time_duration
        return h[(2,2)].real[-int(n):]
    
    def gen_noise(self):
        """
        Generates noise from a psd
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd
        

        N = T_obs * fs          # the total number of time samples
        #Nf = N // 2 + 1
        dt = 1 / fs             # the sampling time (sec)
        df = 1 / T_obs

        amp = np.sqrt(0.25*T_obs*psd)
        idx = np.argwhere(psd==0.0)
        amp[idx] = 0.0
        re = amp*np.random.normal(0,1,self.Nf)
        im = amp*np.random.normal(0,1,self.Nf)
        re[0] = 0.0
        im[0] = 0.0
        x = N*np.fft.irfft(re + 1j*im)*df

        return x
    
    def whiten_data(self,data,flag='td'):
        """
        Takes an input timeseries and whitens it according to a psd
        """
        duration = self.time_duration
        sample_rate = self.sampling_rate
        psd = self.psd

        if flag=='td':
            # FT the input timeseries - window first
            win = self.tukey(int(duration*sample_rate),alpha=1.0/8.0)
            xf = np.fft.rfft(win*data)
        else:
            xf = data

        # deal with undefined PDS bins and normalise
        idx = np.argwhere(psd>0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0/psd[idx]
        xf *= np.sqrt(2.0*invpsd/sample_rate)

        # Detrend the data: no DC component.
        xf[0] = 0.0

        if flag=='td':
            # Return to time domain.
            x = np.fft.irfft(xf)
            return x
        else:
            return xf
        
    def get_snr(self,data):
        """
        computes the snr of a signal given a PSD starting from a particular frequency index
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd
        fmin = self.f_min
        
        N = int(T_obs*fs)
        df = 1.0/T_obs
        dt = 1.0/fs
        fidx = int(fmin/df)

        win = self.tukey(N,alpha=1.0/8.0)
        idx = np.argwhere(psd>0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0/psd[idx]

        xf = np.fft.rfft(data*win)*dt
        SNRsq = 4.0*np.sum((np.abs(xf[fidx:])**2)*invpsd[fidx:])*df
        return np.sqrt(SNRsq)
    
    def get_inner_product(self,data1,data2):
        """
        computes the snr of a signal given a PSD starting from a particular frequency index
        """
        T_obs = self.time_duration
        fs = self.sampling_rate
        psd = self.psd
        fmin = self.f_min
        
        N = int(T_obs*fs)
        df = 1.0/T_obs
        dt = 1.0/fs
        fidx = int(fmin/df)

        win = self.tukey(N,alpha=1.0/8.0)
        idx = np.argwhere(psd>0.0)
        invpsd = np.zeros(psd.size)
        invpsd[idx] = 1.0/psd[idx]

        xf1 = np.fft.rfft(data1*win)*dt
        xf2 = np.fft.rfft(data2*win)*dt
        SNRsq = 2.0*np.sum((xf1[fidx:] * np.conjugate(xf2[fidx:]) + np.conjugate(xf1[fidx:])*xf2[fidx:]) *invpsd[fidx:])*df
        return np.sqrt(SNRsq).real
    
    def get_overlap(self,data1,data2):
        #normalize data
        data_1 = data1 / self.get_snr(data1)
        data_2 = data2 / self.get_snr(data2)
        overlap = self.get_inner_product(data_1,data_2)
        return overlap
    
    def generate_waveform(self):
        self.psd = self.get_psd()
        
        #training set
        m_min,m_max=Dataset.par_range['mass_1']
        for m1 in range(int(m_min),int(m_max)):
            for m2 in range(int(m_min),int(m1)):
                if (m1 / m2 <= self.q_max):
                    for snr in self.SNR:
                        self.train_par.append([m1,m2])
                        hp = self.gen_signal(m1,m2)
                        self.train_waveform['clean'].append(hp/max(hp))
                    
                        hp_tilt = snr * hp / self.get_snr(hp)
                        noise = self.gen_noise()
                        data = hp_tilt + noise
                        self.train_waveform['noisy'].append(data)
                        
        #test set
        for m1 in range(int(m_min),int(m_max-1)):
            for m2 in range(int(m_min),int(m1)):
                m1 += 0.5
                m2 += 0.5
                if (m1 / m2 <= self.q_max):
                    for snr in self.SNR:
                        self.test_par.append([m1,m2])
                        hp = self.gen_signal(m1,m2)
                        self.test_waveform['clean'].append(hp/max(hp))
                    
                        hp_tilt = snr * hp / self.get_snr(hp)
                        noise = self.gen_noise()
                        data = hp_tilt + noise
                        data /= np.std(data) 
                        self.test_waveform['noisy'].append(data)
        self.train_par = np.array(self.train_par)
        self.test_par = np.array(self.test_par)
        for key in self.train_waveform.keys():
            self.train_waveform[key] = np.array(self.train_waveform[key])
            self.test_waveform[key]  = np.array(self.test_waveform[key])
            
        
    def save_waveform(self,DIR='.',data_fn='waveform_dataset.hdf5'):
        p = Path(DIR)
        p.mkdir(parents=True, exist_ok=True)
        
        f_data = h5py.File(p / data_fn, 'w')

        f_data.create_dataset('train_par', data=self.train_par,
                              compression='gzip', compression_opts=9)
        f_data.create_dataset('test_par', data=self.test_par,
                              compression='gzip', compression_opts=9)
        data_name = '0'
        for i in self.train_waveform.keys():
            data_name = 'train_' + i
            f_data.create_dataset(data_name, data=self.train_waveform[i],
                              compression='gzip', compression_opts=9)
        for i in self.test_waveform.keys():
            data_name = 'test_' + i
            f_data.create_dataset(data_name, data=self.test_waveform[i],
                              compression='gzip', compression_opts=9)
        f_data.close()
        
    def load_waveform(self,DIR='.',data_fn='waveform_dataset.hdf5'):
        p = Path(DIR)
        
        f_data = h5py.File(p / data_fn, 'r')
        
        self.train_par = f_data['train_par'][:, :]
        self.test_par  = f_data['test_par'][:, :]

        data_name = '0'
        for i in self.train_waveform.keys():
            data_name = 'train_' + i
            self.train_waveform[i] = f_data[data_name][:, :]
        for i in self.test_waveform.keys():
            data_name = 'test_' + i
            self.test_waveform[i] = f_data[data_name][:, :]
        
        f_data.close()
            
        
        
                        
                    
        
   