#########################################
##        fortytwo_util_noise.py       ##
#      Pulsar noise related functions   #
#########################################

####################################################################

import pylab as plt
import socket
from math import *
from numpy import linalg
import scipy.linalg
import sys, getopt
import os
import inspect
import io
import time
import numpy as np
import scipy.linalg as slg
#import pymultinest
import scipy.integrate
#from sklearn.cluster import KMeans
from Fortytwo.fortytwo_util_prod import *
from Fortytwo.fortytwo_util_main import *
from Fortytwo.fortytwo_util_GWB import *
####################################################################


def PowerLaw_Covgen_K(vt, nf=10, fmin=1/30., fmax=10., fc=1., alpha=-1.5, ac=1.,spacing='log'):
    vx=np.arange(0, int(nf) , dtype=np.float64)/(nf-1)
    if spacing=='log':
        lfmin=log(fmin)
        lfmax=log(fmax)
        romb_coef=np.ones(len(vx))/len(vx)
        vlf=lfmin+(lfmax-lfmin)*vx
        vf=np.exp(vlf)
        k=ac*ac*np.power(vf/fc, alpha*2.)*romb_coef*(lfmax-lfmin)
    elif spacing=='linear':
        romb_coef=np.ones(len(vx))/(len(vx)-1)
        vf=fmin+(fmax-fmin)*vx
        k=ac*ac*np.power(vf/fc, alpha*2.-1)*romb_coef*(fmax-fmin)

    k2 = np.empty((k.size*2,), dtype=np.float64)
    k2[0::2] = k
    k2[1::2] = k
    k=k2
    return k

def PowerLaw_Covgen_Phi_K(vt,   phase_shift, nf=10, fmin=1/30., fmax=10., fc=1., alpha=-1.5, ac=1.,spacing='linear'):
    # RNC note: The if sections for (nf-1)=0 are made to avoid RuntimeWarning - This is handled differently in the next version
    if nf!=1:
        vx=np.arange(0, int(nf) , dtype=np.float64)/(nf-1) 
    if spacing=='log':
        lfmin=log(fmin)
        lfmax=log(fmax)
        romb_coef=np.ones(len(vx))/len(vx)

        vlf=lfmin+(lfmax-lfmin)*vx
        vf=np.exp(vlf)
        k=ac*ac*np.power(vf/fc, alpha*2.)*romb_coef*(lfmax-lfmin)
    elif spacing=='linear':
        if nf==1: #see note above
            romb_coef,vf,k=np.array([inf]),np.array([nan]),np.array([nan])
        else:    
            romb_coef=np.ones(len(vx))/(len(vx)-1)
            vf=fmin+(fmax-fmin)*vx
            k=ac*ac*np.power(vf/fc, alpha*2.-1)*romb_coef*(fmax-fmin) 

    k2 = np.empty((k.size*2,), dtype=np.float64)
    k2[0::2] = k
    k2[1::2] = k
    k=k2
    phic=np.cos(np.kron(vt, vf).reshape((len(vt), len(vf)))*2.*np.pi+phase_shift)
    phis=np.sin(np.kron(vt, vf).reshape((len(vt), len(vf)))*2.*np.pi+phase_shift)
    phi=np.hstack((phic, phis))
    n,m=phic.shape
    phi=np.empty((n,m*2), dtype=np.float64)
    phi[:,0::2] = phic
    phi[:,1::2] = phis
    return (k, phi, vf)

def PowerLaw_Covgen(vt, nf=10, fmin=1/30., fmax=10., fc=1., alpha=-1.5, ac=1.,spacing='log',phase_shift=0):
    vx=np.arange(0, int(nf) , dtype=np.float64)/(nf-1)
    if spacing=='log':
        lfmin=log(fmin)
        lfmax=log(fmax)
        romb_coef=np.ones(len(vx))/len(vx)

        vlf=lfmin+(lfmax-lfmin)*vx
        vf=np.exp(vlf)
        k=ac*ac*np.power(vf/fc, alpha*2.)*romb_coef*(lfmax-lfmin)

    elif spacing=='linear':
        romb_coef=np.ones(len(vx))/(len(vx)-1)

        vf=fmin+(fmax-fmin)*vx
        k=ac*ac*np.power(vf/fc, alpha*2.-1)*romb_coef*(fmax-fmin)

    k2 = np.empty((k.size*2,), dtype=np.float64)
    k2[0::2] = k
    k2[1::2] = k
    k=k2
    phic=np.cos(np.kron(vt, vf).reshape((len(vt), len(vf)))*2.*np.pi+phase_shift) 
    phis=np.sin(np.kron(vt, vf).reshape((len(vt), len(vf)))*2.*np.pi+phase_shift)
    phi=np.hstack((phic, phis))
    n,m=phic.shape
    phi=np.empty((n,m*2), dtype=np.float64)
    phi[:,0::2] = phic
    phi[:,1::2] = phis
    return (phi.dot(np.diag(k)).dot(phi.T), k, phi,vf)

def Calculate_RedNoiseCovariance(msp_info, vec_keys, phase_shift, nf=7, freq_low_cut=1./50.0, freq_high_cut=365.25*1.234567,spacing='log'):
    #Nf=7
    npsr=len(vec_keys)
    key_ind=np.arange(0,npsr)
    cov_r = np.array([[]])
    i=0 #pulsar count for pshift index

    for ind in key_ind:
        key=vec_keys[ind]
        vtBAT = msp_info[key]['avr_bat']
        vt_psr = vtBAT 
        cov_w=np.diag(msp_info[key]['avr_err']*msp_info[key]['avr_err'])
        detw=np.sum(2.0 * np.log(np.abs(msp_info[key]['avr_err'])))


        cov_w=msp_info[key]['qd'].T.dot( cov_w.dot( msp_info[key]['qd'] ) )
        u, s, v = np.linalg.svd(cov_w)
        cov_w=0
        invc = v.transpose().dot(np.diag(1./s)).dot(u.transpose())
        msp_info[key]['cov_winv']=invc.copy()
        msp_info[key]['det_w']=np.sum(np.log(np.abs(s)))
        cov_w=0
        invc=0
        
        pshift=phase_shift[i] #set the phase shift value (pshift)

        if msp_info[key]['amp_red']>0:
            K, Phi,vf=PowerLaw_Covgen_Phi_K(vt_psr, pshift, nf[ind,0], fmin=freq_low_cut[ind,0], fmax=freq_high_cut[ind,0], 
                    fc=1., alpha=msp_info[key]['alpha_red'], 
                    ac=msp_info[key]['amp_red'],spacing=spacing)

            msp_info[key]['K_Red']= K.copy()
            msp_info[key]['Phi_Red']= msp_info[key]['qd'].T.dot(Phi)
            vf2 = np.empty((vf.size*2,), dtype=np.float64)
            vf2[0::2] = vf
            vf2[1::2] = vf
            msp_info[key]['vf']=vf2
            cov_r=0
            K=0
            Phi=0
            
        else:
             msp_info[key]['cov_r'] = 0
             msp_info[key]['K_Red']=0
             msp_info[key]['Phi_Red']=0
        
        if msp_info[key]['amp_dm']>0:
            K, Phi,vf=PowerLaw_Covgen_Phi_K(vt_psr, pshift, nf[ind,1], fmin=freq_low_cut[ind,1], fmax=freq_high_cut[ind,1], 
                    fc=1., alpha=msp_info[key]['alpha_dm'], 
                    ac=msp_info[key]['amp_dm'],spacing=spacing)
            K=K*4.15e3/(365.25*3600*24.)*4.15e3/(365.25*3600*24.)
            f2=np.diag(1./(msp_info[key]['avr_freq']*msp_info[key]['avr_freq']))
            Phi=f2.dot(Phi)
            msp_info[key]['K_DM']= K.copy()
            msp_info[key]['Phi_DM']= msp_info[key]['qd'].T.dot(Phi)
            vf2 = np.empty((vf.size*2,), dtype=np.float64)
            vf2[0::2] = vf
            vf2[1::2] = vf
            msp_info[key]['vf_dm']=vf2
            cov_dm=0
            K=0
            Phi=0

        else:
             msp_info[key]['cov_dm'] = 0

        # scattering variation
        if nf.shape[1]>2:
            K, Phi,vf=PowerLaw_Covgen_Phi_K(vt_psr, pshift, nf[ind,2], fmin=freq_low_cut[ind,2], fmax=freq_high_cut[ind,2], 
                    fc=1., alpha=msp_info[key]['alpha_dm'], 
                    ac=msp_info[key]['amp_dm'],spacing=spacing)
            f4=np.diag(1./(msp_info[key]['avr_freq']**4))
            Phi=f4.dot(Phi)*1400**4*4.15e-3
            msp_info[key]['K_Sv']= K.copy()
            msp_info[key]['Phi_Sv']= msp_info[key]['qd'].T.dot(Phi)
            vf2 = np.empty((vf.size*2,), dtype=np.float64)
            vf2[0::2] = vf
            vf2[1::2] = vf
            msp_info[key]['vf_Sv']=vf2
            

        if nf.shape[1]>3:
            K, Phi,vf=PowerLaw_Covgen_Phi_K(vt_psr, nf[ind,3], pshift, fmin=freq_low_cut[ind,3], fmax=freq_high_cut[ind,3], 
                    fc=1., alpha=msp_info[key]['alpha_red'], 
                    ac=msp_info[key]['amp_red'],spacing=spacing)
            msp_info[key]['K_sys']= K.copy()
            msp_info[key]['Phi_sys']=Phi

            K=K*4.15e3/(365.25*3600*24.)*4.15e3/(365.25*3600*24.)
            f2=np.diag(1./(msp_info[key]['avr_freq']*msp_info[key]['avr_freq']))
            Phi=f2.dot(Phi)
            msp_info[key]['K_sys_DM']= K.copy()
            msp_info[key]['Phi_sys_DM']=Phi

            vf2 = np.empty((vf.size*2,), dtype=np.float64)
            vf2[0::2] = vf
            vf2[1::2] = vf
            msp_info[key]['vf_sys']=vf2

        if nf.shape[1]>4:
            K, Phi,vf=PowerLaw_Covgen_Phi_K(vt_psr, pshift, nf[ind,4], fmin=freq_low_cut[ind,4], fmax=freq_high_cut[ind,4], 
                    fc=1., alpha=msp_info[key]['alpha_red'], 
                    ac=msp_info[key]['amp_red'],spacing=spacing)
            msp_info[key]['K_BN']= K.copy()
            msp_info[key]['Phi_BN']=Phi

            K=K*4.15e3/(365.25*3600*24.)*4.15e3/(365.25*3600*24.)
            f2=np.diag(1./(msp_info[key]['avr_freq']*msp_info[key]['avr_freq']))
            Phi=f2.dot(Phi)
            msp_info[key]['K_BN_DM']= K.copy()
            msp_info[key]['Phi_BN_DM']=Phi

            vf2 = np.empty((vf.size*2,), dtype=np.float64)
            vf2[0::2] = vf
            vf2[1::2] = vf
            msp_info[key]['vf_BN']=vf2

        K=0
        Phi=0
        i+=1  #update pulsar count for pshift index



def DM_event(t,freq,amp,t0,Lambda,n=0):
    wf=amp*(Lambda*np.pi**0.5)**-0.5*np.exp((t-t0)**2/(2*Lambda**2))
    wf=wf*4.15e3/(365.25*3600*24.)/freq**2

def chrom_exp_decay(toas, freqs, log10_Amp=-7, sign_param=-1.0,
                    t0=54000, log10_tau=1.7, idx=2):
    """
    Chromatic exponential-dip delay term in TOAs.

    :param t0: time of exponential minimum [MJD]
    :param tau: 1/e time of exponential [s]
    :param log10_Amp: amplitude of dip
    :param sign_param: sign of waveform
    :param idx: index of chromatic dependence

    :return wf: delay time-series [s]
    """
    t0=t0/365.25
    tau = 10**log10_tau/365.25
    ind = np.where(toas > t0)[0]
    wf = 10**log10_Amp * np.heaviside(toas - t0, 1)
    wf[ind] *= np.exp(- (toas[ind] - t0) / tau)

    return np.sign(sign_param) * wf * (1400 / freqs) ** idx

def UpdateK_SN(vpar, user_in, freq_bin_mode, msp_info, vec_keys, npar_gw, SN_key={},spacing='linear',fitsin=0):
    res=0
    ldet=0
    idx=0  # idx for parameter
    idx+=npar_gw

    vws={}
    sum_npar_sn=0 #idx for correctly selecting pulsar GlEfac vpar index 

    for key in vec_keys:
        if freq_bin_mode==2:
            nDMevent=SN_key[key]['nDMevent']
            # vsystem=SN_key[key]["vsystem"]
            # vband=SN_key[key]["vband"]
            # nsys=len(SN_key[key]["vsystem"])
            # nband=len(SN_key[key]["vband"])
            fitRN = (SN_key[key]['nf_r']>0)
            fitDM = (SN_key[key]['nf_dm']>0)
            fitSv = (SN_key[key]['nf_sv']>0)            
            fit_Global_Efac=user_in.fitWN 

        elif freq_bin_mode==1:
            nDMevent=SN_key[key]['nDMevent']
            fit_Global_Efac=user_in.fitWN 
            vsystem=[]
            # nsys=0
            # nband=0
            # vband=[]
            if hasattr(user_in,'fitRN') and user_in.fitRN==1:
                fitRN=1
            else:
                fitRN=0
                
            if hasattr(user_in,'fitDM') and user_in.fitDM==1:
                fitDM=1
            else:
                fitDM=0
                
            if hasattr(user_in,'fitSv') and user_in.fitSv==1:
                fitSv=1
            else:
                fitSv=0

        elif freq_bin_mode==0: # # RN+DM
            fitRN=1
            fitDM=1
            nDMevent=SN_key[key]['nDMevent']
            nsys=0
            # vsystem=[]
            # nband=0
            # vband=[]
            fitSv=0
            fit_Global_Efac=0 #fixed to zero


        npar_sn=2*fitRN+2*fitDM+2*fitSv+3*nDMevent+fit_Global_Efac
        
        if fit_Global_Efac:
            sum_npar_sn=sum_npar_sn+npar_sn
            idx_glef=npar_gw+sum_npar_sn-1
            ws=vpar[idx_glef]**2
        else:
            ws=1
            
        vws[key]=ws 
        
        if fitRN:
            amp_red=10.**vpar[idx]*(12*np.pi**2)**-0.5
            alpha_red=(1-vpar[idx+1])/2.
            idx+=2
        if fitDM:
            amp_dm=10.**vpar[idx]* 24.*3600.*365.25
            alpha_dm=(1-vpar[idx+1])/2
            idx+=2
        if fitSv:
            amp_sv=10.**vpar[idx]
            alpha_sv=(1-vpar[idx+1])/2  
            idx+=2


        vr=msp_info[key]['avr_post']


        ## DM exp dip
        wf=0
        if nDMevent>0: # only for pulsar(s) with nDMevent > 0
            vidx=[1]
            for j in range(nDMevent):
                log10_Amp=vpar[idx]
                t0=vpar[idx+1]
                log10_tau=vpar[idx+2]
                wf+=chrom_exp_decay(msp_info[key]['avr_bat'],msp_info[key]['avr_freq'],log10_Amp=log10_Amp, sign_param=-1.0,t0=t0, log10_tau=log10_tau,idx=vidx[j])
                idx+=3

            vr=vr-wf
            

        if fitsin>0:
            amp_sin=10.**vpar[npar_gw-3]
            fsin=10.**vpar[npar_gw-2]
            phi_sin=vpar[npar_gw-1]
            wf=amp_sin*np.sin(2*np.pi*fsin*msp_info[key]['avr_bat']*24.*3600.*365.25+phi_sin)

            vr=vr-wf

        if (nDMevent+fitsin)>0:
            invA=msp_info[key]['covi_reduced']
            msp_info[key]['likli_w']=-vr.dot(invA).dot(vr)
            msp_info[key]['d']=msp_info[key]['Phi2_qd'].dot(vr)
            ### two-step update
            msp_info[key]['d_SN']=msp_info[key]['Phi2_SN_qd'].dot(vr)
            msp_info[key]['d_GR']=msp_info[key]['Phi2_GR_qd'].dot(vr)

        res=res+msp_info[key]['likli_w']/ws #!
        ldet=ldet+msp_info[key]['det_w']+(len(msp_info[key]['vx2']))*log(ws) #!

        K=[]
        if fitRN:
            vf=msp_info[key]['vf']
            K_r=msp_info[key]['K_Red']*amp_red**2/msp_info[key]['amp_red']**2*np.power(vf, (alpha_red-msp_info[key]['alpha_red'])*2.)
            K=np.append(K,K_r)
        if fitDM:
            vf=msp_info[key]['vf_dm']
            K_dm=msp_info[key]['K_DM']*amp_dm**2/msp_info[key]['amp_dm']**2*np.power(vf, (alpha_dm-msp_info[key]['alpha_dm'])*2.)
            K=np.append(K,K_dm)
        if fitSv:
            vf=msp_info[key]['vf_Sv']
            K_sv=msp_info[key]['K_Sv']*amp_sv**2/msp_info[key]['amp_dm']**2*np.power(vf, (alpha_sv-msp_info[key]['alpha_dm'])*2.)
            K=np.append(K,K_sv)


        vt=msp_info[key]['avr_bat']
        back_info=msp_info[key]['back_info']
        idx=idx+fit_Global_Efac

        msp_info[key]['K_SN']=K
    return res,ldet,vws

def UpdatePhi_SN(msp_info, user_in, vec_keys, phase_shift, SN_key={},spacing='linear',curn=0):
    # freq_band=[0,1000,2000,3000,1e99] #remove


    i=0 #pulsar count for pshift index
    for key in vec_keys:
        if user_in.freq_bin_mode==2:
        # if key in SN_key.keys():
            vsystem=SN_key[key]["vsystem"]
            vband=SN_key[key]["vband"]
            fitRN = (SN_key[key]['nf_r']>0)
            fitDM = (SN_key[key]['nf_dm']>0)
            fitSv = (SN_key[key]['nf_sv']>0)

        elif user_in.freq_bin_mode==1:
            vsystem=[]
            # nsys=0
            # nband=0
            vband=[]
            if hasattr(user_in,'fitRN') and user_in.fitRN==1:
                fitRN=1
            else:
                fitRN=0
            if hasattr(user_in,'fitDM') and user_in.fitDM==1:
                fitDM=1
            else:
                fitDM=0
            if hasattr(user_in,'fitSv') and user_in.fitSv==1:
                fitSv=1
            else:
                fitSv=0

        elif user_in.freq_bin_mode==0:
            fitRN=1
            fitDM=1              
            vsystem=[]
            vband=[]
            fitSv=0

        vr=msp_info[key]['vx2']
        back_info=msp_info[key]['back_info']
        vt=msp_info[key]['avr_bat']

        phi=np.empty((len(vr),0))
        pshift=phase_shift[i]

        if fitRN:
            phi=np.append(phi,msp_info[key]['Phi_Red'],axis=1)
            
        if fitDM:
            phi=np.append(phi,msp_info[key]['Phi_DM'],axis=1)

        if fitSv:
            phi=np.append(phi,msp_info[key]['Phi_Sv'],axis=1)

        if curn==1:
            phi=np.append(phi,msp_info[key]['Phi_GR'],axis=1)

        msp_info[key]['phi_SN']=phi
        i=+1



def generate_par_SN(user_in,vec_keys,SN_key,npar_gw=0):
    #vsystem and band enabled in next code version
    vpar=[] # RNC # better adjust and rename to vpar_sn
    parname=[]
    vpara=[]
    vparb=[]

    idx=npar_gw
    groups=[]

    for key in vec_keys:
        if user_in.freq_bin_mode==2:
            fitRN = (SN_key[key]['nf_r']>0)
            fitDM = (SN_key[key]['nf_dm']>0)
            fitSv = (SN_key[key]['nf_sv']>0)
            nDMevent=SN_key[key]['nDMevent']
            # vsystem=SN_key[key]["vsystem"]
            # vband=SN_key[key]["vband"]
            fit_Global_Efac=user_in.fitWN

        elif user_in.freq_bin_mode==1:
            nDMevent=SN_key[key]['nDMevent']
            fit_Global_Efac=user_in.fitWN 
            vsystem=[]
            vband=[]
            if hasattr(user_in,'fitRN') and user_in.fitRN==1:
                fitRN=1
            else:
                fitRN=0
            if hasattr(user_in,'fitDM') and user_in.fitDM==1:
                fitDM=1
            else:
                fitDM=0
            if hasattr(user_in,'fitSv') and user_in.fitSv==1:
                fitSv=1
            else:
                fitSv=0

        elif user_in.freq_bin_mode==0:
            fitRN=1
            fitDM=1
            fitSv=0
            nDMevent=SN_key[key]['nDMevent']
            fit_Global_Efac=0 #fixed to zero 
            # vsystem=[]
            # vband=[]            

        if fitRN:
            vpar=np.append(vpar,[-14,4])
            parname=np.append(parname,['Amp_red_'+key,'gamma_red_'+key])
            vpara=np.append(vpara,[user_in.RNampL, user_in.RNgamL])
            vparb=np.append(vparb,[user_in.RNampU, user_in.RNgamU])
            groups.append([idx,idx+1])
            idx+=2
        if fitDM:
            vpar=np.append(vpar,[-15,3])
            parname=np.append(parname,['Amp_dm_'+key,'gamma_dm_'+key])
            vpara=np.append(vpara,[user_in.DMampL, user_in.DMgamL])
            vparb=np.append(vparb,[user_in.DMampU, user_in.DMgamU])
            groups.append([idx,idx+1])
            idx+=2
        if fitSv:
            vpar=np.append(vpar,[-16,2])
            parname=np.append(parname,['Amp_sv_'+key,'gamma_sv_'+key])
            vpara=np.append(vpara,[user_in.SVampL, user_in.SVgamL])
            vparb=np.append(vparb,[user_in.SVampU, user_in.SVgamU])
            groups.append([idx,idx+1])
            idx+=2
        ## special case for J1713
        #for i in range(nDMevent):
        if (key=='J1713+0747') and (nDMevent>0):

            #hard-coded for J1713
            """
            vpar=np.append(vpar,[-16,54700,1.7,-16,57500,1])
            parname=np.append(parname,['Amp_DMe1_'+key,'t0_DMe1_'+key,'tau_DMe1_'+key,'Amp_DMe2_'+key,'t0_DMe2_'+key,'tau_DMe2_'+key])
            vpara=np.append(vpara,[-17.5,54650,0.,-17.5,57490,0.])
            vparb=np.append(vparb,[-9.5,54850,2.5,-9.5,57530,2.5])        
            groups.append([idx,idx+1,idx+2,idx+3,idx+4,idx+5])
            idx+=6
            """
            vpar=np.append(vpar,[-9,57500,1])
            parname=np.append(parname,['Amp_DMe2_'+key,'t0_DMe2_'+key,'tau_DMe2_'+key])
            vpara=np.append(vpara,[-17.5,57490,0.])
            vparb=np.append(vparb,[-9.5,57530,2.5])
            groups.append([idx,idx+1,idx+2])
            idx+=3
        ########
        #system and band noises will be enabled in the next version
        # for system in vsystem:
        #     vpar=np.append(vpar,[-16,2])
        #     parname=np.append(parname,['Amp_SN_'+system+'_'+key,'gamma_SN_'+system])
        #     vpara=np.append(vpara,[user_in.SYSampL, user_in.SYSgamL])
        #     vparb=np.append(vparb,[user_in.SYSampU, user_in.SYSgamU])
        #     groups.append([idx,idx+1])
        #     idx+=2
        # for band in vband:
        #     vpar=np.append(vpar,[-16,2])
        #     parname=np.append(parname,['Amp_BN'+str(band)+'_'+key,'gamma_BN'+str(band)])
        #     vpara=np.append(vpara,[user_in.BANDampL, user_in.BANDgamL])
        #     vparb=np.append(vparb,[user_in.BANDampU, user_in.BANDgamU])
        #     groups.append([idx,idx+1])
        #     idx+=2
        ########
        if fit_Global_Efac:
            vpar=np.append(vpar,[1.])
            parname=np.append(parname,['GlEfac_'+key])
            vpara=np.append(vpara,[user_in.GEfL])
            vparb=np.append(vparb,[user_in.GEfU])
            groups.append([idx])
            idx+=1
        vpar_range=np.dstack((vpara.transpose(),vparb.transpose()))[0,:,:]
        print('')


    return vpar,parname,vpar_range,groups


