#########################################
##        fortytwo_util_main.py        ##
#      Main functions to run the code   #
#########################################

####################################################################
from numpy import linalg
from math import *
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
from Fortytwo.fortytwo_util_prod import *
from Fortytwo.fortytwo_util_noise import *
# from Fortytwo.fortytwo_util_GWB import *
from types import SimpleNamespace
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import copy
from mpi4py import MPI
####################################################################




def read_user_input_vars(loc):

    user_input_vars = {key: value for key, value in loc.items() if not (key.startswith('__') and key.endswith('__')) and not hasattr(value, '__call__')}

    user_input_vars = {key: value for key, value in user_input_vars.items() if not (hasattr(value, '__module__') and hasattr(value, '__name__')) and not key.startswith('_i') and not key.startswith('_o') and not key.startswith('_d') and not key.startswith('In') and not key.startswith('Out')}
    
    user_in=SimpleNamespace(**user_input_vars)
    return user_in


def exe42Bayes(user_input):

    ########################
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    ########################

    if rank==0:
        print()
        print("###########################################################################################\
        \nCopyright (C) 2025: R.N. Caballero, Y.J. Guo, K.J. Lee\
        \nFORTYTWO a free software:you can redistribute it and/or modify it under the terms of\
        \nthe GNU General Public License as published by the Free Software Foundation,\
        \neither version 3 of the License, or (at your option) any later version.\
        \nThis program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;\
        \nwithout even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\
        \nSee the GNU General Public License for more details.\
        \nYou should have received a copy of the GNU General Public License along with this program.\
        \nIf not, see <https://www.gnu.org/licenses/>.\
        \nIf you use FORTYTWO for your work, please cite:\
        \nCaballero, R. N., Lee, K. J., Lentati, L., et al. 2016, MNRAS, Vol. 457, Issue 4, Page 4421\
        \n(https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.4421C/abstract)\
        \n###########################################################################################")
        print()


    if rank==0:
        
        user_in=read_user_input_vars(user_input)
        output_dir=process_outputdir(user_in)
        # print(vars(user_in).keys())
        if user_in.keeplogfile==1:
            logFile=output_dir+str(user_in.output_dir)+"_log.txt" 
            print('log file enabled - log file =',str(logFile))
            sys.stdout=DualOutput(logFile)
        print("Begin Analysis:")
        print('')    
    
        

    if rank>=0:    

        MSP_info, vkeys, SN_key, vt_all, nf, vfl, vnf, vfh, fmax, fmin = preprocess(user_in)

    else:
            MSP_info=None
            vt_all=None
            fmin=None
            fmax=None
            nf=None        


    if hasattr(user_in,'model_comparison'):
        model_comparison=user_in.model_comparison
    else:
        model_comparison=0

    if model_comparison==1:
        MSP_info_1, MSP_info_2, tmp, tmp, mxgrphi=process_covar(MSP_info, vkeys, SN_key, vt_all, nf, vfl, vnf, vfh, fmax, fmin, user_in, spacing='linear',covar_method='smem')

    else:
        tmp, tmp, mxgrphi=process_covar(MSP_info, vkeys, SN_key, vt_all, nf, vfl, vnf, vfh, fmax, fmin, user_in, spacing=user_in.spacing, covar_method='smem')

        

    vpara,vpar,parname,vpar_range,groups_par=process_par(user_in,nf,vkeys,SN_key,corr=user_in.corr,spec=user_in.spec,fixgamma=user_in.fixgamma,fitsin=user_in.fitsin,model_comparison=model_comparison,hypermodel=user_in.hypermodel)
    # sys.exit()

    if rank==0:
        np.savetxt(output_dir+'/'+str(user_in.output_dir)+'_parname.txt',parname,fmt='%s')  # for ptmcmc sampler case

    if user_in.likelihood_test:
        if rank==0:
            process_likelihoodTest(vpar, user_in, SN_key, MSP_info, vkeys, vt_all,fmin,fmax)


    ##Select sampler and run MC analysis##
    print("set and start mcmc")
    start_time = time.time()

    if model_comparison==1:
        #Model Comparison
        process_sampler(sampler=user_in.sampler,model_comparison=model_comparison,user_in=user_in,vpar=vpar,vpara=vpara,vpar_range=vpar_range,MSP_info=MSP_info,vkeys=vkeys,vt_all=vt_all,fixgamma=user_in.fixgamma,Nf=nf,fmin=fmin,fmax=fmax,fit_Global_Efac=user_in.fitWN,method=user_in.method,spacing=user_in.spacing,SN_key=SN_key,groups_par=groups_par,MSP_info_1=MSP_info_1,MSP_info_2=MSP_info_2)

    else:
        #Single CS analysis
        process_sampler(sampler=user_in.sampler,model_comparison=model_comparison,user_in=user_in,vpar=vpar,vpara=vpara,vpar_range=vpar_range,MSP_info=MSP_info,vkeys=vkeys,vt_all=vt_all,fixgamma=user_in.fixgamma,Nf=nf,fmin=fmin,fmax=fmax,fit_Global_Efac=user_in.fitWN,method=user_in.method,spacing=user_in.spacing,SN_key=SN_key,groups_par=groups_par) #sampler is defined in ftwo_config_exec. in this example, sampler=PTMCMC
    
    

    elapsed_time = time.time() - start_time
    print("sampler running time", elapsed_time)
    print("DONE")


def preprocess(user_in):
    rootpath=user_in.rootpath
    psr_list=user_in.rootpath+user_in.psr_list
    srcfile=rootpath+user_in.psr_list
    noise_style=int(user_in.noise_style)
    ephext=str(user_in.ephext)
    output_dir=user_in.rootpath+user_in.output_dir
    print("seleted output dir:",user_in.output_dir)
    # sys.exit()
    
    #Formatting basic variables#
    
    fitWN=int(user_in.fitWN)
    freq_bin_mode=int(user_in.freq_bin_mode)
    fixgamma=int(user_in.fixgamma)

    if hasattr(user_in,'model_comparison'):
        model_comparison=int(user_in.model_comparison)
    else:
        model_comparison=0
        setattr(user_in, 'model_comparison', model_comparison)

    if hasattr(user_in,'fit_phase_shifts'):
        fit_phase_shifts=int(user_in.fit_phase_shifts)
    else:
        fit_phase_shifts=0
        setattr(user_in, 'fit_phase_shifts', fit_phase_shifts)

    if hasattr(user_in,'hypermodel'):
        hypermodel=int(user_in.hypermodel)
    else:
        hypermodel=0
        setattr(user_in, 'hypermodel', hypermodel)


    freq_bin_mode=int(user_in.freq_bin_mode)
    spacing=str(user_in.spacing)
    SN=int(user_in.SN)
    if model_comparison==0 and user_in.CS == 'CURN':
        corr='None'
    else:
        corr=user_in.corr
    navr=int(user_in.navr)


    #Report some basic settings#
    # print("navr =",navr) make this report only if navr>1
    if navr>1:
        print("NOTE: You are avraging data points! navr =",navr)
    print("freq. binning algorithm =",freq_bin_mode)
    print("freq.bin spacing =",spacing)
    print("fit_Global_Efac (fitWN) =",fitWN)
    print("fixgamma =",fixgamma)
    print("psr list file =",srcfile)
    print("spectrum sampling =",user_in.spec)
    print("fit_phase_shifts =",fit_phase_shifts)
    if model_comparison==0:
        print("Analysis type: search for common signal -","signal type:",user_in.CS,"-","correlation type:",corr)
    elif model_comparison==1:
        print("Analysis type: model comparison for common signals -","signal types:",user_in.CS1,user_in.CS2,"-","correlation type:",corr)

    #Read in the data and create MSP_info#
    MSP_info=ReadinData(rootpath=rootpath,srcfile=srcfile,navr=navr, noise_style=noise_style,ephext=ephext) 
    vkeys=GetKey(srcfile=srcfile)

    #phase_shifts# move this part at flag declaration
    if fit_phase_shifts == 1:
        pshift=np.random.rand(len(vkeys))*np.pi*2
    else:
        pshift=np.zeros(len(vkeys))
    setattr(user_in, 'pshift', pshift)

    #set the f ranges and SN_key - if freq_bin_mode !=2, then SN_Key will be creates as empty
    SN_key=get_SN_key(MSP_info,vkeys,SN,freq_bin_mode,user_in)
    
    
    nf,vnf,vfl,vfh,vt_all,fmin,fmax=get_freq_limit(freq_bin_mode,MSP_info,vkeys,user_in,SN,SN_key)

    return MSP_info, vkeys, SN_key, vt_all, nf, vfl, vnf, vfh, fmax, fmin

def process_covar(MSP_info,vkeys, SN_key, vt_all, nf, vfl, vnf, vfh, fmax, fmin, user_in,spacing,covar_method='smem'):

    model_comparison=user_in.model_comparison
    pshift=user_in.pshift

    if model_comparison==0:

        if covar_method=='smem': #split set for later version
            Calculate_RedNoiseCovariance(MSP_info, vkeys, pshift, nf=vnf, freq_low_cut=vfl, freq_high_cut=vfh,spacing=spacing)
            UpdatePhi_SN(MSP_info, user_in, vkeys, pshift, SN_key)
            tmp,tmp,mxgrphi,_=PowerLaw_Covgen(vt_all, nf=nf, fmin=fmin, fmax=fmax, fc=1., alpha=0, ac=1e-20,spacing=spacing)
            n=0
            n2=0
            for key in vkeys:
                npt=len(MSP_info[key]['avr_bat'])
                MSP_info[key]['Phi_GR']=mxgrphi[n:n+npt,:]
                MSP_info[key]['Phi_GR']=MSP_info[key]['qd'].T.dot(MSP_info[key]['Phi_GR'])
    
                vr=MSP_info[key]['vx2']
                phi=MSP_info[key]['phi_SN']
    
                phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                MSP_info[key]['Phi2_SN']=phi2
                MSP_info[key]['Phi3_SN']=phi2.dot(phi)
                MSP_info[key]['Phi3_GR_SN']=phi2.dot(MSP_info[key]['Phi_GR'])
                MSP_info[key]['d_SN']=phi2.dot(vr)
    
                phi2=MSP_info[key]['Phi_GR'].T.dot(MSP_info[key]['cov_winv'])
                MSP_info[key]['Phi2_GR']=phi2
                MSP_info[key]['Phi3_GR']=phi2.dot(MSP_info[key]['Phi_GR'])
                MSP_info[key]['d_GR']=phi2.dot(vr)
    
                n1=phi.shape[1]+n2
                phi=np.append(phi,MSP_info[key]['Phi_GR'],axis=1)
                n2=phi.shape[1]+n2
                indx=np.arange(n1,n2)
                MSP_info[key]['indx_gr']=np.array(indx,dtype=int)
                phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                phi3=phi2.dot(phi)
                d=phi2.dot(vr)
    
                MSP_info[key]['Phi2']=phi2
                MSP_info[key]['Phi3']=phi3
                MSP_info[key]['d']=d
        
                invA=MSP_info[key]['cov_winv']
                MSP_info[key]['likli_w']=-vr.dot(invA).dot(vr)#-logdetA
    
                ## keep invA for determinstic signal 
                if key in SN_key.keys():
                    if (SN_key[key]['nDMevent']+user_in.fitsin)>0:
                        MSP_info[key]['covi_reduced']=MSP_info[key]['qd'].dot(invA).dot(MSP_info[key]['qd'].T)
                        MSP_info[key]['Phi2_qd']=phi2.dot(MSP_info[key]['qd'].T)
                        MSP_info[key]['Phi2_SN_qd']=MSP_info[key]['Phi2_SN'].dot(MSP_info[key]['qd'].T)
                        MSP_info[key]['Phi2_GR_qd']=MSP_info[key]['Phi2_GR'].dot(MSP_info[key]['qd'].T)
    
                MSP_info[key]['cov_winv']=0
                MSP_info[key]['qd']=0
                MSP_info[key]['Phi2']=0
                MSP_info[key]['Phi_SN']=0
                MSP_info[key]['Phi_Red']=0
                MSP_info[key]['Phi_DM']=0
                MSP_info[key]['Phi_Sv']=0
                #
                MSP_info[key]['Phi2_SN']=0
                MSP_info[key]['Phi2_GR']=0
                n=n+npt
                #print('PASS7')
            return tmp, tmp, mxgrphi
    
    elif model_comparison==1:
        #####################################
        fitsin=int(user_in.fitsin)
        list_CS=[user_in.CS1,user_in.CS2]
        #####################################
        if ('GWB2' in list_CS) or ('CURN+GWB' in list_CS):
            ncs=2
        else:
            ncs=1    
        setattr(user_in, 'ncs', ncs)    
    
        if covar_method=='smem':
            Calculate_RedNoiseCovariance(MSP_info, vkeys, pshift, nf=vnf, freq_low_cut=vfl, freq_high_cut=vfh,spacing=spacing)
            UpdatePhi_SN(MSP_info, user_in, vkeys, pshift, SN_key)
            tmp,tmp,mxgrphi,_=PowerLaw_Covgen(vt_all, nf=nf, fmin=fmin, fmax=fmax, fc=1., alpha=0, ac=1e-20,spacing=spacing)
    
            i=0
            n=0
            n2=0
            for key in vkeys:
                pshift=user_in.pshift[i] 
                npt=len(MSP_info[key]['avr_bat'])
                tmp,MSP_info[key]['Phi_GR'],_=PowerLaw_Covgen_Phi_K(MSP_info[key]['avr_bat'], phase_shift=pshift, nf=nf, fmin=fmin, fmax=fmax, fc=1., alpha=0, ac=1e-20,spacing=spacing)
    
    
                i+=1
                MSP_info[key]['Phi_GR']=MSP_info[key]['qd'].T.dot(MSP_info[key]['Phi_GR'])
                vr=MSP_info[key]['vx2']
                phi=MSP_info[key]['phi_SN']
    
                phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                MSP_info[key]['Phi2_SN']=phi2
                MSP_info[key]['Phi3_SN']=phi2.dot(phi)
                MSP_info[key]['Phi3_GR_SN']=phi2.dot(MSP_info[key]['Phi_GR'])
                MSP_info[key]['d_SN']=phi2.dot(vr)
    
                phi2=MSP_info[key]['Phi_GR'].T.dot(MSP_info[key]['cov_winv'])
                MSP_info[key]['Phi2_GR']=phi2
                MSP_info[key]['Phi3_GR']=phi2.dot(MSP_info[key]['Phi_GR'])
                MSP_info[key]['d_GR']=phi2.dot(vr)
    
                n1=phi.shape[1]+n2
                phi=np.append(phi,MSP_info[key]['Phi_GR'],axis=1)
                n2=phi.shape[1]+n2
                indx=np.arange(n1,n2)
                MSP_info[key]['indx_gr']=np.array(indx,dtype=int)
                phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                phi3=phi2.dot(phi)
                d=phi2.dot(vr)
    
                MSP_info[key]['Phi2']=phi2
                MSP_info[key]['Phi3']=phi3
                MSP_info[key]['d']=d
    
                invA=MSP_info[key]['cov_winv']
                MSP_info[key]['likli_w']=-vr.dot(invA).dot(vr)
    
                ## keep invA for determinstic signal
                if key in SN_key.keys():
                    if (SN_key[key]['nDMevent']+fitsin)>0:
                        MSP_info[key]['covi_reduced']=MSP_info[key]['qd'].dot(invA).dot(MSP_info[key]['qd'].T)
                        MSP_info[key]['Phi2_qd']=phi2.dot(MSP_info[key]['qd'].T)
                        MSP_info[key]['Phi2_SN_qd']=MSP_info[key]['Phi2_SN'].dot(MSP_info[key]['qd'].T)
                        MSP_info[key]['Phi2_GR_qd']=MSP_info[key]['Phi2_GR'].dot(MSP_info[key]['qd'].T)
                n=n+npt
    
            MSP_info_1=copy.deepcopy(MSP_info)
            for key in vkeys:
                MSP_info_1[key]['cov_winv']=0
                MSP_info_1[key]['qd']=0
                MSP_info_1[key]['Phi2']=0
                MSP_info[key]['Phi2_SN']=0
                MSP_info[key]['Phi2_GR']=0
                MSP_info_1[key]['Phi_SN']=0
                MSP_info_1[key]['Phi_Red']=0
                MSP_info_1[key]['Phi_DM']=0
                MSP_info_1[key]['Phi_Sv']=0
    
            # prepare likli_gwb2

            if ('GWB2' in list_CS):
                print('list_CS =',list_CS)
                n=0
                n3=0
                for key in vkeys:
                    npt=len(MSP_info[key]['avr_bat'])
                    MSP_info[key]['Phi_GR']=mxgrphi[n:n+npt,:]
                    MSP_info[key]['Phi_GR']=MSP_info[key]['qd'].T.dot(MSP_info[key]['Phi_GR'])
                    desmat_sse=MSP_info[key]['design_sse'].copy()
                    if desmat_sse.shape[1]>0:
                        desmat_sse=MSP_info[key]['qd'].T.dot(desmat_sse)
                        phi_sse=desmat_sse[:,index_margin]
                        desmat_sse_fit=desmat_sse[:,index_fit]
                    else:
                        ntmp=MSP_info[key]['qd'].shape[1]
                        desmat_sse=np.empty((ntmp,0))
                        phi_sse=np.empty((ntmp,0))
                        desmat_sse_fit=np.empty((ntmp,0))
    
                    vr=MSP_info[key]['vx2']
                    phi=MSP_info[key]['phi_SN']
    
                    phi_gr=np.append(MSP_info[key]['Phi_GR'],MSP_info[key]['Phi_GR'],axis=1)
    
                    phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                    MSP_info[key]['Phi2_SN']=phi2
                    MSP_info[key]['Phi3_SN']=phi2.dot(phi)
                    MSP_info[key]['Phi3_GR_SN']=phi2.dot(phi_gr)
                    MSP_info[key]['d_SN']=phi2.dot(vr)
    
                    phi2=phi_gr.T.dot(MSP_info[key]['cov_winv'])
                    MSP_info[key]['Phi2_GR']=phi2
                    MSP_info[key]['Phi3_GR']=phi2.dot(phi_gr)
                    MSP_info[key]['d_GR']=phi2.dot(vr)
    
                    n1=phi.shape[1]+n3
                    phi=np.append(phi,MSP_info[key]['Phi_GR'],axis=1)
                    n2=phi.shape[1]+n3
                    indx1=np.arange(n1,n2)
                    phi=np.append(phi,MSP_info[key]['Phi_GR'],axis=1)
                    n3=phi.shape[1]+n3
                    indx2=np.arange(n2,n3)
                    MSP_info[key]['indx_gr']=np.array(indx1,dtype=int)
                    MSP_info[key]['indx_gr1']=np.array(indx1,dtype=int)
                    MSP_info[key]['indx_gr2']=np.array(indx2,dtype=int)
                    phi=np.append(phi,phi_sse,axis=1)
                    phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                    phi3=phi2.dot(phi)
                    d=phi2.dot(vr)
    
                    MSP_info[key]['Phi2']=phi2
                    MSP_info[key]['Phi3']=phi3
                    MSP_info[key]['d']=d
    
                    phi2=desmat_sse_fit.T.dot(MSP_info[key]['cov_winv'])
                    phi3=phi2.dot(desmat_sse_fit)
                    d=phi2.dot(vr)
                    tmp=phi2.dot(phi)
    
                    MSP_info[key]['Phi3_sse']=phi3
                    MSP_info[key]['d_sse']=d
                    MSP_info[key]['Phi3_cross']=tmp
    
                    invA=MSP_info[key]['cov_winv']
                    #logdetA=MSP_info[key]['det_w']
                    MSP_info[key]['likli_w']=-vr.dot(invA).dot(vr)#-logdetA
                    ## keep invA for determinstic signal
                    if key in SN_key.keys():
                        if SN_key[key]['nDMevent']>0:
                            MSP_info[key]['covi_reduced']=MSP_info[key]['qd'].dot(invA).dot(MSP_info[key]['qd'].T)
                            MSP_info[key]['Phi2_qd']=MSP_info[key]['Phi2'].dot(MSP_info[key]['qd'].T)
                            MSP_info[key]['Phi2_SN_qd']=MSP_info[key]['Phi2_SN'].dot(MSP_info[key]['qd'].T)
                            MSP_info[key]['Phi2_GR_qd']=MSP_info[key]['Phi2_GR'].dot(MSP_info[key]['qd'].T)
    
                    MSP_info[key]['cov_winv']=0
                    MSP_info[key]['qd']=0
                    MSP_info[key]['Phi2']=0
                    MSP_info[key]['Phi2_SN']=0
                    MSP_info[key]['Phi2_GR']=0
                    MSP_info[key]['Phi_SN']=0
                    MSP_info[key]['Phi_Red']=0
                    MSP_info[key]['Phi_DM']=0
                    MSP_info[key]['Phi_Sv']=0
                    n=n+npt
                MSP_info_2=copy.deepcopy(MSP_info)    
            elif ('CURN+GWB' in list_CS):
                print('list_CS =',list_CS)
                n=0
                n3=0
                for key in vkeys:
                    npt=len(MSP_info[key]['avr_bat'])
                    MSP_info[key]['Phi_GR']=mxgrphi[n:n+npt,:]
                    MSP_info[key]['Phi_GR']=MSP_info[key]['qd'].T.dot(MSP_info[key]['Phi_GR'])
                    desmat_sse=MSP_info[key]['design_sse'].copy()
                    if desmat_sse.shape[1]>0:
                        desmat_sse=MSP_info[key]['qd'].T.dot(desmat_sse)
                        phi_sse=desmat_sse[:,index_margin]
                        desmat_sse_fit=desmat_sse[:,index_fit]
                    else:
                        ntmp=MSP_info[key]['qd'].shape[1]
                        desmat_sse=np.empty((ntmp,0))
                        phi_sse=np.empty((ntmp,0))
                        desmat_sse_fit=np.empty((ntmp,0))
                        
                    vr=MSP_info[key]['vx2']
                    phi=MSP_info[key]['phi_SN']
                    n1=phi.shape[1]+n3
                    phi=np.append(phi,MSP_info[key]['Phi_GR'],axis=1)
                    #MSP_info[key]['phi_SN']=phi
                    n2=phi.shape[1]+n3
                    #phi=np.append(phi,MSP_info[key]['Phi_GR'],axis=1)
                    phi_gr=MSP_info[key]['Phi_GR']
    
                    phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                    MSP_info[key]['Phi2_SN']=phi2
                    MSP_info[key]['Phi3_SN']=phi2.dot(phi)
                    MSP_info[key]['Phi3_GR_SN']=phi2.dot(phi_gr)
                    MSP_info[key]['d_SN']=phi2.dot(vr)
    
                    phi2=phi_gr.T.dot(MSP_info[key]['cov_winv'])
                    MSP_info[key]['Phi2_GR']=phi2
                    MSP_info[key]['Phi3_GR']=phi2.dot(phi_gr)
                    MSP_info[key]['d_GR']=phi2.dot(vr)
    
                    indx1=np.arange(n1,n2)
                    phi=np.append(phi,MSP_info[key]['Phi_GR'],axis=1)
                    n3=phi.shape[1]+n3
                    indx2=np.arange(n2,n3)
                    MSP_info[key]['indx_gr']=np.array(indx1,dtype=int)
                    MSP_info[key]['indx_gr1']=np.array(indx1,dtype=int)
                    MSP_info[key]['indx_gr2']=np.array(indx2,dtype=int)
                    phi=np.append(phi,phi_sse,axis=1)
                    phi2=phi.T.dot(MSP_info[key]['cov_winv'])
                    phi3=phi2.dot(phi)
                    d=phi2.dot(vr)
    
                    MSP_info[key]['Phi2']=phi2
                    MSP_info[key]['Phi3']=phi3
                    MSP_info[key]['d']=d
    
                    phi2=desmat_sse_fit.T.dot(MSP_info[key]['cov_winv'])
                    phi3=phi2.dot(desmat_sse_fit)
                    d=phi2.dot(vr)
                    tmp=phi2.dot(phi)
    
                    MSP_info[key]['Phi3_sse']=phi3
                    MSP_info[key]['d_sse']=d
                    MSP_info[key]['Phi3_cross']=tmp
    
                    invA=MSP_info[key]['cov_winv']
                    MSP_info[key]['likli_w']=-vr.dot(invA).dot(vr)#-logdetA
                    ## keep invA for determinstic signal
                    if key in SN_key.keys():
                        if SN_key[key]['nDMevent']>0:
                            MSP_info[key]['covi_reduced']=MSP_info[key]['qd'].dot(invA).dot(MSP_info[key]['qd'].T)
                            MSP_info[key]['Phi2_qd']=MSP_info[key]['Phi2'].dot(MSP_info[key]['qd'].T)
                            MSP_info[key]['Phi2_SN_qd']=MSP_info[key]['Phi2_SN'].dot(MSP_info[key]['qd'].T)
                            MSP_info[key]['Phi2_GR_qd']=MSP_info[key]['Phi2_GR'].dot(MSP_info[key]['qd'].T)
    
                    MSP_info[key]['cov_winv']=0
                    MSP_info[key]['qd']=0
                    MSP_info[key]['Phi2']=0
                    MSP_info[key]['Phi2_SN']=0
                    MSP_info[key]['Phi2_GR']=0
                    MSP_info[key]['Phi_SN']=0
                    MSP_info[key]['Phi_Red']=0
                    MSP_info[key]['Phi_DM']=0
                    MSP_info[key]['Phi_Sv']=0
                    n=n+npt
    
                MSP_info_2=copy.deepcopy(MSP_info)
            else:
                print('list_CS =',list_CS)
                MSP_info_2=[]
        else:
            MSP_info=None
            fmin=None
            fmax=None
            nf=None
            SN_key=None

        return MSP_info_1, MSP_info_2, tmp, tmp, mxgrphi


def get_SN_key(msp_info,vkeys,SN,freq_bin_mode,user_in):
    
    SN_key={}          

    if freq_bin_mode==0:
            
            for key in vkeys:        
                if key=='J1713+0747': 
                    if hasattr(user_in,'fitDMe') and user_in.fitDMe==1:
                        nDMevent=1
                    else:
                        nDMevent=0
                        print('Warning: Not fitting for "DM EVENTS"')
                else:
                    nDMevent=0        
                SN_key[key]={"nDMevent":nDMevent}


    elif freq_bin_mode==1:
            
            for key in vkeys:        
                if key=='J1713+0747': 
                    if hasattr(user_in,'fitDMe') and user_in.fitDMe==1:
                        nDMevent=1
                    else:
                        nDMevent=0
                        print('Warning: Not fitting for "DM EVENTS" (fitDMe flag not specified)')
                else:
                    nDMevent=0        
                SN_key[key]={"nDMevent":nDMevent}        

    elif freq_bin_mode==2:
        # i=0
        for key in vkeys:
            data=np.loadtxt('optimal_nf.txt',skiprows=0,dtype=str)
            ind=(data[:,0]==key)
            nf_r=int(data[ind,1])
            nf_dm=int(data[ind,2])
            nf_sv=int(data[ind,3])

            if nf_r == 0:
                if hasattr(user_in, 'forceRN') and user_in.forceRN==1: #forces fit of red noise, even if optimal_nf has nf_r=0
                    nf_r=user_in.force_nf_rn
                    print("Force Red Noise fitting for:",key,", with nf_rn =",nf_r)

            if key=='J1713+0747': 
                if hasattr(user_in,'fitDMe') and user_in.fitDMe==1:
                    nDMevent=1 #hard-coded for J1713+0747
                else:
                    nDMevent=0
                    print('Warning: Not fitting for "DM EVENTS" (flag fitDMe=0 or not specified)')
            else:
                nDMevent=0        
            SN_key[key]={"nDMevent":nDMevent}            


            SN_key[key]={"nf_r":nf_r,"nf_dm":nf_dm,"nf_sv":nf_sv,"vsystem":[],"vband":[],"nDMevent":nDMevent}
            
    return SN_key


def get_freq_limit(freq_bin_mode,msp_info,vkeys,user_in,SN,SN_key):

    nf=user_in.nf # unless otherwise defined in a freq_bin_mode branch, eg in freq_bin_mode==0.  
    vt_all=[]
    vfl=np.zeros((len(vkeys),1))
    i=0
    for key in vkeys:
        vt=msp_info[key]['avr_bat']
        vt_all=np.append(vt_all,vt)
        vfl[i]=1./(np.max(vt)-np.min(vt))
        i+=1


    if freq_bin_mode==0:
        nf=10
        fmin=1./(np.max(vt_all)-np.min(vt_all))
        fmax=fmin*nf
        vnf=nf*np.ones((len(vkeys),1))
        vfl=fmin*np.ones((len(vkeys),1))
        vfh=vfl*nf
   
    elif freq_bin_mode==1:

        vnfstackdeck =[]

        if hasattr(user_in,'fitRN') and user_in.fitRN==1:
            if hasattr(user_in,'nf_r'):
                nf_r=user_in.nf_r
            else:
                nf_r=15
                print('Warning: using nf_r=15 as default value (nf_r not declared by user)')

            vnfstackdeck.append(nf_r*np.ones((len(vkeys),1)))

        if hasattr(user_in,'fitDM') and user_in.fitDM==1:
            if hasattr(user_in,'nf_dm'):
                nf_dm=user_in.nf_dm
            else:
                nf_dm=15
                print('Warning: using nf_dm=15 as default value (nf_r not declared by user)')

            vnfstackdeck.append(nf_dm*np.ones((len(vkeys),1)))

        if hasattr(user_in,'fitSv') and user_in.fitSv==1:
            if hasattr(user_in,'nf_sv'):
                nf_sv=user_in.nf_sv
            else:
                nf_sv=15
                print('Warning: using nf_sv=15 as default value (nf_r not declared by user)')

            vnfstackdeck.append(nf_sv*np.ones((len(vkeys),1)))

        if len(vnfstackdeck)==0: #RNC# in next verion, allow white noise params only!
            sys.exit("Error: There isn't any pulsar noise component enabled!\nCheck fitRN, fitDM, etc flags in the config file.")
        vnf=np.hstack(vnfstackdeck)

        vfl=np.append(vfl,vfl,axis=1)
        vfh=vnf*vfl             
  
    elif freq_bin_mode==2:
        vnf=np.zeros((len(vkeys),3))
        i=0
        for key in vkeys:
            vnf[i,0]=SN_key[key]["nf_r"]
            vnf[i,1]=SN_key[key]["nf_dm"]
            vnf[i,2]=SN_key[key]["nf_sv"]
            i+=1
        for j in range(2): 
            vfl=np.append(vfl,vfl[:,0:1],axis=1)
        vfh=vnf*vfl

    ################################################

    if vnf.shape[1]==1:
        vnf=np.append(vnf,vnf,axis=1)
        vfl=np.append(vfl,vfl,axis=1)
        vfh=np.append(vfh,vfh,axis=1)

    fmin=1./(np.max(vt_all)-np.min(vt_all))
    fmax=fmin*nf    

    return nf,vnf,vfl,vfh,vt_all,fmin,fmax

    

def ReadParFile(fname, key):
    try:
        f=open(fname)
    except IOError:
        return "NOT_EXIST"

    while (True):
        str = f.readline()
        if not str:
            break
        if (str.find(key) >= 0):
            res = str.split()[1]
            return res

def ReadParameterFile(fname):
    f = open(fname)
    while (True):
        str = f.readline()
        if not str:
            break;
        if (str.find('RAJ') >= 0):
            ra = cvtRAStrtoRad(str.split()[1])
        if (str.find('DECJ') >= 0):
            dec = cvtDECStrtoRad(str.split()[1])
        if (str.split()[0]=='ELONG'):
            elon = np.float64(str.split()[1]) 
        if (str.split()[0]=='ELAT'):
            elat =  np.float64(str.split()[1]) 
    if 'elon' in locals():
        coord = SkyCoord(lon=elon*units.deg, lat=elat*units.deg,frame='barycentrictrueecliptic')#, distance=1*units.kpc)
        coord_eq=coord.icrs
        ra=coord_eq.ra.rad
        dec=coord_eq.dec.rad
    return (ra, dec)

#Average
def AverageData(mspinfo, printmsg=False):
    totaln = 0
    totalnr = 0
    for key in list(mspinfo.keys()):
        err = mspinfo[key]['err']
        vw = 1 / (err * err)
        i = 0
        navr = mspinfo[key]['navr']

        verr = np.array([])
        nlen = len(mspinfo[key]['bat'])
        w = np.array([[]])
        if nlen/navr<40:
            navr=int(nlen/40)
            if navr==0:
                navr=1
        vc=[]
        while (i + navr < nlen):
            normfact = 1. / np.sum(vw[i:i + navr])
            verr = np.append(verr, np.sqrt(normfact))
            vw_chunk = vw[i:i + navr] * normfact
            vw_chunk = vw_chunk.reshape((1, len(vw_chunk)))
            vc.append(vw_chunk)
            i = i + navr

        w=scipy.linalg.block_diag(*vc)
        normfact = 1. / np.sum(vw[i:nlen])
        verr = np.append(verr, np.sqrt(normfact))
        vw_chunk = vw[i:nlen] * normfact
        vw_chunk = vw_chunk.reshape((1, len(vw_chunk)))
        w = appdiag(w, vw_chunk)

        mspinfo[key]['avr_err'] = verr
        mspinfo[key]['w'] = w
        mspinfo[key]['avr_bat'] = w.dot(mspinfo[key]['bat'])
        mspinfo[key]['avr_post'] = w.dot(mspinfo[key]['post'])
        mspinfo[key]['avr_design'] = w.dot(mspinfo[key]['design'])
        mspinfo[key]['avr_freq'] = w.dot(mspinfo[key]['freq'])
        q,r=np.linalg.qr(mspinfo[key]['avr_design'])
        mspinfo[key]['Q']=q
        if printmsg:
            print('PSR ' + key + (' is average from %d to %d' % (len(err), len(verr))) + " points" + (
                " avr_len=%d" % navr))
        totaln += len(err)
        totalnr += len(verr)
    if totaln != totalnr:
        print('Total', totaln, "data pts, averaged to", totalnr,'data points')


        

def GetKey(srcfile):
    sources = np.genfromtxt(srcfile, dtype='str')
    return sources


def ReadinData(rootpath='./alldat/', srcfile='sources_for_GWBamp_FullEPTA', 
        blacklistfile='blacklist', rootparpath='./alldat/', navr=1, 
        noise_style=1, srcname='', Nchunk=1, Ichunk=1, dropmethod='point',obs_drop='',system='',  
        ephext='_DE440',wnext='',sse=0, annual_dm=0):
    print('RNC rootpath:',rootpath)
    print('drop method:',dropmethod)
    if len(srcname)>0:
        sources=np.array([srcname])
    else:
        sources = np.genfromtxt(srcfile, dtype='str')
        sources=np.array(sources)
    try:
        blacklist=np.genfromtxt(rootpath+blacklistfile, dtype='str',autostrip=True)
        blacklist=np.atleast_1d(blacklist)
    except IOError:
        blacklist = np.array([]) #

    for blacksheep in blacklist:
        if len(np.where(sources == blacksheep)[0])>0:
            ind = np.where(sources == blacksheep)[0][0]
            sources = np.delete(sources, ind)

    print('')

    print("Pulsars included in analysis")
    print("--------------------------")
    for i in range(1, len(sources) + 1):
        print(i, "-----" +  sources[i - 1])
    print(" ")
    print("Pulsars excluded from analysis")
    print("--------------------------")
    if len(blacklist)>0:
        for i in range(1, len(blacklist) + 1):
            print(i, "-----" + blacklist[i - 1])
    else:
        print('None')        
    print("--------------------------")
    print('')    

    if noise_style==1:
        print("Using noise parameters from temponest or temponest-style .par file")

    MSP_info = {}
    AVG_Err = ([])
    WeightMatrix = ([[]])
    Totmin=1e99
    Totmax=-1e99
    for MSP in sources:
        name = MSP
        short = MSP[:5]
        battemplate='%s/bat_info_'+ephext+wnext+'.txt'
        destemplate='%s/designmatrix_'+ephext+'.txt'
        #load data
        data_array = np.loadtxt(rootpath + battemplate % name, dtype=str)
        if data_array.shape[1]>4:
            back_info=data_array[:,4:]
        else:
            back_info=np.zeros((data_array.shape[0],1))
        data_array=data_array[:,0:4].astype(float)

        # Sort data by BAT and change time units to yr
        ind = np.argsort(data_array[:, 0])
        data_array = data_array[ind, :]
        back_info=back_info[ind]
        #correct units
        yr_data_array = np.array([data_array[:, 0].T / 365.25,
                                  data_array[:, 1].T * 3.16888e-8,
                                  data_array[:, 2].T * 3.16888e-14,
                                  data_array[:, 3].T]
        ).T
        inputBAT = yr_data_array[:, 0]
        Totmin=np.min([Totmin,  np.min(inputBAT)])
        Totmax=np.max([Totmax,  np.max(inputBAT)])
        # print("Span=", (Totmax-Totmin)*365.25)
    for MSP in sources:
        name = MSP
        short = MSP[:5]
        battemplate='%s/bat_info_'+ephext+'.txt'
        destemplate='%s/designmatrix_'+ephext+'.txt'
        bat_raw_template='%s/bat_info_'+ephext+'.txt'
        sse_des='%s/SSE_desmat_DE435.txt'

        #load data
        data_array = np.loadtxt(rootpath + battemplate % name, dtype=str)
        data_array_raw = np.loadtxt(rootpath + bat_raw_template % name, 
                dtype=str)
        if data_array.shape[1]>4:
            back_info=data_array[:,4:]
        else:
            back_info=np.zeros((data_array.shape[0],1))
        data_array=data_array[:,0:4].astype(float)

        # Sort data by BAT and change time units to yr
        ind = np.argsort(data_array[:, 0])
        data_array = data_array[ind, :]
        data_array_raw = data_array_raw[ind, :]

        back_info=back_info[ind]
        #correct units
        yr_data_array = np.array([data_array[:, 0].T / 365.25,
                                  data_array[:, 1].T * 3.16888e-8,
                                  data_array[:, 2].T * 3.16888e-14,
                                  data_array[:, 3].T]
        ).T
        inputBAT = yr_data_array[:, 0]
        inputPost = yr_data_array[:, 1]
        inputErr = yr_data_array[:, 2]
        inputFreq = yr_data_array[:, 3]
        inputErr_raw=data_array_raw[:,2].astype(float).T*3.16888e-14

        ntotpt=len(inputBAT)    #total number of data points
        nechcnk=ntotpt/Nchunk   #number of pt of each chunk
        indchunk=1
        if dropmethod=='point':
            #remove data per point bases
            ichunk_start=Ichunk*nechcnk #the starting index of ith chunk
            ichunk_end=np.min([ichunk_start+nechcnk, ntotpt ] )

            ichunk_start=int(ichunk_start)
            ichunk_end=int(ichunk_end)

            indchunk=range(ichunk_start, ichunk_end)
            indall=range(0, ntotpt)
            indchunk=np.setdiff1d(indall, indchunk)
            indchunk=np.in1d(indall,indchunk)
 
        elif dropmethod=='time':
            # print("drop: time basis")
            #per time basis
            if np.ndim(Ichunk)==0:
                indchunk= ( inputBAT<=(Totmax-Totmin)/Nchunk*(Ichunk+1)+Totmin ) & ( inputBAT>=(Totmax-Totmin)/Nchunk*(Ichunk)+Totmin )
            else:
                indchunk=np.zeros(ntotpt,dtype=bool)
                for i in Ichunk:
                    indchunki= ( inputBAT<=(Totmax-Totmin)/Nchunk*(i+1)+Totmin ) & ( inputBAT>=(Totmax-Totmin)/Nchunk*(i)+Totmin )
                    indchunk=indchunk | indchunki

            indchunk=~indchunk

        elif dropmethod=="time_cadence_point":
            # print("drop: time cadence point")
            Nchunk=2
            Ichunk=1
            indchunk= ( inputBAT<=(Totmax-Totmin)/Nchunk*(Ichunk+1)+Totmin ) & ( inputBAT>=(Totmax-Totmin)/Nchunk*(Ichunk)+Totmin )
            indchunk=~indchunk
            
            Ichunk2=0
            indchunk2=range(Ichunk2,ntotpt,2)
            indall=range(0,ntotpt)
            indchunk2=np.in1d(indall,indchunk2)

            indchunk=indchunk | indchunk2


        elif dropmethod=="time_cadence_mjd":
            # print("drop: time cadence mjd")
            Nchunk=2
            Ichunk=1
            indchunk= ( inputBAT<=(Totmax-Totmin)/Nchunk*(Ichunk+1)+Totmin ) & ( inputBAT>=(Totmax-Totmin)/Nchunk*(Ichunk)+Totmin )
            indchunk=~indchunk
            
            vt=np.round(inputBAT)
            u,counts=np.unique(vt,return_counts=True)
            u=u[::2]
            indchunk2=np.zeros(ntotpt,dtype=bool)
            for i in u:
                indchunk2=(indchunk2) | (vt==i)

            indchunk=indchunk | indchunk2


        elif dropmethod=='point2':
            # print("drop: select one in two points (point2)")
            indchunk=range(Ichunk,ntotpt,2)
            indall=range(0,ntotpt)
            indchunk=np.in1d(indall,indchunk)

        elif dropmethod=='observatory':
            # print("drop: data from observatory ",obs_drop)
            indchunk=np.array([((obs_drop in x[1]) )  for x in back_info])
            print(np.sum(indchunk))
            indchunk=~indchunk

        elif dropmethod=="obs_cadence_point":
            # print("drop: select one in two points for data from observatory ",obs_drop)
            indchunk=np.array([((obs_drop in x[1]) )  for x in back_info])

            indchunk_num=np.arange(0,ntotpt)
            indchunk_num=indchunk_num[indchunk]
            indchunk_num=indchunk_num[::2]

            indall=np.arange(0,ntotpt)
            indchunk=np.in1d(indall,indchunk_num)

            print(np.sum(indchunk))
            indchunk=~indchunk
            

        elif dropmethod=="obs_cadence_mjd":
            # print("drop: select one in two mjd for data from observatory ",obs_drop)
            indchunk=np.array([((obs_drop in x[1]) )  for x in back_info])
            indchunk_num=np.arange(0,ntotpt)
            indchunk_num=indchunk_num[indchunk]

            inputBAT_obs=inputBAT[indchunk]
            
            vt=np.round(inputBAT_obs)
            u,counts=np.unique(vt,return_counts=True)
            u=u[::2]
            indchunk2=np.zeros(len(vt),dtype=bool)
            for i in u:
                indchunk2=(indchunk2) | (vt==i)
            indchunk_num=indchunk_num[indchunk2]

            indall=np.arange(0,ntotpt)
            indchunk=np.in1d(indall,indchunk_num)

            print(np.sum(indchunk))
            indchunk=~indchunk


        elif dropmethod=='obs_time':
            # print("drop: data with MJD > 57600 from observatory ",obs_drop)
            indchunk1=np.array([((obs_drop in x[1]) )  for x in back_info])
            #indchunk2= ( inputBAT<=(Totmax-Totmin)/Nchunk*(Ichunk+1)+Totmin ) & ( inputBAT>=(Totmax-Totmin)/Nchunk*(Ichunk)+Totmin )
            indchunk2=(inputBAT>=(57600/365.25))
            indchunk= indchunk1 & indchunk2
            print(np.sum(indchunk))
            indchunk=~indchunk

        elif dropmethod=='opt_system':
            # print("drop: select data from system ",system)
            if np.ndim(system)==0:
                indchunk=np.array([((system in x[1]) )  for x in back_info])
            else:
                indchunk=np.zeros(ntotpt,dtype=bool)
                for sys in system:
                    indchunki=np.array([((sys in x[1]) )  for x in back_info]) 
                    indchunk=indchunk | indchunki

            print(np.sum(indchunk))
            if np.sum(indchunk)==0:
                continue;
            #indchunk=~indchunk

        else :
            # print("drop: time basis")
            #per time basis
            indchunk= ( inputBAT<=(Totmax-Totmin)/Nchunk*(Ichunk+1)+Totmin ) & ( inputBAT>=(Totmax-Totmin)/Nchunk*(Ichunk)+Totmin )
            indall=np.array(range(0, ntotpt))
            indchunk=indall[indchunk]
            indchunk=np.setdiff1d(indall, indchunk)

        
        condition = (inputFreq>12e99)
        # Sort design matrix by BAT
        desmat = np.loadtxt(rootpath + destemplate % name)
        desmat = desmat[ind, 0:]
        
        condition=~condition
        condition=condition & indchunk

        inputDesign = desmat
        
        # sse conditions not added in this version
        # if sse==1:
        #     desmat_sse = np.loadtxt(rootpath + sse_des % name)
        #     desmat_sse = desmat_sse[ind, 0:] * 3.16888e-8
        #     desmat_sse = desmat_sse[condition,:]
        # else:
        #     desmat_sse=np.array([[]])



        parfile=rootpath+"/"+name+"/"+name+"_"+ephext+".rn.par"
        Coordinates = ReadParameterFile(parfile)
        # convert to eliptical coordinate
        coord = SkyCoord(ra=Coordinates[0]*units.rad, dec=Coordinates[1]*units.rad)#, distance=1*units.kpc)
        coord_ecl=coord.barycentrictrueecliptic
        lon=coord_ecl.lon.rad
        lat=coord_ecl.lat.rad
 
        fc=1.


        #using tempo2 par file noise parameter
        if noise_style==1:
            SeGLBEfact=0.
            SEDGEGLBEfact=1.
            TNredNoiseLogA = ReadParFile(parfile, "TNRedAmp")
            TNredNoiseGamma = ReadParFile(parfile, "TNRedGam")
            TNdmNoiseLogA = ReadParFile(parfile, "TNDMAmp")
            TNdmNoiseGamma = ReadParFile(parfile, "TNDMGam")

            
            if TNredNoiseLogA==None: #Avoid ReadinErrors
                TNredNoiseLogA=-20
                TNredNoiseGamma=0
            TNredNoiseA = np.power(10., float(TNredNoiseLogA))
            SEDGEredNoiseA = (TNredNoiseA / np.pi) * sqrt(fc / 12.)
            SEDGEredNoiseGamma = (1. - float(TNredNoiseGamma)) / 2.
        
            
            if TNdmNoiseLogA==None: #Avoid ReadinErrors
                TNdmNoiseLogA=-20
                TNdmNoiseGamma=0
            TNdmNoiseA = np.power(10., float(TNdmNoiseLogA))
            SEDGEDMA = TNdmNoiseA * 24.*3600.*365.25
            SEDGEDMGamma = (1. - float(TNdmNoiseGamma)) / 2.

        
        #Q is the QR decompositon of avr_design
        #sv is the waveform of the single source
        #cos_o_t, and sin_o_t is the cos and sin of omega*t
        #qd is the matrix G
        #vx2=G.T.dot(avr_post)
        
        MSP_info[MSP] = {"bat": inputBAT[condition],
                         "post": inputPost[condition],
                         "err": inputErr[condition],
                         "err_raw": inputErr_raw[condition],
                         "freq":inputFreq[condition],
                         "design": inputDesign[condition,:],
                        #  "design_sse": desmat_sse,
                         "amp_red": SEDGEredNoiseA,
                         "alpha_red": SEDGEredNoiseGamma,
                         "amp_dm": SEDGEDMA,
                         "alpha_dm":SEDGEDMGamma,
                         "ra": Coordinates[0],
                         "dec": Coordinates[1],
                         "lon": lon,
                         "lat": lat,
                         "w": WeightMatrix,
                         'avr_bat': np.array([]),
                         'avr_freq': np.array([]),
                         'avr_post': np.array([]),
                         "avr_err": AVG_Err,
                         'avr_design': np.array([]),
                         'avr_post_dm_removed':np.array([]),
                         'cos_o_t':np.array([[]]),
                         'sin_o_t':np.array([[]]),
                         'Q':np.array([[]]),
                         'qd':np.array([[]]),
                         'vx2':np.array([[]]),
                         'sv':np.array([[]]),
                         'cov_r':np.array([[]]),
                         'cov_dm':np.array([[]]),
                         'det_w':1.0,
                         'cov_w':np.array([[]]),
                         'cov_winv':np.array([[]]),
                         'cov_inv':np.array([[]]),
                         'covi_reduced':np.array([[]]),
                         'back_info':back_info[condition],
                         'dm_sig':np.array([[]]),
                         'freq_lab':np.array([[]]),
                         'Phi_Red':np.array([[]]),
                         'K_Red':np.array([[]]),
                         'Phi_DM':np.array([[]]),
                         'K_DM':np.array([[]]),
                         'vf':np.array([[]]),
                         'logdetc':0,
                         'glb_efac':SEDGEGLBEfact,
                         "navr": navr,
                         "nband":5,
                         "update_noise":True}#False}

    sources=MSP_info.keys()
    vmxt=[]
    for MSP in sources:
        vmxt.append(np.max(MSP_info[MSP]["bat"]))

    maxt=np.max(np.array(vmxt))
    for MSP in sources:
        MSP_info[MSP]["bat"]=MSP_info[MSP]["bat"]#-maxt
    
    AverageData(MSP_info)

    for key in list(MSP_info.keys()):
        mxdes=MSP_info[key]['avr_design']
        n, m = mxdes.shape
        ud, sd, vd = np.linalg.svd(mxdes, full_matrices=True)
        qd = ud[:, m:]
        qdt=qd.T
        vx2=qdt.dot(MSP_info[key]['avr_post'])
        MSP_info[key]['qd']=qd
        MSP_info[key]['vx2']=vx2


    return (MSP_info)


def process_par(user_in,nf,vkeys,SN_key,corr=0,spec='PL',fixgamma=0,fitsin=0,model_comparison=0,hypermodel=0):

    

    #First set the appropriate CS values
    if model_comparison==0:
        CS=user_in.CS
    elif model_comparison==1:
        CS1=user_in.CS1
        CS2=user_in.CS2
        list_CS=[CS1,CS2]
        setattr(user_in, 'list_CS', list_CS)
        ncs=user_in.ncs #in fact, ncs is set automatically, not set by user, but ok...

    if user_in.spec=="PL":

        if model_comparison==0:

            parname=['Amp_'+CS,'gamma_'+CS]
            vpar=[-14.,3.]
            npar_gw=2
            vpara=[user_in.CSampL, user_in.CSgamL]
            vparb=[user_in.CSampU, user_in.CSgamU]
            groups=[list([0,1])]
            if fixgamma==1:
                vpar=np.delete(vpar,1)
                parname=np.delete(parname,1)
                npar_gw=1
                vpara=np.delete(vpara,1)
                vparb=np.delete(vparb,1)
                groups=[list([0])]

        elif model_comparison==1:

            if ncs==1:
                parname=['Amp_CS','gamma_CS']
                vpar=[-14.,3.]
                npar_gw=2
                vpara=[user_in.CSampL, user_in.CSgamL]
                vparb=[user_in.CSampU, user_in.CSgamU]
                groups=[list([0,1])]
                if fixgamma==1:
                    vpar=np.delete(vpar,1)
                    parname=np.delete(parname,1)
                    npar_gw=1
                    vpara=np.delete(vpara,1)
                    vparb=np.delete(vparb,1)
                    groups=[list([0])]
                    
            elif ncs==2:
                vpar=[-14.,3.,-16.,2.]
                parname=['Amp_CS1','gamma_CS1','Amp_CS2_','gamma2_CS2']
                npar_gw=4
                vpara=np.array([user_in.CS1ampL, user_in.CS1gamL, user_in.CS2ampL, user_in.CS2gamL])
                vparb=np.array([user_in.CS1ampU, user_in.CS1gamU, user_in.CS2ampU, user_in.CS2gamU])
                groups=[list([0,1]),list([2,3])]
                if fixgamma==1:
                    vpar=np.delete(vpar,[1,3])
                    parname=np.delete(parname, [1,3])
                    npar_gw=2
                    vpara=np.delete(vpara,[1,3])
                    vparb=np.delete(vparb,[1,3])
                    groups=[list([0]),list([1])]


    elif spec=="brokenPL":

        if model_comparison==0:
            
            vpar=[-14.,3.,-8.]
            parname=['Amp_'+CS,'gamma_'+CS,'fb_'+CS] 
            npar_gw=3
            vpara=np.array([user_in.CSampL, user_in.CSgamL, user_in.CSfbL])
            vparb=np.array([user_in.CSampU, user_in.CSgamU, user_in.CSfbU])
            groups=[list([0,1,2])]
            if fixgamma==1:
                vpar=np.delete(vpar,1)
                parname=np.delete(parname,1)
                npar_gw=2
                vpara=np.delete(vpara,1)
                vparb=np.delete(vparb,1)
                groups=[list([0,1])] 

        elif model_comparison==1:

            if ncs==1:
                vpar=[-14.,3.,-8.]
                parname=['Amp_CS','gamma_CS','fb_CS'] 
                npar_gw=3
                vpara=np.array([user_in.CSampL, user_in.CSgamL, user_in.CSfbL])
                vparb=np.array([user_in.CSampU, user_in.CSgamU, user_in.CSfbU])
                groups=[list([0,1,2])]
                if fixgamma==1:
                    vpar=np.delete(vpar,1)
                    parname=np.delete(parname,1)
                    npar_gw=2
                    vpara=np.delete(vpara,1)
                    vparb=np.delete(vparb,1)
                    groups=[list([0,1])] 

            elif ncs==2:
                vpar=[-14.,3.,-8.,-15.,2.,-8.]
                parname=['Amp_CS1','gamma_CS1','fb_CS1','Amp_CS2','gamma_CS2','fb_CS2']
                npar_gw=6
                vpara=np.array([user_in.CS1ampL, user_in.CS1gamL, user_in.CS1fbL, user_in.CS2ampL, user_in.CS2gamL, user_in.CS2fbL])
                vparb=np.array([user_in.CS1ampU, user_in.CS1gamU, user_in.CS1fbU, user_in.CS2ampU, user_in.CS2gamU, user_in.CS2fbU])
                groups=groups=[list([0,1,2]),list([3,4,5])]
                if fixgamma==1:
                    vpar=np.delete(vpar,[1,4])
                    parname=np.delete(parname,[1,4])
                    npar_gw=2
                    vpara=np.delete(vpara,[1,4])
                    vparb=np.delete(vparb,[1,4])
                    groups=[list([0,1]),list([2,3])] 

    elif spec=="freespec":
        if model_comparison==0:
            CS=user_in.CS
            vpar=np.ones(nf)*(-17)
            parname=np.array(['Amp_fbin_'+CS+'_'+str(i) for i in range(nf)])
            npar_gw=nf
            vpara=np.ones(nf)*(user_in.FSampL)
            vparb=np.ones(nf)*(user_in.FSampU)
            # vpara=np.ones(nf)*(-18.)
            # vparb=np.ones(nf)*(-10.)
            groups=[list(np.arange(0, nf))]
        else:
            sys.exit('freespec not supported for model comparison, yet')



    idx=npar_gw

    if fitsin:
        vpar=np.append(vpar,[-15,-8,1])
        parname=np.append(parname,['Amp_sin','f_sin','phase_sin'])
        npar_gw+=3
        vpara=np.append(vpara,[user_in.SINEampL,user_in.SINEfreqL,user_in.SINEphL])
        vparb=np.append(vparb,[user_in.SINEampU,user_in.SINEfreqU,user_in.SINEphU])
        groups.append([idx,idx+1,idx+2])
        idx+=3

    vpar_SN,parname_SN,vpar_range_SN,groups_SN=generate_par_SN(user_in,vkeys, SN_key,npar_gw)
    vpar=np.append(vpar,vpar_SN)
    parname=np.append(parname,parname_SN)
    groups.extend(groups_SN)

    vpara_SN=vpar_range_SN[:,0]
    vparb_SN=vpar_range_SN[:,1]
    vpara=np.append(vpara,vpara_SN)
    vparb=np.append(vparb,vparb_SN)
    idx=len(vpar)

    if corr==-1:
        vpar=np.append(vpar,(np.random.rand(7)*2-1)*0.01) #RNC 7 points: this number will be dynamic in next version
        for jj in np.arange(0,7):
            parname=np.append(parname,'corr_'+str(jj))
        vpara=np.append(vpara,np.ones(7)*-1.)
        vparb=np.append(vparb,np.ones(7)*1)
        groups.append(list(np.arange(idx,idx+7)))
        idx+=7

    if hypermodel:
        vpar=np.append(vpar,[1.])
        parname=np.append(parname,['nmodel'])
        vpara=np.append(vpara,[-0.5])
        vparb=np.append(vparb,[1.5])
        groups.append([idx])
        idx+=1

    vpar_range=np.dstack((vpara.transpose(),vparb.transpose()))[0,:,:]

    return vpara,vpar,parname,vpar_range,groups

def process_outputdir(user_in):
    rootpath=user_in.rootpath
    output_dir=user_in.rootpath+user_in.output_dir+'/'
    print("Analysis results directory:",output_dir)
    if os.path.isdir(output_dir):
        print('--directory exists')
        print('')
        if user_in.overwrite_protection==1:
            print("Selected output directory ("+output_dir+") already exist! Code terminated to avoid overwriting previous results.")
            print('')
            sys.exit()
        elif user_in.overwrite_protection==0:
                print('No data overwrite protection for output files selected. You may be overwriting previous results or mixing results.')
                print('')
        else:
            print('No valid value for data overwrite protection (0 or 1 accepted).')
            print('Code terminated')
            print('')
            sys.exit()        
    else:
        os.system("mkdir -v "+output_dir)
        print('Created the output directory')
        print('')


    return output_dir

def process_likelihoodTest(vpar, user_in, SN_key, MSP_info, vkeys, vt_all,fmin,fmax):
    #RNC! Warning! I think it works correct with with ncs=2, I want to check again.
    print('#######################################################################################################################')
    print('Likelihood test: Value and calculations times')
    print('#######################################################################################################################')
    print('---------------------------------------------')
    print('Used parameter matrix:',vpar)
    print('---------------------------------------------')

    mxtheta=mxtheta_calc(MSP_info)
    print()
    # print("mxtheta=",mxtheta)
    t0 = time.time()

    if user_in.model_comparison == 1 and 'CURN+GWB' in user_in.list_CS:
        for i in range(10):
            res1=likli_curn_gwb_sn_new(vpar, user_in.freq_bin_mode, user_in, MSP_info,vkeys,vt_all,mxtheta,index_margin=[],corr=user_in.corr,SN_key=SN_key,fixgamma=user_in.fixgamma,Nf=user_in.nf,fmin=fmin,fmax=fmax,method=user_in.method,spec=user_in.spec, spacing=user_in.spacing)
        t1=time.time()

        print(user_in.CS1+"-"+user_in.CS2+" Liklihood: approx. likelihood =",res1," - approx. calculation time (secs) =", (t1-t0)/10)

    elif user_in.model_comparison == 1 and 'GWB2' in user_in.list_CS:    
        for i in range(10):
            res1=likli_gwb2_sn_new(vpar, user_in.freq_bin_mode, user_in, MSP_info,vkeys,vt_all,mxtheta,index_margin=[],corr=user_in.corr,SN_key=SN_key,fixgamma=user_in.fixgamma,Nf=user_in.nf,fmin=fmin,fmax=fmax,method=user_in.method,spec=user_in.spec, spacing=user_in.spacing)
        t1=time.time()

        print(user_in.CS1+"-"+user_in.CS2+" Liklihood: approx. likelihood =",res1," - approx. calculation time (secs) =", (t1-t0)/10)

    else:
        for i in range(10):
            res1=likli_curn_sn(vpar,user_in.freq_bin_mode, user_in, MSP_info,vkeys,vt_all,SN_key=SN_key,fixgamma=user_in.fixgamma,Nf=user_in.nf,fmin=fmin,fmax=fmax, method=user_in.method,spec=user_in.spec,spacing=user_in.spacing)
        t1=time.time()
        
        for i in range(10):
            res2=likli_gwb_sn(vpar,user_in.freq_bin_mode, user_in, MSP_info,vkeys,vt_all,mxtheta,corr=user_in.corr,SN_key=SN_key,fixgamma=user_in.fixgamma,Nf=user_in.nf,fmin=fmin,fmax=fmax,method=user_in.method,spec=user_in.spec, spacing=user_in.spacing)
        t2=time.time()

        for i in range(10):
            res3=likli_gwb_sn_new(vpar,user_in.freq_bin_mode, user_in, MSP_info,vkeys,vt_all,mxtheta,corr=user_in.corr,SN_key=SN_key,fixgamma=user_in.fixgamma,Nf=user_in.nf,fmin=fmin,fmax=fmax,method=user_in.method,spec=user_in.spec, spacing=user_in.spacing)
        t3=time.time()

        print("CURN Liklihood: approx. likelihood =",res1," - approx. calculation time (secs) =", (t1-t0)/10)
        print("GWB1 Liklihood: approx. likelihood =",res2," - approx. calculation time (secs) =", (t2-t1)/10)
        print("GWB2 Liklihood: approx. likelihood =",res3," - approx. calculation time (secs) =", (t3-t2)/10)

    print('#######################################################################################################################')
    print('#######################################################################################################################')


def process_sampler(sampler, model_comparison, user_in, vpar, vpara, vpar_range, MSP_info, vkeys, vt_all, fixgamma, Nf,fmin,fmax,fit_Global_Efac,method,spacing,SN_key,groups_par,MSP_info_1=[],MSP_info_2=[]):

    ##Sampler setup##
    fbm=user_in.freq_bin_mode

    if model_comparison!=1 and user_in.CS == 'CURN':
        gwb_lik='None'
    elif model_comparison==1:
        gwb_lik=1
    else:
        gwb_lik=user_in.gwb_lik

    output_dir=user_in.output_dir
    mxtheta=mxtheta_calc(MSP_info)
    if sampler == "PTMCMC":
        index_margin=[] # RNC # this is hardcoded only for sse=3 
        index_fit=[]    

        def mylogprior(cube):
            ndim=len(cube)
            cube2=np.zeros(ndim)
            for i in range(0,ndim):
                if (cube[i]>vpar_range[i,0]) & (cube[i]<vpar_range[i,1]):
                    cube2[i]=np.log(1./(vpar_range[i,1]-vpar_range[i,0]))
                else:
                    cube2[i]=-np.inf
            return np.sum(cube2)

        if model_comparison!=1:
    
            def myloglike(cube):
                res=loglike(cube,fbm,user_in,MSP_info,vkeys,vt_all,mxtheta,CS=user_in.CS,corr=user_in.corr,SN_key=SN_key,gwb_lik=gwb_lik,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax,method=user_in.method,fitsin=user_in.fitsin,spec=user_in.spec,spacing=user_in.spacing)

                return res

        elif model_comparison==1:

            def myloglike(cube):
                res=loglike_MC(cube,fbm,user_in,MSP_info_1,MSP_info_2,vkeys,vt_all,mxtheta,index_margin, list_CS=user_in.list_CS,corr=user_in.corr,SN_key=SN_key,gwb_lik=gwb_lik,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax,method=user_in.method,fitsin=user_in.fitsin,spec=user_in.spec,spacing=user_in.spacing)
                return res

        def draw_from_prior(x,iter,beta):
            """
            uniform distribution prior draw.
            """
            q = x.copy()

            # randomly choose parameter
            idx=np.random.choice(ndim)
            q[idx] = np.random.uniform(vpar_range[idx,0],vpar_range[idx,1])
            lqxy = 0

            # forward-backward jump probability
            if (x[idx]>vpar_range[idx,0]) & (x[idx]<vpar_range[idx,1]):
                lqxy=0
            else:
                lqxy=-np.inf
            return q, float(lqxy)
        print("start PTMCMC")
        
        resume=True
        ndim=len(vpar)

        groups=[list(np.arange(0,ndim))]
        groups.extend(groups_par)

        # initial jump covariance matrix
        cov = np.diag(np.ones(ndim) * 1**2) ## used to be 0.1
        sampler = ptmcmc(ndim, myloglike, mylogprior, cov,
                             groups=groups,
                             outDir=output_dir, resume=resume)
        # additional jump proposals?
        # always add draw from prior
        sampler.addProposalToCycle(draw_from_prior, 5)
        x0 = vpar

        if hasattr( user_in,'ptmcmc_settings'):
            SE=user_in.ptmcmc_settings
            # N,SCAMweight, AMweight, DEweight = int(SE[0]),SE[1],SE[2],SE[3]
            N = int(SE[0])
            print('Custom sampler settings:\n N,SCAMweight, AMweight, DEweight =',N,SE[1], SE[2], SE[3])
            sampler.sample(x0, N, SCAMweight=SE[1], AMweight=SE[2], DEweight=SE[3])

        else:
            N = int(3e6)
            sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50) 


    elif sampler == "MN":
        print("MC sampler = PyMultinest")
        sys.exit("Not implemented in this version.")
    elif sampler == "PC":
        print("MC sampler = PyPolychord")
        sys.exit("Not implemented in this version.")
    elif sampler == "FTMH":
        print("MC sampler = 42's Metropolis-Hastings")
        sys.exit("Not implemented in this version.")
    else:
        sys.exit("Error: no existing sampler selected. Options are MN, PL, or FTMH. Code terminated")