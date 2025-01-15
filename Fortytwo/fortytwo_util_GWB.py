#########################################
##         fortytwo_util_GWB.py        ##
#      GWB and CURN related functions   #
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
#from Fortytwo.fortytwo_util_noise import *
import Fortytwo.fortytwo_util_noise
####################################################################


#the general mxtheta function##
def mxtheta_calc(MSP_info):
    vkeys = list(MSP_info.keys())
    mxtheta=np.zeros((len(vkeys), len(vkeys)))
    for i in range(0,len(vkeys)):
        raa=MSP_info[vkeys[i]]['ra']
        deca=MSP_info[vkeys[i]]['dec']
        for j in range(i+1,len(vkeys)):
            if i!=j:
                rab=MSP_info[vkeys[j]]['ra']
                decb=MSP_info[vkeys[j]]['dec']
                theta=acos(cos(deca)*cos(decb)*cos(raa-rab)+sin(deca)*sin(decb))
            else:
                theta=0
            mxtheta[i,j]=theta
            mxtheta[j,i]=theta
    return mxtheta

##The Hellings-Downs function##
def HD_func(theta):
    theta2 = np.copy(theta)
    theta2[theta == 0] = 1e-9
    res = (3. + np.cos(theta2)) / 8. - 3. / 2. * (np.cos(theta2) - 1) * np.log(np.sin(theta2 * 0.5))
    res[theta == 0] = 1
    return res

#######################
def likli_curn_sn(vpar, fbm, user_in, msp_info, vec_keys, vt_all, SN_key={}, fixgamma=0, Nf=2^5, fmin=1/20., fmax=365.25*0.5, method='CH',spec='PL',spacing='linear'):
    fitsin=0 ##RNC to update
    if spec=="PL":
        if fixgamma:
            npar_gw=1
            alpha_gr=-2./3-1 # different parametrizations is a legacy issue that's resolved in the next version?
        else:
            npar_gw=2
            alpha_gr=(1-vpar[1])/2.
        amp_gr=10.**vpar[0]*(12*np.pi**2)**-0.5
        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)
    elif spec=="brokenPL":
        if fixgamma:
            npar_gw=1
            alpha_gr=-2./3-1
        else:
            npar_gw=2
            alpha_gr=(1-vpar[1])/2.
        amp_gr=10.**vpar[0]*(12*np.pi**2)**-0.5

        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)

        fb=10**vpar[2]*365.25*24*3600
        kappa=0.1
        delta=0
        vx=np.arange(0, int(Nf) , dtype=np.float64)/(Nf-1)
        vf=fmin+(fmax-fmin)*vx
        vf2 = np.empty((vf.size*2,), dtype=np.float64)
        vf2[0::2] = vf
        vf2[1::2] = vf
        K_gr=K_gr*((1+(vf2/fb)**(1./kappa))**(kappa*(vpar[1]-delta)))
        npar_gw+=1
    elif spec=="freespec":
        vx=np.arange(0, int(Nf) , dtype=np.float64)/(Nf-1)
        romb_coef=np.ones(len(vx))/len(vx)
        K_gr = np.empty((Nf*2,), dtype=np.float64)
        K_gr[0::2] = 10**(vpar[0:Nf]*2)/(12*np.pi**2)*(fmax-fmin)*romb_coef
        K_gr[1::2] = 10**(vpar[0:Nf]*2)/(12*np.pi**2)*(fmax-fmin)*romb_coef
        npar_gw=Nf

    if fitsin>0:
        amp_sin=10.**vpar[npar_gw]
        fsin=10.**vpar[npar_gw+1]
        phi_sin=vpar[npar_gw+2]
        npar_gw+=3

    ## mainly to get K for all single psr noise, also update res,ldet for deterministic signal (DM exp dip)
    res,ldet,vws=Fortytwo.fortytwo_util_noise.UpdateK_SN(vpar, user_in, fbm, msp_info, vec_keys, npar_gw, SN_key,spacing=spacing,fitsin=fitsin)

    for key in vec_keys:
        ws=vws[key]
        K=msp_info[key]['K_SN']
        K=np.append(K,K_gr)

        phi3=msp_info[key]['Phi3']/ws
        d=msp_info[key]['d']/ws
        logdetK=np.sum(np.log(K))

        B=phi3+np.diag(1./K)

        try:
            if method=='QR':
                q, r = np.linalg.qr(B)
                logdetB = np.sum(np.log(np.abs(np.diag(r))));
                res = res+d.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d)))
            elif method=='CH':
                lc = scipy.linalg.cho_factor(B)
                logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
                res = res+d.T.dot(scipy.linalg.cho_solve(lc, d))
            else:
                invB=np.linalg.inv(B)
                logdetB=np.linalg.slogdet(B)[1]
                res=res+d.T.dot(invB).dot(d)
        except np.linalg.LinAlgError:
            print("LinAlgError", vpar)
            return -np.inf

        if res>=0:
            print("non neg",vpar,res)
            return -np.inf

        ldet=ldet+logdetB+logdetK

    res=(res-ldet)*0.5
    return res

def likli_gwb_sn(vpar, fbm, user_in, msp_info, vec_keys, vt_all, mxtheta, corr=2, SN_key={}, fixgamma=0, Nf=2^5, fmin=1/20., fmax=365.25*0.5, method='CH',spec='PL', spacing='linear'):
    if spec=="PL":
        if fixgamma==1:
            npar_gw=1
            alpha_gr=13/3.
        else:
            npar_gw=2
            alpha_gr=vpar[1]
        alpha_gr=(1-alpha_gr)/2.
        amp_gr=np.power(10., vpar[0])*(12*np.pi**2)**-0.5
        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)
    elif spec=="brokenPL":
        if fixgamma:
            npar_gw=1
            alpha_gr=13/3.
        else:
            npar_gw=2
            alpha_gr=vpar[1]
        alpha_gr=(1-alpha_gr)/2.
        amp_gr=np.power(10., vpar[0])*(12*np.pi**2)**-0.5
        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)

        fb=10**vpar[2]*365.25*24*3600
        kappa=0.1
        delta=0
        vx=np.arange(0, int(Nf) , dtype=np.float64)/(Nf-1)
        vf=fmin+(fmax-fmin)*vx
        vf2 = np.empty((vf.size*2,), dtype=np.float64)
        vf2[0::2] = vf
        vf2[1::2] = vf
        K_gr=K_gr*((1+(vf2/fb)**(1./kappa))**(kappa*(vpar[1]-delta)))
        npar_gw+=1 
    elif spec=='freespec':
        vx=np.arange(0, int(Nf) , dtype=np.float64)/(Nf-1)
        romb_coef=np.ones(len(vx))/len(vx)
        K_gr = np.empty((Nf*2,), dtype=np.float64)
        K_gr[0::2] = 10**(vpar[0:Nf]*2)/(12*np.pi**2)*(fmax-fmin)*romb_coef
        K_gr[1::2] = 10**(vpar[0:Nf]*2)/(12*np.pi**2)*(fmax-fmin)*romb_coef
        npar_gw=Nf

    ## mainly to get K for all single psr noise, also update res,ldet for deterministic signal (DM exp dip)
    res,ldet,vws=Fortytwo.fortytwo_util_noise.UpdateK_SN(vpar, user_in, fbm, msp_info, vec_keys, npar_gw, SN_key,spacing=spacing,fitsin=0)

    vphi3=[]
    d=[]
    K=[]
    indx=[]
    for key in vec_keys:
        ws=vws[key]
        K_SN=msp_info[key]['K_SN']
        K=np.append(K,K_SN)
        K=np.append(K,K_gr)
        vphi3.append(msp_info[key]['Phi3']/ws)
        d=np.append(d,msp_info[key]['d']/ws)
        indx=np.append(indx,msp_info[key]['indx_gr'])
    phi3=scipy.linalg.block_diag(*vphi3)
    # index for K_gr
    indx=np.array(indx,dtype=int)

    if corr==0:
        mxH=mxtheta*0+1
        mxH+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==1:
        mxH=np.cos(mxtheta)
        mxH+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==2:
        mxH=HD_func(mxtheta)
    elif corr==-1: 
        y=vpar[-7:] #RNC 7 points: this number will be dynamic in next version
        mxH=np.zeros(mxtheta.shape)
        for i in range(0,len(vec_keys)):
            mxH[i,i]=1
            for j in range(i+1,len(vec_keys)):
                mxH[i,j]=y[int(np.floor(mxtheta[i,j]/(np.pi/7)))]
                mxH[j,i]=mxH[i,j]

    n=K_gr.shape[0]
    invK=np.diag(1./K)
    logdetK=np.sum(np.log(K))-np.sum(np.log(K[indx]))
    K=np.diag(K)
    invH=np.linalg.inv(mxH)
    t1=time.time()
    for i in range(0,len(vec_keys)):
        for j in range(0,len(vec_keys)):
            indxi=msp_info[vec_keys[i]]['indx_gr']
            indxj=msp_info[vec_keys[j]]['indx_gr']
            indx_ij=np.ix_(indxi,indxj)
            K[indx_ij]= np.diag(K_gr)*mxH[i,j]
            invK[indx_ij]= np.diag(1./K_gr)*invH[i,j]
    indx2=np.ix_(indx,indx)        
    logdetK+=np.linalg.slogdet(K[indx2])[1]
    t2=time.time()

    B=phi3+invK

    try:
        if method=='QR':
            q, r = np.linalg.qr(B)
            logdetB = np.sum(np.log(np.abs(np.diag(r))))
            res = res+d.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d)))
        elif method=='CH':
            lc = scipy.linalg.cho_factor(B)
            logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
            res = res+d.T.dot(scipy.linalg.cho_solve(lc, d))
        else:
            invB=np.linalg.inv(B)
            logdetB=np.linalg.slogdet(B)[1]
            res=res+d.T.dot(invB).dot(d)
    except np.linalg.LinAlgError:
        print("LinAlgError", vpar)
        return -np.inf

    if res>=0:
        print("non neg",vpar,res)
        return -np.inf
    ldet=ldet+logdetB+logdetK

    res=(res-ldet)*0.5

    return res

def likli_gwb_sn_new(vpar, fbm, user_in, msp_info, vec_keys, vt_all, mxtheta, corr=2, SN_key={}, fixgamma=0, Nf=2^5, fmin=1/20., fmax=365.25*0.5, method='CH',spec='PL',spacing='linear'):

    if spec=="PL":
        if fixgamma:
            npar_gw=1
            alpha_gr=13/3.
        else:
            npar_gw=2
            alpha_gr=vpar[1]
        alpha_gr=(1-alpha_gr)/2.
        amp_gr=np.power(10., vpar[0])*(12*np.pi**2)**-0.5
        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)

    elif spec=="brokenPL":
        if fixgamma:
            npar_gw=1
            alpha_gr=13/3.
        else:
            npar_gw=2
            alpha_gr=vpar[1]
        alpha_gr=(1-alpha_gr)/2.
        amp_gr=np.power(10., vpar[0])*(12*np.pi**2)**-0.5 
        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)

        fb=10**vpar[2]*365.25*24*3600
        kappa=0.1
        delta=0
        vx=np.arange(0, int(Nf) , dtype=np.float64)/(Nf-1)
        vf=fmin+(fmax-fmin)*vx
        vf2 = np.empty((vf.size*2,), dtype=np.float64)
        vf2[0::2] = vf
        vf2[1::2] = vf
        K_gr=K_gr*((1+(vf2/fb)**(1./kappa))**(kappa*(vpar[1]-delta)))
        npar_gw+=1            
    elif spec=='freespec':
        vx=np.arange(0, int(Nf) , dtype=np.float64)/(Nf-1)
        romb_coef=np.ones(len(vx))/len(vx)
        K_gr = np.empty((Nf*2,), dtype=np.float64)
        K_gr[0::2] = 10**(vpar[0:Nf]*2)/(12*np.pi**2)*(fmax-fmin)*romb_coef
        K_gr[1::2] = 10**(vpar[0:Nf]*2)/(12*np.pi**2)*(fmax-fmin)*romb_coef
        npar_gw=Nf        

    ## mainly to get K for all single psr noise, also update res,ldet for deterministic signal (DM exp dip)
    res,ldet,vws=Fortytwo.fortytwo_util_noise.UpdateK_SN(vpar, user_in, fbm,  msp_info, vec_keys, npar_gw, SN_key, spacing=spacing,fitsin=0) 
    vphi3=[]
    d_gr=[]
    K=[]
    indx=[]
    for key in vec_keys:
        ws=vws[key]
        K_SN=msp_info[key]['K_SN']
        phi3=msp_info[key]['Phi3_SN']/ws
        d=msp_info[key]['d_SN']/ws
        logdetK=np.sum(np.log(K_SN))
        B=phi3+np.diag(1./K_SN)

        try:
            if method=='CH':
                lc = scipy.linalg.cho_factor(B)
                logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
                invBd=scipy.linalg.cho_solve(lc, d)
                res = res+d.T.dot(invBd)
                phi3_g=msp_info[key]['Phi3_GR']/ws
                phi3_gs=msp_info[key]['Phi3_GR_SN']/ws
                phi3_gr_r=phi3_g-phi3_gs.T.dot(scipy.linalg.cho_solve(lc, phi3_gs))
                d_gr_r=msp_info[key]['d_GR']/ws-phi3_gs.T.dot(invBd)
            
            elif method=='QR':
                q, r = np.linalg.qr(B)
                logdetB = np.sum(np.log(np.abs(np.diag(r))))
                res = res+d.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d)))
                
                phi3_g=msp_info[key]['Phi3_GR']/ws
                phi3_gs=msp_info[key]['Phi3_GR_SN']/ws
                
                phi3_gr_r=phi3_g-phi3_gs.T.dot(scipy.linalg.solve_triangular(r, phi3_gs))

                d_gr_r=msp_info[key]['d_GR']/ws-phi3_gs.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d)))            

            else:
                invB=np.linalg.inv(B)
                logdetB=np.linalg.slogdet(B)[1]
                res=res+d.T.dot(invB).dot(d)
        except np.linalg.LinAlgError:
            print("LinAlgError1", vpar)
            return -np.inf
        ldet=ldet+logdetB+logdetK
        # print("vpar=",vpar) 

        K=np.append(K,K_gr)

        vphi3.append(phi3_gr_r) 
        d_gr=np.append(d_gr,d_gr_r) 
    phi3=scipy.linalg.block_diag(*vphi3)
    # index for K_gr
    indx=np.array(indx,dtype=int)

    if corr==0:
        mxH=mxtheta*0+1
        mxH+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==1:
        mxH=np.cos(mxtheta)
        mxH+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==2:
        mxH=HD_func(mxtheta)
    elif corr==-1:
        y=vpar[-7:] #RNC 7 points: this number will be dynamic in next version
        mxH=np.zeros(mxtheta.shape)
        for i in range(0,len(vec_keys)):
            mxH[i,i]=1
            for j in range(i+1,len(vec_keys)):
                mxH[i,j]=y[int(np.floor(mxtheta[i,j]/(np.pi/7)))]
                mxH[j,i]=mxH[i,j]

    n=K_gr.shape[0]
    invK=np.diag(1./K)
    K=np.diag(K)
    invH=np.linalg.inv(mxH)
    t1=time.time()

    for i in range(0,len(vec_keys)):
        for j in range(0,len(vec_keys)):
            indxi=np.arange(i*Nf*2,(i+1)*Nf*2)
            indxj=np.arange(j*Nf*2,(j+1)*Nf*2)
            indx_ij=np.ix_(indxi,indxj)
            K[indx_ij]= np.diag(K_gr)*mxH[i,j]
            invK[indx_ij]= np.diag(1./K_gr)*invH[i,j]

    logdetK=np.linalg.slogdet(K)[1]
    t2=time.time()

    B=phi3+invK

    try:
        if method=='QR':
            q, r = np.linalg.qr(B)
            logdetB = np.sum(np.log(np.abs(np.diag(r))))
            res = res+d_gr.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d_gr)))
        elif method=='CH':
            lc = scipy.linalg.cho_factor(B)
            logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
            res = res+d_gr.T.dot(scipy.linalg.cho_solve(lc, d_gr))
        else:
            invB=np.linalg.inv(B)
            logdetB=np.linalg.slogdet(B)[1]
            res=res+d_gr.T.dot(invB).dot(d_gr)
    except np.linalg.LinAlgError:
        print("LinAlgError2", vpar)
        return -np.inf

    if res>=0:
        print("non neg",vpar,res)
        return -np.inf
    ldet=ldet+logdetB+logdetK

    res=(res-ldet)*0.5

    return res


def likli_gwb2_sn_new(vpar, fbm, user_in, msp_info, vec_keys, vt_all, mxtheta, index_margin=[], corr=2, SN_key={}, fixgamma=0, Nf=2^5, fmin=1/20., fmax=365.25*0.5, method='CH',spec='PL',spacing='linear'):

    if spec=="PL":
        if fixgamma==1:
            npar_gw=1
            alpha_gr=13/3.
        else:
            npar_gw=2
            alpha_gr=vpar[1]
        alpha_gr=(1-alpha_gr)/2.
        amp_gr=np.power(10., vpar[0])*(12*np.pi**2)**-0.5
        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)
        alpha_gr=vpar[npar_gw+1]
        alpha_gr=(1-alpha_gr)/2.
        amp_gr=np.power(10., vpar[npar_gw])*(12*np.pi**2)**-0.5
        K_gr2=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)
        npar_gw+=2
    else:
        sys.exit('Model comparison mode currently available only for spec PL (change flag and try again.)')  

    ####################################

    ## mainly to get K for all single psr noise, also update res,ldet for deterministic signal (DM exp dip)
    res,ldet,vws=Fortytwo.fortytwo_util_noise.UpdateK_SN(vpar, user_in, fbm, msp_info, vec_keys, npar_gw, SN_key,spacing=spacing,fitsin=0)

    vphi3=[]
    d_gr=[]
    K=[]
    indx1=[]
    indx2=[]
    for key in vec_keys:
        ws=vws[key]
        K_SN=msp_info[key]['K_SN']
        phi3=msp_info[key]['Phi3_SN']/ws
        d=msp_info[key]['d_SN']/ws
        logdetK=np.sum(np.log(K_SN))
        B=phi3+np.diag(1./K_SN)

        try:
            if method=='QR':
                q, r = np.linalg.qr(B)
                logdetB = np.sum(np.log(np.abs(np.diag(r))))
                res = res+d.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d)))
            elif method=='CH':
                lc = scipy.linalg.cho_factor(B)
                logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
                invBd=scipy.linalg.cho_solve(lc, d)
                res = res+d.T.dot(invBd)

                phi3_g=msp_info[key]['Phi3_GR']/ws
                phi3_gs=msp_info[key]['Phi3_GR_SN']/ws
 
                phi3_gr_r=phi3_g-phi3_gs.T.dot(scipy.linalg.cho_solve(lc, phi3_gs))
                d_gr_r=msp_info[key]['d_GR']/ws-phi3_gs.T.dot(invBd)

            else:
                invB=np.linalg.inv(B)
                logdetB=np.linalg.slogdet(B)[1]
                res=res+d.T.dot(invB).dot(d)
        except np.linalg.LinAlgError:
            print("LinAlgError", vpar)
            return -np.inf
        ldet=ldet+logdetB+logdetK

        K=np.append(K,K_gr)
        K=np.append(K,K_gr2)

        vphi3.append(phi3_gr_r) 
        d_gr=np.append(d_gr,d_gr_r)
    phi3=scipy.linalg.block_diag(*vphi3)

    mxH1=HD_func(mxtheta)

    if corr==0:
        mxH2=mxtheta*0+1
        mxH2+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==1:
        mxH2=np.cos(mxtheta)
        mxH2+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==2:
        mxH2=HD_func(mxtheta)

    invK=np.diag(1./K)
    K=np.diag(K)
    invH1=np.linalg.inv(mxH1)
    t1=time.time()
    for i in range(0,len(vec_keys)):
        for j in range(0,len(vec_keys)):
            indxi=np.arange(2*i*Nf*2,(2*i+1)*Nf*2)
            indxj=np.arange(2*j*Nf*2,(2*j+1)*Nf*2)
            indx_ij=np.ix_(indxi,indxj)
            K[indx_ij]= np.diag(K_gr)*mxH1[i,j]
            invK[indx_ij]= np.diag(1./K_gr)*invH1[i,j]
    t2=time.time()

    invH2=np.linalg.inv(mxH2) 
    for i in range(0,len(vec_keys)):
        for j in range(0,len(vec_keys)):
            indxi=np.arange((2*i+1)*Nf*2,(2*i+2)*Nf*2)
            indxj=np.arange((2*j+1)*Nf*2,(2*j+2)*Nf*2)
            indx_ij=np.ix_(indxi,indxj)
            K[indx_ij]= np.diag(K_gr2)*mxH2[i,j]
            invK[indx_ij]= np.diag(1./K_gr2)*invH2[i,j]
    logdetK=np.linalg.slogdet(K)[1]


    B=phi3+invK

    try:
        if method=='QR':
            q, r = np.linalg.qr(B)
            logdetB = np.sum(np.log(np.abs(np.diag(r))))
            res = res+d_gr.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d_gr)))
        elif method=='CH':
            lc = scipy.linalg.cho_factor(B)
            logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
            res = res+d_gr.T.dot(scipy.linalg.cho_solve(lc, d_gr))
        else:
            invB=np.linalg.inv(B)
            logdetB=np.linalg.slogdet(B)[1]
            res=res+d_gr.T.dot(invB).dot(d_gr)
    except np.linalg.LinAlgError:
        print("LinAlgError", vpar)
        return -np.inf

    if res>=0:
        print("non neg",vpar,res)
        return -np.inf
    ldet=ldet+logdetB+logdetK

    res=(res-ldet)*0.5

    return res



### to be changed!
def likli_curn_gwb_sn_new(vpar, fbm,  user_in, msp_info, vec_keys, vt_all, mxtheta, index_margin=[], corr=2, SN_key={}, fixgamma=0, Nf=2^5, fmin=1/20., fmax=365.25*0.5, method='CH',spec='PL',spacing='linear'):

    if spec=="PL":
        if fixgamma:
            npar_gw=1
            alpha_gr=13/3.
        else:
            npar_gw=2
            alpha_gr=vpar[1]
        alpha_gr=(1-alpha_gr)/2.
        amp_gr=np.power(10., vpar[0])*(12*np.pi**2)**-0.5
        K_gr=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)
    else:
         sys.exit('Model comparison mode currently available only for spec PL (change flag and try again.)')    
####################################   

    # gwb
    alpha_gr=vpar[npar_gw+1]
    alpha_gr=(1-alpha_gr)/2.
    amp_gr=np.power(10., vpar[npar_gw])*(12*np.pi**2)**-0.5
    K_gr2=Fortytwo.fortytwo_util_noise.PowerLaw_Covgen_K(vt_all, nf=Nf, fmin=fmin, fmax=fmax, fc=1., alpha=alpha_gr, ac=amp_gr,spacing=spacing)
    npar_gw+=2

    ## mainly to get K for all single psr noise, also update res,ldet for deterministic signal (DM exp dip)
    # res,ldet,vws=Fortytwo.fortytwo_util_noise.UpdateK_SN(vpar, msp_info, vec_keys, npar_gw, SN_key)
    res,ldet,vws=Fortytwo.fortytwo_util_noise.UpdateK_SN(vpar, user_in, fbm, msp_info, vec_keys, npar_gw, SN_key,spacing=spacing,fitsin=0)

    vphi3=[]
    d_gr=[]
    K=[]
    indx=[]
    for key in vec_keys:
        ws=vws[key]
        K_SN=msp_info[key]['K_SN']
        K_SN=np.append(K_SN,K_gr)
        phi3=msp_info[key]['Phi3_SN']/ws
        d=msp_info[key]['d_SN']/ws
        logdetK=np.sum(np.log(K_SN))

        # print('shapes curn+gwb: phi3,K_SN =',key,':',phi3.shape,K_SN.shape)

        B=phi3+np.diag(1./K_SN)

        try:
            if method=='QR':
                q, r = np.linalg.qr(B)
                logdetB = np.sum(np.log(np.abs(np.diag(r))))
                res = res+d.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d)))
            elif method=='CH':
                lc = scipy.linalg.cho_factor(B)
                logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
                invBd=scipy.linalg.cho_solve(lc, d)
                res = res+d.T.dot(invBd)

                phi3_g=msp_info[key]['Phi3_GR']/ws
                phi3_gs=msp_info[key]['Phi3_GR_SN']/ws
                phi3_gr_r=phi3_g-phi3_gs.T.dot(scipy.linalg.cho_solve(lc, phi3_gs))
                d_gr_r=msp_info[key]['d_GR']/ws-phi3_gs.T.dot(invBd)
                
                """
                invC=msp_info[key]['cov_winv']-msp_info[key]['Phi2_SN'].T.dot(scipy.linalg.cho_solve(lc,msp_info[key]['Phi2_SN']))
                phi_gr=msp_info[key]['Phi_GR']
                phi2_gr=phi_gr.T.dot(invC)
                phi3_gr_r=phi2_gr.dot(phi_gr)
                d_gr_r=phi2_gr.dot(msp_info[key]['vx2'])
                """
            else:
                invB=np.linalg.inv(B)
                logdetB=np.linalg.slogdet(B)[1]
                res=res+d.T.dot(invB).dot(d)
        except np.linalg.LinAlgError:
            print("LinAlgError", vpar)
            return -np.inf
        ldet=ldet+logdetB+logdetK

        K=np.append(K,K_gr2)

        vphi3.append(phi3_gr_r)
        d_gr=np.append(d_gr,d_gr_r)

    phi3=scipy.linalg.block_diag(*vphi3)
    # index for K_gr
    indx=np.array(indx,dtype=int)

    if corr==0:
        mxH=mxtheta*0+1
        mxH+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==1:
        mxH=np.cos(mxtheta)
        mxH+=np.diag(np.ones(len(vec_keys))*1e-5)
    elif corr==2:
        mxH=HD_func(mxtheta)
    elif corr==-1:
        y=vpar[-7:] #RNC 7 points: this number will be dynamic in next version
        mxH=np.zeros(mxtheta.shape)
        for i in range(0,len(vec_keys)):
            mxH[i,i]=1        
            for j in range(i,len(vec_keys)):
                mxH[i,j]=y[int(np.floor(mxtheta[i,j]/(np.pi/7)))]
                mxH[j,i]=mxH[i,j]

    n=K_gr.shape[0]
    invK=np.diag(1./K)
    K=np.diag(K)
    invH=np.linalg.inv(mxH)
    t1=time.time()
    for i in range(0,len(vec_keys)):
        for j in range(0,len(vec_keys)):
            indxi=np.arange(i*Nf*2,(i+1)*Nf*2)
            indxj=np.arange(j*Nf*2,(j+1)*Nf*2)
            indx_ij=np.ix_(indxi,indxj)
            K[indx_ij]= np.diag(K_gr)*mxH[i,j]
            invK[indx_ij]= np.diag(1./K_gr)*invH[i,j]
    logdetK=np.linalg.slogdet(K)[1]
    t2=time.time()

    B=phi3+invK

    try:
        if method=='QR':
            q, r = np.linalg.qr(B)
            logdetB = np.sum(np.log(np.abs(np.diag(r))))
            res = res+d_gr.T.dot(scipy.linalg.solve_triangular(r, q.T.dot(d_gr)))
        elif method=='CH':
            lc = scipy.linalg.cho_factor(B)
            logdetB = 2 * np.sum(np.log(np.abs(np.diag(lc[0]))))
            res = res+d_gr.T.dot(scipy.linalg.cho_solve(lc, d_gr))
        else:
            invB=np.linalg.inv(B)
            logdetB=np.linalg.slogdet(B)[1]
            res=res+d_gr.T.dot(invB).dot(d_gr)
    except np.linalg.LinAlgError:
        print("LinAlgError", vpar)
        return -np.inf

    if res>=0:
        print("non neg",vpar,res)
        return -np.inf
    ldet=ldet+logdetB+logdetK

    res=(res-ldet)*0.5
    
    return res



# We limit the loglike function to two options. CURN (uncorrelated) and GWB (correlated). For GWB, the correlation type is specified via the corr parameter 
def loglike(vpar, fbm, user_in, MSP_info, vkeys, vt_all, mxtheta, CS, corr, SN_key, gwb_lik, fixgamma=0, Nf=2^5, fmin=1/20., fmax=365.25*0.5, method='CH',fitsin=0,spec='PL',spacing='linear'):

    if CS=='CURN':
        res=likli_curn_sn(vpar,fbm,user_in,MSP_info,vkeys,vt_all,SN_key=SN_key,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax,method=method,spec=spec,spacing=spacing)
        
    elif CS=='GWB':
        #Select one: Both sn likelihoods work
        if gwb_lik==1:
            res=likli_gwb_sn_new(vpar,fbm,user_in,MSP_info,vkeys,vt_all,mxtheta,corr=corr,SN_key=SN_key,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax, method=method,spec=spec,spacing=spacing)
        elif gwb_lik==0:
            res=likli_gwb_sn(vpar,fbm,user_in,MSP_info,vkeys,vt_all,mxtheta,corr=corr,SN_key=SN_key,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax,method=method,spec=spec,spacing=spacing)
        else:
            sys.exit('Define GWB likelihood (flag gwb_lik) and try again')
        

    return res

def loglike_MC(vpar, fbm, user_in, msp_info_1, msp_info_2, vkeys, vt_all, mxtheta, index_margin, list_CS, corr=2, SN_key={}, gwb_lik=1, fixgamma=0, Nf=2^5, fmin=1/20., fmax=365.25*0.5, method='CH',fitsin=0,spec='PL',spacing='linear'):
    #note: index margin is not used at all in this Fortytwo version, but is in place for future version.
    nmodel=vpar[-1]
    nmodel=int(np.rint(nmodel))
    CS=list_CS[nmodel]
    if ('GWB2' in list_CS) or ('CURN+GWB' in list_CS):
        vpar_i=1 
    else:
        vpar_i=0 
    vpar1=np.append(vpar[0:2],vpar[2+vpar_i*2:-1])
    vpar2=vpar[:-1]
    if CS=='CURN':
        res=likli_curn_sn(vpar1,fbm,user_in,msp_info_1,vkeys,vt_all,SN_key=SN_key,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax,method=method,spec=spec,spacing=spacing)
        
    elif CS=='GWB':
        if 'GWB2' in list_CS:
            corr=2
        res=likli_gwb_sn_new(vpar1,fbm,user_in,msp_info_1,vkeys,vt_all,mxtheta,corr=corr,SN_key=SN_key,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax,method=method,spec=spec,spacing=spacing)
                            
        if corr<2:
            res+=10
    elif CS=='GWB2':
        res=likli_gwb2_sn_new(vpar2, fbm, user_in, msp_info_2, vkeys, vt_all, mxtheta, index_margin=[], corr=corr, SN_key=SN_key, fixgamma=fixgamma, Nf=Nf, fmin=fmin, fmax=fmax, method=method, spec=spec, spacing=spacing)
    elif CS=='CURN+GWB':
        res=likli_curn_gwb_sn_new(vpar2, fbm, user_in, msp_info_2,vkeys,vt_all,mxtheta,index_margin=[],corr=corr,SN_key=SN_key,fixgamma=fixgamma,Nf=Nf,fmin=fmin,fmax=fmax,method=method,spec=spec, spacing=spacing)
    else:
        print("nmodel out of range!")

    return res


