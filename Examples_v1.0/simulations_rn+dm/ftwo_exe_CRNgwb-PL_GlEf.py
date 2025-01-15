##########################################################################################
#######Example config file for single common signal analysis, using PTMCMC sampler########
##########################################################################################
import subprocess
from Fortytwo.fortytwo_util_main import exe42Bayes

##########################################################################################
##########################################################################################
##########################################################################################

#example: search for a GWB (common signal with Hellings-Down correlation), modeled as 2-parameter power-law, custom pulsar models, incl. global EFACs

######### Data and results - input-output directories/files #########

rootpath=subprocess.os.getcwd()+("/") # directory with pulsar data subdirectories. By default, each pulsar data are expected to be subdirectories of rootpath and named with the pulsar's J name, eg J0613-0200
psr_list="psrlist_6.txt" # filename of ascii file with list of pulsar names
ephext='DE440' # Solar-system ephemeris flag; points to correct data files

#example for PTMCMC output directory#
output_dir='CRN_6PSR_GWB_PL_psh0_fbm1_GlEf' # Set output directory name (rootpath subdirectory) for all results
overwrite_protection=1 # if 1, code is terminated if designated output directory already exists. optrions: 1=protection on, 0=protection off
keeplogfile=1 #1 keep a log file in output_dir. Will be named output_dir_log.txt

######### Main methods settings #########

sampler='PTMCMC' # (options are: (MN, PC, PTMCMC, FTMH) = (MultiNest, Polychord, PTMCMC, FTwo Metropolis-Hastings) #Note: only PTMCMC is available in this version
#Note: The default ptmcmc settings are (N=3e6, SCAMweight=30, AMweight=15, DEweight=50). For custom settings, enbale and adjast the line below 
# ptmcmc_settings=[2e6,40,20,70]# = [N, SCAMweight, AMweight, DEweight] - only if you want custom settings. Don't enable this unless you know what you're doing         
likelihood_test=True # Set to 'True' if you want to test the likelihood functions and how much time their calculations cost (approx.)
navr=1 # data point averaging; must be integer: 1=no averaging. any other integer n, means average per n data points. Leave navr=1 unless you really know what you're doing
method='CH' # matrix inversion method # 'CH' = Choleski, 'QR' = qr , 'NP' = standard numpy inversion 
noise_style=1 # math related to power-law noise parameters. Leave at 1 unless you really know what you're doing (other methods are currently disabled anyway)
SN=0 # special flag for custom system noise set-up. Leave at 0 unless you know what you're doing.
freq_bin_mode=1 # control number of frequency bins for stochastic signals. Important for common signals and pulsar noise. Options: 0=default choice for 15 frequency bins for all components (RN+DM noise and GW). 1=set freq. bin number for each type of component (read in with nf_### flags), but same for all pulsars.   2=custom per pulsar on its own freq. grid - requires ascii file with all freq. bin numbers (read in with opt_freq_list flag)
opt_freq_list='optimal_nf.txt' #file with optimal noise frequency bin numbers per pulsar - REQUIRED when  freq_bin_mode=2
spacing='linear' # distribution of frequency bin spaces:  'linear'=linear , 'log'= base 10 logarithmic 

######### Settings for common signal(s) #########

model_comparison=0 # 0 (or ommited) when searching for single signal/pair pf signals , 1 to perform model comparison. 
CS='GWB' # set common signal analysis type for model_comparison=0 # 'GWB' for any type of spatially correlated signal (GWB, monopolar, dipolar, unknown correlation - corr paramter controls the type) , 'CURN' for common-uncorrelated noise
corr=2 # correlation type for CS='GWB': 0= monopolar correlation, 1=dipolar correlation, 2=quadrupolar HD correlation, -1=search for ORF
nf=10 # number of CS frequency bins # Note: for freq_bin_mode=0, nf is fixed and preset at 10
gwb_lik=1 #which gwb likelihood to use - leave 1  unless you really know what you're doing
fixgamma=0 # fixed spectral index for CS or not. 0=fixed at nominal GWB index of 4/3, 1=gamma is free parameter
fitsin=0 #fit sine - leave at 0
spec='PL' # modelling of CS power spectrum # 'PL'=power law, 'brokenPL'=broken power law, 'freespec'= free spectrum
#if spec='brokenPL' we must declare the range of the turn-over frequency prior with CSfbL & CSfbU 
fit_phase_shifts=0 # used to apply random phase shifts on CS signals to break spatial correlations; usually used in model comparison. 0=do not apply phase shifts , 1=apply phase shifts

#Prios 

#if spec=PL or brokenPL (otherwise they are ignored)
CSampL,CSampU=-18.,-10. # Common signal amplitude (log10) priors (lower, upper)
CSgamL,CSgamU=0,7 # Common signal spectral index (gamma) prios (lower,upper)

#if spec=brokenPL (otherwise they are ignored)
CSfbL,CSfbU=-9.,-6.6  # priors for brokenPL turn-over frequency

#if spec=freespec (otherwise they are ignored)
FSampL,FSampU=-18.,-10.

######### Noise signals settings #########

fitDMe=0 #controls if "DM events will be modelled (1) or not (0)" - currently only for J1713+0747 w/ fixed priors
forceRN=0 #when 1, it forces RN fit with force_nf_rn IF (freq_bin_mode=2 AND nf_rn=0), where nf_rn=number of freq.bins for RN
force_nf_rn=5 # number of RN freq.bins for RN when forceRN=1
################################
fitWN=1 # 0=fixed white noise level (ML values), 1=fit global efac per pulsar
GEfL,GEfU=0.5,2 # (linear) Global EFAC value (lower,upper).
#Parameter fitting and priors
#amplitude prior values are in log10#

#################################################################################################
# For the rest below, note the following: 
#Note 1: fit flags for noise components and nf flags are relevant only for freq_bin_mode=1. 
#They are irrelevant for freq_bin_mode=2 as these are then controlled by the opt_freq_list file!
#Also for freq_bin_mode=0, all these flags are fixed and preset
#
#Note 2: PriorType options are not ready yet; currently, only log priors are possible
#################################################################################################

#Red Noise#
fitRN=1 # 0=do not include red noise in pulsar noise model, 1=include red noise in pulsar noise model
nf_r=15 #RN number of frequency bins (for freq_bin_mode=1)
RNgamPriorType=0 # RN spectral index prior type. options: 0=linearly uniform
RNampL,RNampU=-18,-10 #RN amplitude prior range in power of 10 (lower limit,upper limit)
RNgamL,RNgamU=0,7 #RN spectral index range (lower limit,upper limit)

#DM noise#
fitDM=1 # 0=do not include DM in pulsar noise model, 1=include DM in pulsar noise model
nf_dm=15 #DM number of frequency bins (for freq_bin_mode=1)
DMgamPriorType=0 # DM spectral index prior type. options: 0=linearly uniform
DMampL,DMampU=-18,-10 #DM amplitude prior range in power of 10 (lower limit,upper limit)
DMgamL,DMgamU=0,7 #DM spectral index range (lower limit,upper limit)

fitSv=0
nf_sv=15 # Sv number of frequency bins (for freq_bin_mode=1 AFTER next code upgrade)
SVgamPriorType=0 # SV spectral index prior type. options: 0=linearly uniform
SVampL,SVampU=-18,-10 #SV amplitude prior range in power of 10 (lower limit,upper limit)
SVgamL,SVgamU=0,7 #SV spectral index range (lower limit,upper limit)

#Note: system- and band-noise options will be enabled in the next version

#system noise#
##fitSys=0
# SYSgamPriorType=0 # SYS spectral index prior type. options: 0=linearly uniform
# SYSampL,SYSampU=-18,-10 #SYS amplitude prior range in power of 10 (lower limit,upper limit)
# SYSgamL,SYSgamU=0,7 #SYS spectral index range (lower limit,upper limit)

#band noise#


#run the analysis code!!
uvars=dict(locals()); exe42Bayes(uvars) #runs the code 
