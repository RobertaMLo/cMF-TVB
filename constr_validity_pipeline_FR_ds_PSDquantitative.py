import numpy as np 
from scipy.signal import welch 
import pandas as pd 
import matplotlib.pyplot as plt 
import os


def load_crbl_cortex(input_dir, subject_ids, remove_dcns_bool, verbose = True):
    
    """
    Function to load the cerebellar population-specific firing rates simulated using TVB.
    Input: 
        input_dir           (string) = directory where I stored the firing rate for each population for each subjects
        subject_ids         (string) = IDs of the subjects
        remove_dcns_bool    (bool)   = True to remove dcn from the firing rates and region labels (True if hybrid TVB sim).
        verbose             (bool)   = Print info

    Output:
        <pop_name>_fr   (array) = firing rate for <pop> [nsubj, timepoints, regions]
        t_ds            (array) = time instants of the simulation [timepoints,]
    """
    
    grc_fr = []
    goc_fr = []
    mli_fr = []
    pc_fr = []

    t_ds = np.load(input_dir+'100307_CRBLMFds_500_time.npy', allow_pickle=True) #same downsample for all the subjects
    DCN_idx = [103, 104, 105, 113, 114, 115]
    DCN_idx = np.array(DCN_idx) - 93
    # # LOOP BABYYYYY ===============================================================================================================================================

    for index, sub_id in enumerate(subject_ids):
    
        # # SIMULATED BOLDs ==========================================================================================================================================
        print('================ SUBJECT ID ===============: ', sub_id)

        grc, goc, mli, pc = np.load(input_dir+sub_id+'_CRBLMFds_500.npy', allow_pickle = True)
        

        if remove_dcns_bool:
            # if evaluating hybrid TVB simulation set this at TRUE because in the .npy file of FR also DCN are stored
            grc = remove_dcns(grc, DCN_idx)
            goc = remove_dcns(goc, DCN_idx)
            mli = remove_dcns(mli, DCN_idx)
            pc = remove_dcns(pc, DCN_idx)

        grc_fr.append(grc)
        goc_fr.append(goc)
        mli_fr.append(mli)
        pc_fr.append(pc)


    grc_fr = np.array(grc_fr)
    goc_fr = np.array(goc_fr)
    mli_fr = np.array(mli_fr)
    pc_fr = np.array(pc_fr)

    if verbose:
        print('********************* Check activity loaded (shape and overall max): ********************* ')
        print('GrC: ',np.shape(grc_fr), np.max(grc_fr))
        print('GoC: ', np.shape(goc_fr), np.max(goc_fr))
        print('MLI: ', np.shape(mli_fr), np.max(mli_fr))
        print('PC : ', np.shape(pc_fr), np.max(pc_fr))
        
    return grc_fr, goc_fr, mli_fr, pc_fr, t_ds


def remove_dcns(mysig, DCN_idx):
   
    """
    Removing DCN data from mysig
    """
    
    # # Remove Deep Cerebellar Nuclei
    mask = np.ones(np.shape(mysig)[1], dtype=bool)
    mask[DCN_idx] = False
    mysig = mysig[:,mask]
    return mysig


def reading_regions(labels_path):
    """
    Function to load the region lables
    Input:
            labels_path (string) = fullpath + name of the labels txt file
    Output:
            regions_name (array) =  regions name
    """
    regions_labels = labels_path
    # # Reading the name of the regions used here ---------------------------------------------------------------------------------------------------------------------
    names = []
    with open(regions_labels, "r") as file:
        for line in file:
            parts = line.split('\t')
            name = parts[0].strip()
            names.append(name)
            
    regions_name = np.array(names)

    return regions_name 


def compute_avg(grc_fr, goc_fr, mli_fr, pc_fr, axis_idx, verbose = True):
    
    """
    Function compute the average on the regions. 
    Choose this kind of average if the goal is to study the overall PSD of the cerebellim.
    Input: 
        <pop>_fr              (narray) = [nsubjects x ntimepoints x regions] as loaded with loading function
        verbose               (bool)   = Print info

    Output:
        <pop_name>_fr_ar     (array) = avg firing rate for <pop> [nsubj, timepoints]
        <pop_name>_fr_ar_sd  (array) = sd firing rate for <pop> [nsubj, timepoints]
        
    """

    
    grc_fr_ar = np.mean(grc_fr, axis=axis_idx)
    goc_fr_ar = np.mean(goc_fr, axis=axis_idx)
    mli_fr_ar = np.mean(mli_fr, axis=axis_idx)
    pc_fr_ar = np.mean(pc_fr, axis=axis_idx)

    grc_fr_ar_sd = np.std(grc_fr, axis=axis_idx)
    goc_fr_ar_sd = np.std(goc_fr, axis=axis_idx)
    mli_fr_ar_sd = np.std(mli_fr, axis=axis_idx)
    pc_fr_ar_sd = np.std(pc_fr, axis=axis_idx)

    if verbose:
        if axis_idx == -1:
            print('****************** Average on regions  ******************')
            # # Average on regions: for each subject, one signal[n_subject x 500]
            print('Shape expected (8x500): ', np.shape(grc_fr_ar))
        
        if axis_idx == 0:
            print('****************** Average on subjects  ******************')
            # # Average on regions: for each subject, one signal[n_regiond x 500]
            print('Shape expected (27x500): ', np.shape(grc_fr_ar))
            # devo trasporre
            grc_fr_ar, goc_fr_ar, mli_fr_ar, pc_fr_ar =  grc_fr_ar.T, goc_fr_ar.T, mli_fr_ar.T, pc_fr_ar.T
            grc_fr_ar_sd, goc_fr_ar_sd, mli_fr_ar_sd, pc_fr_ar_sd = grc_fr_ar_sd.T, goc_fr_ar_sd.T, mli_fr_ar_sd.T, pc_fr_ar_sd.T

    return grc_fr_ar, goc_fr_ar, mli_fr_ar, pc_fr_ar, grc_fr_ar_sd, goc_fr_ar_sd, mli_fr_ar_sd, pc_fr_ar_sd


def compute_psd(signal, fs, nperseg, noverlap): 
    """
    Function compute the psd using welch method for FFT to reduce variance and improve Singal to Noise Ration.
    Signal is divided into segments with the same length using a windowing methods
    ensuring a certain overlap between each pair of asjacent segment.
    FFT is computed for each segment; PSD of each segment is averaged getting one PSD. 
    This reduce the variance of the PSD. 

    Input: 
        signal          (narray) = data on which the FFT is performed  [nvar x timepoints] 
        fr              (int)    = sampling frequency choosen according to Nyquist theorem
        nperseg         (int)    = length of the segment. Larger = ++freq_res --stability
        noverlap        (int)    = number of samples in each segment overlapping with previous segment.
                                    typical choice = 50% to 75 % of nperseg.
                                    ++ overla = --variance, ++ computational costs

    Output:
        <pop_name>_fr_ar     (array) = avg firing rate for <pop> [nsubj, timepoints]
        <pop_name>_fr_ar_sd  (array) = sd firing rate for <pop> [nsubj, timepoints]
        
    """
    
    freqs_all, psd_all = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    #normalizing psd and removing highest frequencies 
    #psd = psd_all[:int(len(psd_all)*0.7)]
    #freqs  = freqs_all[:int(len(psd_all)*0.7)] 

    psd  = psd_all
    freqs  =freqs_all
    psd = psd/np.max(psd) 
    
    return freqs, psd 


def compute_auc_and_significant_counts(freqs, psd, bands, threshold_ratio): 
    """
    Function to compute holisitc quantitative scores of PSD for each band:
    - AUC using trapz method
    - Frequency densitiy = #freq above a threshold

    Input:
        freqs           (array) = frequency output of welch method
        psd             (array) = psd output of welch method
        bands           (dict)  = <band_name>: (low, high) frequency bands of iterest
        threshold_ratio (float) = significant frequency (here 0.2)  
    """
    max_psd = np.max(psd) 
    threshold = max_psd * threshold_ratio 

    auc_counts = {band: {'auc': 0, 'significant_count': 0} for band in bands} 
    
    # doing the computation for each band separately
    for band, (low, high) in bands.items(): 
        
        # splitting the psd for each band
        band_mask = (freqs >= low) & (freqs <= high) 
        band_freqs = freqs[band_mask] 
        band_psd = psd[band_mask]

        #AUC
        auc = np.trapz(band_psd, band_freqs)
        
        # Frequency density
        significant_count = np.sum(band_psd > threshold) 
        
        auc_counts[band]['auc'] = auc 
        auc_counts[band]['significant_count'] = significant_count 
        
    return auc_counts 
    

def find_dominant_frequency(freqs, psd, bands): 
    """
    Function to compute the maximum frequency for each band.
    "Classic" apprach to get the band-specific carrier frequency
    """
    dominant_frequencies = {band: {'dominant_freq': 0, 'dominant_psd': 0} for band in bands} 

    for band, (low, high) in bands.items(): 
        
        band_mask = (freqs > low) & (freqs <= high) 
        band_freqs = freqs[band_mask] 
        band_psd = psd[band_mask] 
        
        dominant_freq = band_freqs[np.argmax(band_psd)] 
        dominant_psd = band_psd[np.argmax(band_psd)]
        
        dominant_frequencies[band]['dominant_freq'] = dominant_freq
        dominant_frequencies[band]['dominant_psd'] = dominant_psd
        
        #print(band)
        #print(dominant_frequencies[band]['dominant_freq'], dominant_frequencies[band]['dominant_psd'])
            
    return dominant_frequencies 
        
    
def find_dominant_band(freqs, psd, bands):

    """
    Function to find out the relevance of each band in the signal, i.e. Band Power.
    Band power is compured by summing the PSD for each band and the dominant band is returned.

    N.B.: This function is intended to be used in the spectral_mode_analysis function
    """

    #dictionary
    band_powers = {} 
    
    # for loop on each band
    for band, (low, high) in bands.items(): 
        #splitting psd in band
        band_mask = (freqs >= low) & (freqs <= high) 

        #band power
        band_power = np.sum(psd[band_mask]) 
        band_powers[band] = band_power

        #dominant_band of my signal
        dominant_band = max(band_powers, key=band_powers.get) 
            
    return dominant_band 
    

def spectral_mode_analysis(signals, fs, window_length, noverlap, bands): 
    """
    Analysis of the percentage of time each frequency band is dominant. 
    Reference: Capilla et al., 2022; Neuroimage


    """
    nsubjects, timepoints = signals.shape 
    dominant_band_percentages = {band: [] for band in bands} 
    
    #for each subject
    for subject in range(nsubjects): 
        band_count = {band: 0 for band in bands} 
        total_windows = 0 
        
        # Definition of a sliding window across the signal
        for start in range(0, timepoints - window_length + 1, window_length - noverlap): 
            end = start + window_length
            # computing welch transform for each window 
            freqs, psd = welch(signals[subject, start:end], fs=fs, nperseg=window_length, noverlap=noverlap) 
            dominant_band = find_dominant_band(freqs, psd, bands) 
            band_count[dominant_band] += 1 
            total_windows += 1 
            
        # computing the percentage for each band
        for band in bands: 
            percentage = (band_count[band] / total_windows) * 100 
            dominant_band_percentages[band].append(percentage)

    #converting the dictionary into array for easier handling
    dominant_band_percentages = {band: np.array(percentages) for band, percentages in dominant_band_percentages.items()} 
                
    return dominant_band_percentages 


def analyze_population(population_name, signals, fs, window_length, noverlap, bands, threshold_ratio, outdir, sub_id): 
    
    """
    Routine to compute:
    - AUC and frequency density
    - dominant_frequency 
    - spectral_modes

    for each subject
    """
    print(f"\nAnalyzing population: {population_name}") 
    
    nsubjects, timepoints = signals.shape
    auc_counts_all_subjects = [] 
    dominant_frequencies_all_subjects = [] 


    spectral_modes = spectral_mode_analysis(signals, fs, window_length, noverlap, bands) 
    
    #For each subject
    for subject in range(nsubjects):
        
        print(sub_id[subject])

        freqs, psd = compute_psd(signals[subject], fs, window_length, noverlap) 
        
        auc_counts = compute_auc_and_significant_counts(freqs, psd, bands, threshold_ratio) 
        dominant_frequencies = find_dominant_frequency(freqs, psd, bands)
        
        auc_counts_all_subjects.append(auc_counts)
        dominant_frequencies_all_subjects.append(dominant_frequencies)
        
        plot_psd_with_significant_freqs(subject, population_name, freqs, psd, bands, threshold_ratio, outdir, sub_id[subject]) 
        
    return auc_counts_all_subjects, dominant_frequencies_all_subjects, spectral_modes 


def analyze_populations_regions(population_name, signals, fs, window_length, noverlap, bands, threshold_ratio, outdir, sub_id): 
    
    """
    Routine to compute:
    - AUC and frequency density
    - dominant_frequency 
    - spectral_modes

    for each regions
    """
    print(f"\nAnalyzing population: {population_name}") 
    
    nsubjects, timepoints = signals.shape
    auc_counts_all_subjects = [] 
    dominant_frequencies_all_subjects = [] 


    spectral_modes = spectral_mode_analysis(signals, fs, window_length, noverlap, bands) 
    
    #For each subject
    for subject in range(nsubjects):
        
        print(sub_id[subject])

        freqs, psd = compute_psd(signals[subject], fs, window_length, noverlap) 
        
        auc_counts = compute_auc_and_significant_counts(freqs, psd, bands, threshold_ratio) 
        dominant_frequencies = find_dominant_frequency(freqs, psd, bands)
        
        auc_counts_all_subjects.append(auc_counts)
        dominant_frequencies_all_subjects.append(dominant_frequencies)
        
        plot_psd_with_significant_freqs(subject, population_name, freqs, psd, bands, threshold_ratio, outdir, sub_id[subject]) 
        
    return auc_counts_all_subjects, dominant_frequencies_all_subjects, spectral_modes 
    

def plot_psd_with_significant_freqs(subject, population_name, freqs, psd, bands, threshold_ratio, outdir, sub_id, title_font = 14): 
    
    max_psd = np.max(psd) 
    threshold = max_psd * threshold_ratio 
    
    significant_freqs = freqs[psd > threshold]
    significant_psd = psd[psd > threshold]

    
    # Colors for each band
    colors_band = {
        'delta': 'red',
        'theta': 'green',
        'alpha': 'blue',
        'beta': 'purple',
        'gamma': 'orange'
    }
    
    
    plt.figure(figsize=(12, 8)) 
    #plt.stem(freqs, psd, basefmt=" ", use_line_collection=True)
    
    plt.semilogy(significant_freqs, significant_psd) 
    #plt.stem(significant_freqs, significant_psd, linefmt='k-', markerfmt='ko', basefmt=" ", use_line_collection=True)
    plt.xlabel('Frequency (Hz)', fontsize = title_font-2) 
    plt.ylabel('PSD', fontsize = title_font-2) 
    plt.title('Subject '+sub_id +f'- {population_name}', fontsize = title_font) 
    
    for band, (low, high) in bands.items(): 
        plt.axvspan(low, high, color=colors_band[band], alpha=0.2, label=band) 
    
    plt.legend() 
    plt.savefig(outdir+'/psd_subject_'+sub_id+population_name+'.png', dpi=300) 
    plt.close() 
        


def save_results_to_csv(population_name, auc_counts, dominant_frequencies, spectral_modes, outdir, sub_id): 
    
    auc_df = pd.DataFrame(columns=['Subject', 'Band', 'AUC', 'Significant Count']) 
    
    for subject_idx, auc_data in enumerate(auc_counts): 
        
        for band, values in auc_data.items():
            
            auc_df = auc_df.append(
                { 'Subject': sub_id[subject_idx], 
                 'Band': band, 
                 'AUC': values['auc'], 
                 'Significant Count': values['significant_count'] }, 
                 ignore_index=True) 
            
    dominant_freq_df = pd.DataFrame(columns=['Subject', 'Band', 'Dominant frequency', 'Dominant psd'])        
    
    for subject_idx, dominant_freq_data in enumerate(dominant_frequencies): 
        
        for band, values in dominant_freq_data.items(): 
            
            dominant_freq_df = dominant_freq_df.append(
                {'Subject': sub_id[subject_idx], 
                 'Band': band,
                 'Dominant Frequency': values['dominant_freq'],
                 'Dominant PSD': values['dominant_psd'] }, 
                ignore_index=True) 
                    
    spectral_mode_df = pd.DataFrame(columns=['Subject', 'Band', 'Percentage']) 
    for band, percentages in spectral_modes.items(): 
        for subject_idx, percentage in enumerate(percentages): 
            spectral_mode_df = spectral_mode_df.append(
                { 'Subject': sub_id[subject_idx], 
                 'Band': band, 
                 'Percentage': percentage }, 
                 ignore_index=True)
    
    auc_df.to_csv(outdir+'/'+population_name+'_auc_counts.csv', index=False) 
    dominant_freq_df.to_csv(outdir+'/'+population_name+'_dominant_frequencies.csv', index=False) 
    spectral_mode_df.to_csv(outdir+'/'+population_name+'_spectral_modes.csv', index=False) 
                            

def analyze_populations_with_averaged_psd(population_name, signals, fs, window_length, noverlap, bands, threshold_ratio, outdir, sub_id): 
    """
    Routine to compute:
    - AUC and frequency density
    - dominant_frequency 
    - spectral_modes

    for each region in a given population.
    """
    print(f"\nAnalyzing population: {population_name}") 
    
    nsubjects, timepoints = signals.shape
    auc_counts_all_subjects = [] 
    dominant_frequencies_all_subjects = [] 
    psds_all_subjects = []

    spectral_modes = spectral_mode_analysis(signals, fs, window_length, noverlap, bands) 
    
    # For each subject
    for subject in range(nsubjects):
        
        #print(sub_id[subject])

        freqs, psd = compute_psd(signals[subject], fs, window_length, noverlap) 
        
        auc_counts = compute_auc_and_significant_counts(freqs, psd, bands, threshold_ratio) 
        dominant_frequencies = find_dominant_frequency(freqs, psd, bands)
        
        auc_counts_all_subjects.append(auc_counts)
        dominant_frequencies_all_subjects.append(dominant_frequencies)
        psds_all_subjects.append(psd)
        
    # Compute averaged PSD
    averaged_psd = np.mean(psds_all_subjects, axis=0)
    plot_averaged_psd_for_population(population_name, freqs, averaged_psd, bands, threshold_ratio, outdir)
    
    return auc_counts_all_subjects, dominant_frequencies_all_subjects, spectral_modes


def plot_averaged_psd_for_population(population_name, freqs, averaged_psd, bands, threshold_ratio, outdir, title_font=14): 
    """
    Plot the averaged PSD for a given population, highlighting significant frequencies and frequency bands.
    """
    max_psd = np.max(averaged_psd) 
    threshold = max_psd * threshold_ratio 
    
    significant_freqs = freqs[averaged_psd > threshold]
    significant_psd = averaged_psd[averaged_psd > threshold]
    
    # Colors for each band
    colors_band = {
        'delta': 'red',
        'theta': 'green',
        'alpha': 'blue',
        'beta': 'purple',
        'gamma': 'orange'
    }
    
    plt.figure(figsize=(12, 8)) 
    plt.stem(significant_freqs, significant_psd, linefmt='k-', markerfmt='ko', basefmt=" ", use_line_collection=True) 
    
    plt.xlabel('Frequency (Hz)', fontsize=title_font-2) 
    plt.ylabel('PSD', fontsize=title_font-2) 
    plt.title(f'Averaged PSD - {population_name}', fontsize=title_font) 
    
    for band, (low, high) in bands.items(): 
        plt.axvspan(low, high, color=colors_band[band], alpha=0.2, label=band) 
    
    plt.legend() 
    plt.savefig(outdir + f'/averaged_psd_{population_name}.png', dpi=300) 
    plt.close()



def main(): 
    
    input_dir = '/media/bcc/Volume/Analysis/Roberta/DOWNSAMPLE_4_TVB_WB/' #../DOWNSAMPLE_4_TVB_WB/'
    outdir = '/home/bcc/regions_analysis_new/'
    
    remove_dcns_bool = True #Make much more sense doing this analysis for whole brain
    
    # sampling frequency and window parameters 
    fs = 250 #100 

    # parameters for welch FFT    
    window_length = 500 # Example window length in samples 
    
    noverlap = window_length // 2 # 50% overlap 
    
    # threshold for significant frequencies 
    threshold_ratio = 0.2 # 20% of max

    #decide if I want to go with avg on regions or on subjects
    # 0 = avg on subjecsts --> signal = nregions x time
    # -1 = avg on regions --> signal = nsubjects x time
    axis_idx = 0 
    

    print('Sampling freqiuency: ', fs, '\nnperseg: ',window_length, '\noverlap: ',noverlap)

    # Loading data
    subject_ids = ['100307', '101915', '103818', '106016', '108828', '110411', '111312', '111716']
    
    
    grc_fr, goc_fr, mli_fr, pc_fr, t_ds = load_crbl_cortex(input_dir, subject_ids, remove_dcns_bool)

    
    #Averaging on regions (one big cerebellar regions for all the subjects)
    grc_fr_ar, goc_fr_ar, mli_fr_ar, pc_fr_ar, grc_fr_ar_sd, goc_fr_ar_sd, mli_fr_ar_sd, pc_fr_ar_sd = \
    compute_avg(grc_fr, goc_fr, mli_fr, pc_fr, axis_idx, verbose = True)


    #Mega average on subject and regions
    #grc_fr_ar, goc_fr_ar, mli_fr_ar, pc_fr_ar, grc_fr_ar_sd, goc_fr_ar_sd, mli_fr_ar_sd, pc_fr_ar_sd = \
    #    compute_avg_on_subjects(grc_fr, goc_fr, mli_fr, pc_fr, verbose = True)
    
    
    labels_path = '/home/bcc/HCP_TVBmm_30M/106016/T1w/SC_dirCB_ONLYCRBL/centres.txt'
    regions_name = reading_regions(labels_path)
    
    
    # Bands of interest
    bands = { 
    'delta': (0.5, 4), 
    'theta': (4, 8), 
    'alpha': (8, 13), 
    'beta': (13, 30), 
    'gamma': (30, 100) 
    } 

    outdir = outdir+str(fs)+'nperseg_'+str(window_length)+'cut_'+str(threshold_ratio)
    os.makedirs(outdir, exist_ok=True) #checking or creating the output_dir

    populations = { 'grc': grc_fr_ar, 'goc': goc_fr_ar, 'mli': mli_fr_ar, 'pc': pc_fr_ar } 
    results = {}

    if axis_idx == -1:
        sub_id = subject_ids #in function analyse pop, the variable of the loop is called sub_id
    else:
        sub_id = regions_name

    for population_name, signals in populations.items(): 
        
        # Calling main routine - FOR EACH REGIONS - AVG subject
        #auc_counts, dominant_frequencies, spectral_modes = \
        #analyze_population( population_name, signals, fs, window_length, noverlap, bands, threshold_ratio, 
        #                   outdir, regions_name) 
        
        # Calling main routine - FOR EACH SUBJECT
        auc_counts, dominant_frequencies, spectral_modes = \
        analyze_population( population_name, signals, fs, window_length, noverlap, bands, threshold_ratio, 
                           outdir, sub_id) 

        #analyze_populations_with_averaged_psd(population_name, signals, fs, window_length, 
        #                                      noverlap, bands, threshold_ratio, outdir, subject_ids) 
    
        # for each population saving the results as a dictionary 
        results[population_name] = { 
            'auc_counts': auc_counts, 
            'dominant_frequencies': dominant_frequencies, 
            'spectral_modes': spectral_modes } 
        
        # Save results to CSV
        save_results_to_csv(population_name, auc_counts, dominant_frequencies, spectral_modes, outdir, regions_name)
        print("Analysis complete and results saved to CSV files at: ",outdir) 
   

if __name__ == "__main__": main()