import mne
import numpy as np
from mne_icalabel import label_components
from numpy.random import seed
seed(4)

def array_to_edf(data, ch_names, ch_types, fname,sfreq=256):
	""" 
	Converts a numpy array to an EDF file. 
	:param data: the numpy array containing the data 
	:param sfreq: the sampling frequency 
	:param ch_names: the channel names 
	:param ch_types: the channel types 
	:param fname: the file name of the resulting EDF file 
	:return: None 
	""" 
	info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types) 
	raw = mne.io.RawArray(data, info) 
	return(raw)

def ft_z_normalize(data): 
	""" Z-normalizes data :param data: the data to be z-normalized :return: the z-normalized data """ 
	mean_data = np.mean(data) 
	std_data = np.std(data) 
	data = (data - mean_data) / std_data 
	return data, mean_data, std_data

#make sliding window without mne
def sliding_window_eeg_manually(raw, win_length_sec, overlap_perc=1.0, z_normalize=True, normalize=True):
	raw_array = raw.get_data()
	sampling_freq = raw.info['sfreq']
	win_length = int(win_length_sec * sampling_freq)
	overlap_samples = int(win_length * overlap_perc)  # Calculate overlap in samples
	step_size = win_length - overlap_samples
	n_channels = raw_array.shape[0]
	n_samples = raw_array.shape[1]
	n_windows = int((n_samples - win_length) / step_size) + 1  # Adjusting for overlap
	data_array = np.zeros((n_windows, n_channels, win_length))

	for i in range(n_windows):
		start = i * step_size
		end = start + win_length
		data_array[i, :, :] = raw_array[:, start:end]

	# Perform z-normalization
	if z_normalize:
		for i in range(n_windows):
			for j in range(n_channels):
				data_array[i, j, :], mean, std = ft_z_normalize(data_array[i, j, :])
		return data_array

	# Normalize the data between -1 and 1
	if normalize:
		for i in range(n_windows):
			for j in range(n_channels):
				data_array[i, j, :] = data_array[i, j, :] / np.max(np.abs(data_array[i, j, :]))

	return data_array



def sliding_window_eeg(raw, win_length_sec, overlap_pct=0.75):
	"""
	Applies a sliding window to EEG data loaded with MNE-Python.
	Returns a numpy array of shape (n_epochs, n_channels, n_samples_per_epoch).
	
	Parameters:
	-----------
	raw : mne.io.Raw object
		The raw EEG data to be windowed.
	win_length_sec : float, optional (default=4)
		The duration of each window in seconds.
	overlap_pct : float, optional (default=0.75)
		The percentage of overlap between adjacent windows, expressed as a fraction between 0 and 1.
	
	Returns:
	--------
	data_array : numpy array of shape (n_epochs, n_channels, n_samples_per_epoch)
		The windowed EEG data.
	"""
	overlap = win_length_sec * overlap_pct
	events = mne.make_fixed_length_events(raw, id=1, duration=win_length_sec, overlap=overlap)
	epochs = mne.Epochs(raw, events, tmin=0, tmax=win_length_sec, baseline=None)
	data_array = epochs.get_data()
	return data_array[:,:,1:]

def filter_raw(raw, l_filter=1, h_filter=100, notch=50, reference=None):
	"""
	Applies a bandpass filter to raw EEG data loaded with MNE-Python.
	Returns a filtered mne.io.Raw object.
	
	Parameters:
	-----------
	raw : mne.io.Raw object
		The raw EEG data to be filtered.
	l_filter : float, optional (default=0.1)
		The lower bound of the passband, in Hz.
	h_filter : float, optional (default=100)
		The upper bound of the passband, in Hz.
	notch : float, optional (default=50)
		The frequency to notch filter out, in Hz.
	
	Returns:
	--------
	raw : mne.io.Raw object
		The filtered EEG data.
	"""
	raw = raw.notch_filter(notch)
	raw = raw.filter(l_filter, h_filter, fir_design="firwin")
	if reference is not None:
		if reference == "average":
			raw = raw.set_eeg_reference(ref_channels='average', projection=True)
			raw.apply_proj()
		else:
			raw = raw.set_eeg_reference(ref_channels=[reference])
	return raw

def make_montage(raw, montage = "standard_1020"):
	"""
	Applies a montage to raw EEG data loaded with MNE-Python.
	Returns a mne.io.Raw object with the montage applied.
	
	Parameters:
	-----------
	raw : mne.io.Raw objecty
		The raw EEG data to be filtered.
	montage : str
		The name of the montage to be applied.
	
	Returns:
	--------
	raw : mne.io.Raw object
		The EEG data with the montage applied.
	"""
	montage = mne.channels.make_standard_montage(montage)
	raw = raw.set_montage(montage, on_missing="ignore")
	return raw

def ica_filter(raw, labels_to_keep=["brain", "other"], n_components=0.99, max_iter="auto", method="infomax", random_state=42, fit_params=dict(extended=True)):
	"""
	Applies an ICA filter to raw EEG data loaded with MNE-Python.
	Returns a mne.io.Raw object with the ICA filter applied.
	
	Parameters:
	-----------
	raw : mne.io.Raw object
		The raw EEG data to be filtered.
	labels_to_keep : list of str, optional (default=["brain", "other"])	
		The labels of components to keep. See mne_icalabel documentation for more information.
	n_components : int or float, optional (default=0.95)
		The number of components to keep. See mne.preprocessing.ICA documentation for more information.
	max_iter : int or str, optional (default="auto")
		The number of iterations to run the ICA algorithm. See mne.preprocessing.ICA documentation for more information.
	method : str, optional (default="infomax")
		The ICA algorithm to use. See mne.preprocessing.ICA documentation for more information.
	random_state : int, optional (default=42)
		The random seed to use. See mne.preprocessing.ICA documentation for more information.
	fit_params : dict, optional (default=dict(extended=True))
		The parameters to use when fitting the ICA algorithm. See mne.preprocessing.ICA documentation for more information.

	Returns:
	--------
	raw : mne.io.Raw object
		The EEG data with the ICA filter applied.
	"""
	ica_filt = mne.preprocessing.ICA(
	n_components=n_components,
	max_iter=max_iter,
	method=method,
	random_state=random_state,
	fit_params=fit_params,)
	ica_filt.fit(raw)
	ic_labels = label_components(raw, ica_filt, method="iclabel")
	labels = ic_labels["labels"]
	exclude_idx = [idx for idx, label in enumerate(labels) if label not in labels_to_keep]
	ica_filt.apply(raw, exclude=exclude_idx)
	return raw



