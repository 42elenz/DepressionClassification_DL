import torch
import matplotlib.pyplot as plt
def print_psd(raw_list):
    for raw in raw_list:
        try:
            spectrum = raw.compute_psd()
        except :
            print('ok')
        spectrum.plot(average=True, picks="data", exclude="bads")
        plt.show()

def transform_array(x):
	x = torch.tensor(x)
	min_val = x.min()
	max_val = x.max()
	x = 2 * ((x - min_val) / (max_val - min_val)) - 1
	return x, min_val, max_val

def transform_array_global(x, min_val, max_val):
	x = torch.tensor(x)
	x = 2 * ((x - min_val) / (max_val - min_val)) - 1
	return x

def reverse_transform_array(array, min_val, max_val, changed_shape=False):
	array = ((array + 1) / 2) * (max_val - min_val) + min_val
	if changed_shape==True:
		pure_length = array.shape[1]
		y =1024
		x = int(pure_length/y)
		array = array.reshape(x,y)
	return array

class DiffusionModel:
	def __init__(self, start_shedule=0.0001, end_schedule=0.02, timesteps=300):
		self.start_schedule = start_shedule
		self.end_schedule = end_schedule
		self.timesteps = timesteps
		
		self.betas = torch.linspace(self.start_schedule, self.end_schedule, self.timesteps)
		self.alphas = 1-self.betas
		self.alpha_cumprod = torch.cumprod(self.alphas,axis=0)

	def forward(self, x0, t, device, min_val, max_val):
		#noise= (min_val - max_val) * torch.rand_like(x0) + max_val * -1
		noise = torch.randn_like(x0)/2 #funktion mit std und min, max von meinen Daten
		sqrt_alphas_cumprod_t = self.get_index_from_list(self.alpha_cumprod.sqrt(), t, x0.shape)
		sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alpha_cumprod), t, x0.shape)
		mean = sqrt_alphas_cumprod_t.to(device) * x0.to(device)
		variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
		return mean + variance, noise.to(device)
	
	def backward(self, x, t, model, **kwargs):
		"""
		Calls the model to predict the noise in the image and returns 
		the denoised image. 
		Applies noise to this image, if we are not in the last step yet.
		"""
		betas_t = self.get_index_from_list(self.betas, t, x.shape)
		sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alpha_cumprod), t, x.shape)
		sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
		mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
		posterior_variance_t = betas_t

		if t == 0:
			return mean
		else:
			noise = torch.randn_like(x)/2
			variance = torch.sqrt(posterior_variance_t) * noise 
			return mean + variance

	@staticmethod
	def get_index_from_list(values, t, x_shape):
		batch_size = x_shape[0]
		result = values.gather(-1,t.cpu())

		return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
