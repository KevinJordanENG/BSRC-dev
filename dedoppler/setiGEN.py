# Generate standard setigen frame for testing of pipelines
# turboSETI, SETIcore, & hyperSETI. -kevin.jordan.ee@gmail.com

from astropy import units as u
import matplotlib.pyplot as plt
import setigen as stg

filename = 'setigen_testfile.fil'

frame = stg.Frame(fchans=65536*u.pixel,
                  tchans=16*u.pixel,
                  df=2.7939677238464355*u.Hz,
                  dt=18.253611008*u.s,
                  fch1=6095.2148*u.MHz)
noise = frame.add_noise(x_mean=10, noise_type='chi2')
signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=8000),
                                            drift_rate=4*u.Hz/u.s),
                          stg.constant_t_profile(level=frame.get_intensity(snr=50)),
                          stg.gaussian_f_profile(width=400*u.Hz),
                          stg.constant_bp_profile(level=1))
frame.data = frame.data.astype('float32')
#fig = plt.figure(figsize=(10,6))
#frame.bl_plot()
#plt.show()

frame.save_fil(filename=filename)