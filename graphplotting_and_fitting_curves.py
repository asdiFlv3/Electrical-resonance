import numpy as np
pi=np.pi
C=4.55e-10 #capacitance
L=1.59e-2 #inductance
#%%load data
def dataload(filepath): #load data from csv
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, usecols=(0,1,2,3,4,5))
    #extract data from each column
    freq = data[:, 0]#in kHz,from column 1
    freq_error = data[:, 1]#col2
    pk2pk = data[:, 2]#in V, from column 3
    pk2pk_error = data[:, 3]#col4
    phase = data[:, 4]#col5
    phase_error = data[:, 5]#col6
    #%%calculations and convertions
    #calculate precentage errors
    f_precentage_error = freq_error/freq
    pk2pk_percentage_error = pk2pk_error/pk2pk
    phase_percentage_error = phase_error/phase
    #calculate angular frequency and angular phase
    angular_freq  =(freq*1000) * (2*pi)
    angular_phase = (phase/180)*pi
    #calculate amplitude
    amp=pk2pk/2
    #convert percentage into abs
    angular_phase_error = angular_phase * phase_percentage_error
    amp_error = pk2pk_percentage_error * amp
    angular_freq_error = angular_freq * f_precentage_error
    #%%split frequency for phase and amplitude and exclude NaN
    #indexing tutorial from https://www.runoob.com/numpy/numpy-advanced-indexing.html
    #Filter out the elements that are not NaN in both angular frequency and amplitude
    #so they can be fitted by scipy optimize
    #put them into adifferent arrays
    def filNaN(a, b , a_err, b_err):
        #If the nth element is not NaN in both array a and b, put the element into array a1
        #put the error of that element into array a1_error and b1_error
        mask_a_b = ~np.isnan(a) & ~np.isnan(b)
        a1 = a[mask_a_b]
        b1 = b[mask_a_b]
        a1_err = a_err[mask_a_b]
        b1_err = b_err[mask_a_b]
        return a1, b1, a1_err, b1_err

    angfreq_amp, filtamp, angfreq_amp_error, filtamp_error = filNaN(angular_freq, amp, angular_freq_error, amp_error)
    angfreq_pha, filtphase, angfreq_pha_error, filtphase_error = filNaN(angular_freq, angular_phase, angular_freq_error, angular_phase_error)
    
    return angfreq_amp, filtamp, angfreq_amp_error, filtamp_error, angfreq_pha, filtphase, angfreq_pha_error, filtphase_error

higherdamp=np.array(dataload('1500ohm.csv'), dtype=object)#dtype obeject for saving arrays not homogeneous
lowerdamp=np.array(dataload('500ohm.csv'), dtype=object)

#%%plotting&fitting for amplitude
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.misc import derivative
colours=['#FFBE7A', '#8ECFC9', '#FA7F6F', '#82B0D2']#colour design
#fitting
#for angular frequency against voltage formula is provided on labscript
#x-angular frequency, x0-frequency at peak, E0-max amplitude, r-resistance
def voltage(x, E0, x0, r):
    return E0/(C*np.sqrt(x**2 * r**2 + L**2 * (x**2 - x0**2)**2))
#provide the derivative of this curve for finding q factor
def voltage_derivative(x, E0, x0, r):
    return derivative(lambda x: voltage(x, E0, x0, r), x, dx=1e-4)

def plot_amplitude(data, R, guess, label, colour):
    #Plot amplitude with error bars and fitted curve.
    #data-dataset we want to use, R-resistance, guess- guess max amplitude E0, resonance f x0
    par, cvm=curve_fit(voltage, data[0], data[1], p0=guess)
    #par[0]-amplitude, par[1]- frequency, par[3]-R
    fit_values=voltage(data[0], *par)        
    #extract uncertainties(std deviation) by taking sqrt of the diagnonals in the covariance matrix(variance)
    E0_error, x0_error, r_error = np.sqrt(np.diag(cvm))
    #calculate Q factors
    #use derivative to find peak coordinate
    peak_guess=par[1]#initial guess
    #find the x where derivative is zero ==> maximum point
    x_peak=fsolve(lambda omega: voltage_derivative(omega, *par), peak_guess)[0]
    max_amp=voltage(x_peak, *par)
    half_height= max_amp/np.sqrt(2)
    #find frequencies that is not larger than x that produces half height
    #x_L/R: x on the left/right
    x_L=data[0][(fit_values<=half_height) & (data[0]<x_peak)]
    x_R=data[0][(fit_values<=half_height) & (data[0]>x_peak)]
    delta_x=x_R[0]-x_L[-1]#x_R[0] nearest to x half height(which is smallest) on the right, vice versa
    Q_theo=(par[1]*L)/R
    Q_factor = x_peak/delta_x
    #calculate the uncertainty of Q
    half_height_error=half_height*np.sqrt((x0_error/x_R[0])**2 + (x0_error/x_L[-1])**2)
    #propagate uncertainty to Q
    Q_factor_error=Q_factor * np.sqrt((x0_error/x_peak)**2 + (half_height_error/half_height)**2)
    print(f'Q factor for {label} is theoretically:{Q_theo}, experimentally {Q_factor}± {Q_factor_error}')
    print(f'Resonance frequency at R = {R}: {par[1]} ± {x0_error} rad/s, Max amplitude: {max_amp}± {E0_error} V')

    plt.errorbar(data[0], data[1], xerr=data[2], yerr=data[3],
                 fmt=',',color=colour[0], capsize=2, elinewidth=1, label=f'{label} Data')
    plt.plot(data[0], fit_values,color=colour[1], label=f'{label} Fit')

#plot
import matplotlib.ticker as ticker
plt.figure(figsize=(10, 6))
plot_amplitude(higherdamp, 1520, [0.9, 350000, 1520], 'Higher Damping', [colours[1],colours[0]])
plot_amplitude(lowerdamp, 502, [1.9, 355000, 502], 'Lower Damping', [colours[3],colours[2]])
plt.xlabel(r'Angular Frequency $\omega$ (rad/s)')
plt.ylabel('Amplitude (V)')
plt.title('Amplitude vs. Angular Frequency')
plt.legend()
#build grids in background to make the data more readable 
# Major ticks
plt.grid(True, which="major", linestyle="--", linewidth=0.8, color="#E7DAD2")
# Minor ticks
plt.grid(True, which="minor", linestyle="--", linewidth=0.5, color="#999999")
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(10000)) # Adjust interval for x minor grid
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1)) # Adjust interval for y minor grid
plt.show()

#%%plot phase
def phase(omega, r, l, c): #model for phase difference
    return np.arctan((omega*l - 1 / (omega*c))/r) + pi/2

#define a function to find intersection at pi/2
def g(omega, r, l, c):
    return phase(omega, r, l, c) - np.pi / 2

def plot_phase(x, R, colour, label):
    #Plot phase difference with functions, x: data araay used
    plt.errorbar(x[4], x[5], xerr=x[6], yerr=x[7],
                 fmt=',', ecolor=colour[0], capsize=2, elinewidth=1, label=f'{label} Experiment Data') #scatter points with error bars
    par, cvm = curve_fit(phase, x[4], x[5], p0=[R, L, C])
    fit_values = phase(x[4], *par)
    #find intersection at pi/2
    x_intersection = fsolve(g, 364699, args=(par[0], par[1], par[2]))
    print(f'the intersection of the fitted curve with 1/2 pi is {x_intersection}')
    plt.plot(x[4], fit_values, color=colour[1], label=f'{label} Fit')
    

#Plot
plt.figure(figsize=(10, 6))
plot_phase(higherdamp, 1520, [colours[1],colours[0]], 'Higher Damping Phase')
plot_phase(lowerdamp, 502, [colours[3],colours[2]], 'Lower Damping Phase')
#generate a line at 1/2 pi for reference
plt.axhline(y=pi/2, color='#999999', linestyle='--', label=r'$\frac{\pi}{2}$')
#label
plt.xlabel(r'Angular Frequency $\omega$ (rad/s)')
plt.ylabel(r'Phase Difference $\phi$ (rad)')
plt.title('Phase Difference vs. Angular Frequency')
plt.legend()
plt.show()