import os
import csv
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_signal(length_seconds, sampling_rate, frequencies_list, func=[], trend=[0,1], alea=False, bornes=[-10,10], add_noise=0, plot=False):
    """
    Generate a `length_seconds` seconds signal at `sampling_rate` sampling rate. See torchsignal (https://github.com/jinglescode/torchsignal) for more info.
    Args:
        length_seconds : int
            Duration of signal in seconds (i.e. `10` for a 10-seconds signal)
        sampling_rate : int
            The sampling rate of the signal.
        frequencies_list : 1 dimension python list a floats
            An array of floats, where each float is the desired frequencies to generate (i.e. [5, 12, 15] to generate a signal containing a 5-Hz, 12-Hz and 15-Hz)
        alea : boolean, default : false
            When alea==true, an and bn are randomly picked
        func : list, default: []
            Contains the periodic functions to generate signal, either `sin` or `cos`
            Must match frequencies_list size. If not, will be automatically completed by "sin"
        bornes: list, 
            Specifies the max and min values for the rand function that will then pick the value of the "an"/"bn" coefficient of the Fourier series
        trend : list, default : [0,1]>
            Coefficient of a polynomial functions that will give a generall trend to the signal
        add_noise : float, default: 0
            Add random noise to the signal, where `0` has no noise
        plot : boolean
            Plot the generated signal
    Returns:
        signal : 1d ndarray
            Generated signal, a numpy array of length `sampling_rate*length_seconds`
    """
    
    
    frequencies_list = np.array(frequencies_list, dtype=object)
    nf=frequencies_list.shape[0]
    assert len(func)<=nf, "unuseful functions have been precised"
    while nf>len(func) :
        func.append('sin')
        
    npnts = sampling_rate*length_seconds  # number of time samples 
    
    
    time = np.arange(0, npnts)/sampling_rate
    signal = np.zeros(npnts)
    
    
    assert bornes[0]<bornes[1], "on doit pouvoir extraire entre les bornes fournises"
    
    if alea==True : 
        for k in range(nf) :
            rd.seed=1024
            mu=rd.uniform(bornes[0],bornes[1])
            sigma=(rd.uniform(bornes[0],bornes[1]))**2
            theta=rd.gauss(mu,sigma)**2 #i wanted a positiv amplitude
        
            if func[k] == "cos":
                signal = signal + theta*np.cos(2*np.pi*frequencies_list[k]*time)
            else:
                signal = signal + theta*np.sin(2*np.pi*frequencies_list[k]*time)
    else : 
        for k in range(nf) :
            if func[k] == "cos":
                signal = signal + np.cos(2*np.pi*frequencies_list[k]*time)
            else:
                signal = signal + np.sin(2*np.pi*frequencies_list[k]*time)
                   
    if add_noise:
        print(signal.shape)
        noise = np.random.uniform(low=0, high=add_noise, size=(npnts,))
        print(noise.shape)
        signal = signal + noise

    if plot:
        xtrain = np.zeros(200)
        k=rd.randint(0,signal.size-200)
        xtrain=signal[k:k+200]
        plt.plot(xtrain)  
        plt.show()
        
    return signal,time,length_seconds

def perturbation(signal,time,length_seconds):
    '''creates (and shows) a linear piecewise perturbation that matchs with the signal
    Args : 
        signal ; array, the signal you want to disturb
        time : an array, the temporal values that correspond to the numpy signal
        length_seconds : int, duration of signal'''
    rd.seed=1024
    u=rd.uniform(0,length_seconds)
    print(u)
    
    def fun(n,x):
            if x <= n :
                return -x
            else :
                return x
        
    vfun=np.vectorize(fun)
    
    perturbation=vfun(u,time)
    print(np.size(perturbation))
    print(np.size(signal))
    
    plt.plot(time, perturbation.T)
    plt.show()
    
    return(perturbation)
    
def register_signal(filename) :
    with open(filename, newline='') as csvfile :
         reader=csv.reader(csvfile, delimiter=',')
         for n,row in enumerate(reader) :
             if n==0 :
                 length=len(row)
                 x=np.empty((1,length))
             else :
                 time_series=np.zeros((1,length))
                 for k,column in enumerate(row) :
                     time_series[0,k]=column
                    
                 x=np.vstack((x,time_series))
    return(x)
             
             
def generate_multi(n, length_seconds, sampling_rate, frequencies_array, add_noise=0):

    assert n==frequencies_array.shape[0], "the number of freq provided does not match the number of signal you specify"

    storage=np.zeros((3*n,length_seconds*sampling_rate))
    time=np.arange(0,length_seconds*sampling_rate)/sampling_rate
    
    for k in range(n):
        signal,_,_=generate_signal(length_seconds, sampling_rate, list(frequencies_array[k]), add_noise=add_noise)
        storage[k,:]=signal
        storage[k+1,:]=signal*np.sin(time)
        storage[k+2,:]=signal*np.cos(time)

    indice=np.arange(3*n)
    rd.shuffle(indice)

    toregister=np.zeros(storage.shape)
    for k,i in enumerate(indice) :
        toregister[i,:]=storage[k,:]

    return(toregister)

def load_signal(filename):
    filename='./data/{}/signal.txt'.format(filename)
    time_series=np.loadtxt(filename)
    return(time_series)

            
def main():
    freq=[0.5,5,10,100]
    length_seconds=100
    sampling_rate=150
    generate_signal(length_seconds, sampling_rate, freq, plot=True)

if __name__=='__main__':
    main()
