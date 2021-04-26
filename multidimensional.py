import numpy as np
import generate_signal as gs


def generate_multi(n, length_seconds, sampling_rate, frequencies_array, add_noise=0):

    assert n==frequencies_array.shape[0], "the number of freq provided does not match the number of signal you specify"

    storage=np.zeros((3*n,length_seconds*sampling_rate))
    time=np.arange(0,length_seconds*sampling_rate)/sampling_rate
    
    for k in range(n):
        signal=generate_signal(length_seconds, sampling_rate, list(frequencies_array[k]), add_noise=add_noise)
        storage[k]=signal
        storage[k+1]=signal*np.sin(time)
        storage[k+2]=signal*np.cos(time)

    indice=random.shuffle(np.arange(3*n))

    toregister=np.zeros(storage.shape)
    for k,i in enumerate(indice) :
        toregister[i]=storage[k]

    return(toregister)

