from matplotlib import pyplot as plt
import torch
import argparse
from load_data import get_data, register_training_signal, normalize_datas
import numpy as np
from mini_model import TransformerModel

def main(model_name) :
    
# load the model optimized on GPU on CPU
    backast_size=100
    forecast_size=100
    quantiles=[0.05, 0.5, 0.95]
    model_path='./data/{}/mymodel.pt'.format(model_name)
    model_state_dict=torch.load(model_path, map_location=torch.device('cpu'))
    model=TransformerModel(backast_size, forecast_size, quantiles) # warning : architecture must be the same, make sure you do not change hyperparameters between 2 sessions
    model.load_state_dict(model_state_dict)

# get the data
    step=10
    data_set=register_training_signal(nlimit=0).transpose() #[time_step]x[dim] > [dim]x[time_step]
    data_set=normalize_datas(data_set)
    x_train, y_train = get_data(backast_size, forecast_size, step, data_set)
    print('|------------------ xtrain.shape : {} | ', x_train.shape) # [N][backast_size][dim]
    print('|------------------ ytrain.shape : {} | ', y_train.shape)

 

    x=torch.tensor(x_train[:,:,:],dtype=torch.float)

    # compute the quantiles

    pred_low = model(x,0).transpose(0,1).detach().numpy()
    print(pred_low.shape)
    sys.exit()

    median = model(x,1).tranpose(0,1).detach().numpy()
    pred_high = model(x,2).transpose(0,1).detach().numpy()
    numpy_x=x.numpy()
    numpy_y=y_train[:,:,:]

    plt.figure(figsize=(5,3))

    def merge_with_x(tab) :    
        to_plot=np.zeros(numpy_x.shape[1]+tab.shape[1])
        to_plot[:numpy_x.shape[1]]=numpy_x
        to_plot[numpy_x.shape[1]:]=tab
        return(to_plot)

    id=3 # random btw 0 and N-1
    plt.plot(merge_with_x(numpy_y[id,:,:]), color="gray", lw=2, legend='original signal')

    plt.fill_between(merge_with_x(pred_low[id,:,:]), merge_with_x(pred_high[id,:,:]), color="gray", alpha=0.25, label='aleatoric')

    plt.legend(loc=3)
    plt.show()

if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('model_name', help='Name of the model you wanna load')
    args=parser.parse_args()
    print(args.model_name)
    main(args.model_name)
