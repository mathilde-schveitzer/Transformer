import numpy as np
import argparse
from process_data import from_txt_file_filter




def main(id):
   # name=pretreatment(id)
    name='TM11_*.txt'
    data=np.loadtxt('./data/'+name)
    y=from_txt_file_filter(data)
    print(y.shape)
    np.savetxt('./data/cleaned_time_series_11', y)
    


if __name__ == '__main__' :
    parser=argparse.ArgumentParser()
    parser.add_argument('id', help='Identify the TM you want to analyze')
    args=parser.parse_args()
    main(int(args.id))
