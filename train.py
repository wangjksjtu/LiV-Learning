import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
import cnn_dvr
import cnn_dvr_fmap

if __name__ == "__main__":

    test = cnn_dvr.Model()
    result = test.cnn_mlp()
    test.train(1, 64)
    '''
    test2 = cnn_dvr_fmap.Model()
    result = test2.cnn_mlp()
    test2.train(1, 64)
    '''
