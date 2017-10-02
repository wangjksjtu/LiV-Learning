import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
import cnn_dvr

if __name__ == "__main__":
    test = cnn_dvr.Model()
    result = test.cnn_mlp()
    test.train(100, 64)
