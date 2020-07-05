from dataio import get_train_test_data
from model import train_model

if __name__=='__main__':
    x_train, x_val, y_train, y_val = get_train_test_data()
    model = train_model((x_train, y_train), (x_val, y_val))

