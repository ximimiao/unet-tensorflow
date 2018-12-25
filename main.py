import dataprovider
import net

if __name__ == '__main__':

    data = dataprovider.Dataprovider()
    model = net.Model(data)
    train = net.Train(data, model, 16)
    train.train(200000)