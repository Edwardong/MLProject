import utils.mnist_reader as um

def Load():
	X_train, y_train = um.load_mnist('data/fashion', kind='train')
	X_test, y_test = um.load_mnist('data/fashion', kind='t10k')
	return X_train, y_train, X_test, y_test
