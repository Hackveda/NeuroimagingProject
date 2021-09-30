import tensorflow.keras
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *


def get_model():
	aliases = {}
	Input_11 = Input(shape=(128, 128, 3), name='Input_11')
	UpSampling2D_6 = UpSampling2D(name='UpSampling2D_6')(Input_11)
	Conv2D_107 = Conv2D(name='Conv2D_107',filters= 3,kernel_size= (1,1))(UpSampling2D_6)
	UpSampling2D_7 = UpSampling2D(name='UpSampling2D_7')(Conv2D_107)
	Conv2D_108 = Conv2D(name='Conv2D_108',filters= 3,kernel_size= (1,1))(UpSampling2D_7)
	Conv2D_109 = Conv2D(name='Conv2D_109',filters= 3,kernel_size= (3,1),dilation_rate= 6,use_bias= False)(Conv2D_108)
	Conv2D_110 = Conv2D(name='Conv2D_110',filters= 3,kernel_size= (1,3),dilation_rate= 6,use_bias= False)(Conv2D_109)
	Conv2D_111 = Conv2D(name='Conv2D_111',filters= 3,kernel_size= (3,1),dilation_rate= 12,activation= 'softmax' )(Conv2D_110)
	Conv2D_112 = Conv2D(name='Conv2D_112',filters= 3,kernel_size= (1,3),dilation_rate= 12)(Conv2D_111)
	Conv2D_113 = Conv2D(name='Conv2D_113',filters= 3,kernel_size= (3,1),dilation_rate= 18,use_bias= False)(Conv2D_112)
	Conv2D_114 = Conv2D(name='Conv2D_114',filters= 3,kernel_size= (1,3),dilation_rate= 18)(Conv2D_113)
	Conv2D_115 = Conv2D(name='Conv2D_115',filters= 3,kernel_size= (1,3),dilation_rate= 24)(Conv2D_114)
	Conv2D_116 = Conv2D(name='Conv2D_116',filters= 3,kernel_size= (3,1),dilation_rate= 24)(Conv2D_115)
	Conv2D_117 = Conv2D(name='Conv2D_117',filters= 3,kernel_size= (1,1))(Conv2D_116)
	MaxPooling2D_32 = MaxPooling2D(name='MaxPooling2D_32')(Conv2D_117)
	Conv2D_118 = Conv2D(name='Conv2D_118',filters= 3,kernel_size= (16,16),dilation_rate= 2,activation= 'softmax' ,use_bias= False)(MaxPooling2D_32)
	Conv2D_119 = Conv2D(name='Conv2D_119',filters= 3,kernel_size= (16,16),dilation_rate= 2,use_bias= False)(Conv2D_118)
	Conv2D_120 = Conv2D(name='Conv2D_120',filters= 3,kernel_size= (8, 8),dilation_rate= (1,1))(Conv2D_119)
	Conv2D_121 = Conv2D(name='Conv2D_121',filters= 3,kernel_size= (2,2))(Conv2D_120)

	model = Model([Input_11],[Conv2D_121])
	return aliases, model


from tensorflow.keras.optimizers import *

def get_optimizer():
	return Adadelta()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 100

def get_data_config():
	return '{"mapping": {"Label": {"type": "Categorical", "port": "", "shape": "", "options": {}}, "Filename": {"type": "Image", "port": "InputPort0,OutputPort0", "shape": "", "options": {"pretrained": "InceptionV3", "Augmentation": true, "rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "shear_range": 0, "horizontal_flip": false, "vertical_flip": false, "Scaling": 1, "Normalization": false, "Resize": true, "Width": "128", "Height": "128"}}}, "numPorts": 1, "samples": {"training": 932, "validation": 51, "test": 51, "split": 5}, "dataset": {"name": "BratsProcessednNative", "type": "private", "samples": 1036}, "datasetLoadOption": "batch", "shuffle": false, "kfold": 1}'