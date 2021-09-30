import argparse
import os
import sys
import csv
import pickle
import glob
import oyaml as yaml
import mxnet as mx
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms
from gluoncv import data
from gluoncv.model_zoo import get_model


def load_classes(run_dir):
    train_classes = []
    with open(run_dir+"/categories.pkl", 'rb') as f:
        train_classes = pickle.load(f)
    
    return train_classes

def write_data(filename, data):
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def read_data(filename):
    temp_data = {}
    with open(filename, "r") as f:
        temp_data = yaml.safe_load(f)
    return temp_data

#def predict_image(config_yaml, params_dir, input_pic, result_dir):
def predict_image(run_dir, input_pic, result_dir, max_file=10):

    if os.path.isfile(input_pic):
        input_files = [input_pic]
    elif isinstance(input_pic, list):
        input_files = [input_file for input_file in input_pic if os.path.exists(input_pic)]
    else:
        input_files = glob.glob(os.path.join(input_pic, "*.*"))
    input_files = [input_pic for input_pic in input_files if input_pic.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    input_files = input_files[:max_file]
    config_yaml = os.path.join(run_dir, 'config.yaml')

    os.makedirs(result_dir, exist_ok=True)
    
    project_settings = read_data(config_yaml)
    model_name = project_settings.get("net_name", "resnet50_v2")
    pretrained = project_settings.get("pretrained", True)
    saved_params = run_dir + "/{}-best.params".format(model_name)
    if not os.path.isfile(saved_params):
        sys.stderr.write('Error in starting inference. Weight file does not exist')

    class_names = load_classes(run_dir)
    classes = len(class_names)
    # print(class_names)

    context = [mx.cpu()]

    pretrained = True if saved_params == '' else False
    kwargs = {'classes': classes, 'pretrained': pretrained}
    net = get_model(model_name, **kwargs)

    if not pretrained:
        net.load_parameters(saved_params, ctx=context)



    # Transform
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])

    # Load Images
    result = {}
    for input_pic in input_files:
        img = image.imread(input_pic)
        img = transform_fn(img)
        pred = net(img.expand_dims(axis=0))

        ind = nd.argmax(pred, axis=1).astype('int')
        pred_class, prob = (class_names[ind.asscalar()], nd.softmax(pred)[
                            0][ind].asscalar())
        #data = [pred_class, str(prob)]
        result[input_pic] = [pred_class, str(prob)]
        print('The input picture is classified to be [%s], with probability %.3f.' % (pred_class, prob))
    return result

def parse_args():
    parser = argparse.ArgumentParser(
        description='test a model for image classification.')

    # parser.add_argument('--config_yaml', type=str, required=True,
    #                     help='Project directory')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='path to the input image or directory containing images')
    # parser.add_argument('--params_dir', type=str, required=True,
    #                     help='Run directory')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Result directory')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    args = parse_args()
    input = args.input
    
    result_file = os.path.join(args.result_dir, "test_result.csv")
    if os.path.exists(result_file):
        os.remove(result_file)

    out_data = predict_image(args.run_dir, input_pic=input, result_dir= args.result_dir)
    data = []
    for k, v in out_data.items():
        file_name = [os.path.basename(k)]
        data.append(file_name + v)
    
    if data:
        with open(result_file, 'w', newline='') as fp:
            a = csv.writer(fp, delimiter=',')
            a.writerow(['image', 'prediction', 'probabilities'])
            a.writerows(data)
