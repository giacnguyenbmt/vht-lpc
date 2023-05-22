import argparse
import os
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import TrafficLightNetModel
from utils import preprocessing_image, load_weights_transfer, save_dataloader_img


SEED = 999
AUTOTUNE = tf.data.experimental.AUTOTUNE

def parse_args():
    parser = argparse.ArgumentParser(description='Data augmentation')
    parser.add_argument('--train_path', default="./data_total/train", help='train folder path', type=str)
    parser.add_argument('--val_path', default="./data_total/val", help='val folder path', type=str)
    parser.add_argument('--model_path', default=None, help='model file path', type=str)
    parser.add_argument('--output_path', default="./models/Modeltflight_13_12_time0.h5", help='output model file path', type=str)
    parser.add_argument('--lr', default=0.0005, help='learning rate', type=float)
    parser.add_argument('--batchsize', default=64, help='batch size', type=int)
    parser.add_argument('--epoch', default=10, help='num epoch', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    train_data_gen = ImageDataGenerator(
                                rotation_range=10,
                                # preprocessing_function=preprocessing_image,
                                preprocessing_function=None,
                                data_format='channels_last',
                                rescale=1. / 255,
                                )

    val_data_gen = ImageDataGenerator(
                                rotation_range=10,
                                # preprocessing_function=preprocessing_image,
                                preprocessing_function=None,
                                data_format='channels_last',
                                rescale=1. / 255,
                                )

    train_generator=train_data_gen.flow_from_directory(
							    directory = args.train_path,
							    target_size=(75, 75),
							    color_mode='rgb',
							    classes=['b', 'r', 'w', 'y'],
							    class_mode='categorical',
							    batch_size=args.batchsize,
							    shuffle=True,
							    seed=SEED,
							    interpolation='nearest',
							)
    valid_generator=val_data_gen.flow_from_directory(
							    directory = args.val_path,
							    target_size=(75, 75),
							    color_mode='rgb',
							    classes=['b', 'r', 'w', 'y'],
							    class_mode='categorical',
							    batch_size=args.batchsize,
							    shuffle=False,
							    seed=SEED,
							    interpolation='nearest',
							)



    tf_model = TrafficLightNetModel((75, 75, 3), 4, 256)
    # model_old = tf.keras.models.load_model("/content/drive/MyDrive/Project/TrafficLight/Modeltflight_08_11_time1.h5")
    if args.model_path != None:
        tf_model.load_model(args.model_path)

    # load_weights_transfer(model_old, tf_model.model, 11)

    # plot batch of train image with augment
    files = glob.glob('train_img_trans/*')
    for f in files:
        os.remove(f)
    if os.path.isdir('train_img_trans') is False:
        os.makedirs('train_img_trans')
    for batch in train_generator:
        images = batch[0]
        labels = batch[1]
        for i in range(len(labels)):
            save_dataloader_img('train_img_trans/'+str(i)+'_'+str(labels[i])+'.jpg',images[i])
        break

    # plot batch of val image with augment
    files = glob.glob('val_img_trans/*')
    for f in files:
        os.remove(f)
    if os.path.isdir('val_img_trans') is False:
        os.makedirs('val_img_trans')
    for batch in valid_generator:
        images = batch[0]
        labels = batch[1]
        for i in range(len(labels)):
            save_dataloader_img('val_img_trans/'+str(i)+'_'+str(labels[i])+'.jpg',images[i])
        break

    his = tf_model.train_model(path_save=args.output_path, 
                               ds_train=train_generator, 
                               epochs=args.epoch, 
                               batch_size=args.batchsize, 
                               ds_val=valid_generator, 
                               lr=args.lr, 
                               verbose=1)

if __name__ == "__main__":
    main()
