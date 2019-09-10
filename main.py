from __future__ import print_function //输出函数
import os  //调用系统命令，文件路径等
import argparse//为PY文件封装好可以选择的参数
from glob import glob//文件搜索模块

from PIL import Image //图像处理标准库
import tensorflow as tf

from model import lowlight_enhance
from utils import *

parser = argparse.ArgumentParser(description='')//添加参数

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoches')//全部样本训练的次数
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')//每批数据量的大小，一次训练16张图片，一次迭代训练31次
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')//子图像块大小
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=20, help='evaluating and saving checkpoints every #  epoch')多次保存模型
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')//保存模型目录
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')//评价输出的目录

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')//测试输出的目录
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/low', help='directory for testing inputs')//测试输入的目录
parser.add_argument('--decom', dest='decom', default=0, help='decom flag, 0 for enhanced results only and 1 for decomposition results')//只保存增强的结果

args = parser.parse_args()//创建对象

def lowlight_train(lowlight_enhance)://定义函数：低照度训练，参数为低照度增强
    if not os.path.exists(args.ckpt_dir)://创建保存模型目录和评价输出目录
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])//产生100 个一维数组，每个元素乘以学习率
    lr[20:] = lr[0] / 10.0 //前20次学习率是0.001，20到100次的时候就是0.0001

    train_low_data = []//训练低光照和正常光照图片
    train_high_data = []

    train_low_data_names = glob('./data/our485/low/*.png') + glob('./data/syn/low/*.png')
    train_low_data_names.sort()//图片按标号排序
    train_high_data_names = glob('./data/our485/high/*.png') + glob('./data/syn/high/*.png')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names))://低光照和正常光照图片都以矩阵的形式存储起来，方便后续操作
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)

    eval_low_data = []
    eval_high_data = []

    eval_low_data_name = glob('./data/eval/low/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])//为什么要加 eval_low_im，也加入训练呢？
        eval_low_data.append(eval_low_im)
//os.path.join(path1[,path2[,......]]),将多个路径组合后返回,这里的decome和relight是用来干嘛的
    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, 
                           epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'),
                           eval_every_epoch=args.eval_every_epoch, train_phase="Decom")

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, 
                           epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'),
                           eval_every_epoch=args.eval_every_epoch, train_phase="Relight")


def lowlight_test(lowlight_enhance):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_low_data = []
    test_high_data = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)

    lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, save_dir=args.save_dir, decom_flag=args.decom)


def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
