# -PaddleSeg-
【AI达人创造营第二期】 基于PaddleSeg的肺部图像分割

## 一、项目背景

在示例项目：‘基于PaddleSeg的眼底血管分割——使用飞桨助力医学影像分析’的启发下选择了利用PaddleSeg对数据集‘胸部x光肺部分割数据’进行了图像分割处理。

## 二、数据介绍

本项目使用的数据集名为‘胸部x光肺部分割数据’，内含两组各51张.png及.jpg格式的图片。
数据集的创建者利用Labelme工具标记X光中的肺部 

## 三、模型训练

### 1、解压数据集

In [1]
! unzip -oq /home/aistudio/data/data57558/chest.zip -d work/

### 2、生成图像列表

In [3]
import os

path_origin = 'work/chest/origin/'
path_seg = 'work/chest/seg/'
pic_dir = os.listdir(path_origin)

f_train = open('train_list.txt', 'w')
f_val = open('val_list.txt', 'w')

for i in range(len(pic_dir)):
    if i % 30 != 0:
        f_train.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')
    else:
        f_val.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')

f_train.close()
f_val.close()

3、安装PaddleSeg

In [11]
!unzip -o work/PaddleSeg.zip

%cd PaddleSeg

!pip install -r requirements.txt

/home/aistudio/PaddleSeg
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple

### 4、配置超参数并训练模型

#数据集配置

DATASET:
    DATA_DIR: "/home/aistudio/"
    NUM_CLASSES: 2
    TEST_FILE_LIST: "/home/aistudio/val_list.txt"
    TRAIN_FILE_LIST: "/home/aistudio/train_list.txt"
    VAL_FILE_LIST: "/home/aistudio/val_list.txt"
    VIS_FILE_LIST: "/home/aistudio/val_list.txt"

#预训练模型配置

MODEL:
    MODEL_NAME: "unet"
    DEFAULT_NORM_TYPE: "bn"

#其他配置

TRAIN_CROP_SIZE: (512, 512)
EVAL_CROP_SIZE: (512, 512)
AUG:
    AUG_METHOD: "unpadding"
    FIX_RESIZE_SIZE: (512, 512)
    
    # 图像镜像左右翻转
    
    MIRROR: True
    RICH_CROP:
        
        # RichCrop数据增广开关，用于提升模型鲁棒性
        
        ENABLE: True
        
        # 图像旋转最大角度，0-90
        
        MAX_ROTATION: 15
        
        # 裁取图像与原始图像面积比，0-1
        
        MIN_AREA_RATIO: 0.5
        
        # 裁取图像宽高比范围，非负
        
        ASPECT_RATIO: 0.33
        
        # 亮度调节范围，0-1
        
        BRIGHTNESS_JITTER_RATIO: 0.2
        
        # 饱和度调节范围，0-1
        
        SATURATION_JITTER_RATIO: 0.2
        
        # 对比度调节范围，0-1
        
        
        CONTRAST_JITTER_RATIO: 0.2
        
        # 图像模糊开关，True/False
        
        BLUR: False
        
        # 图像启动模糊百分比，0-1
        
        BLUR_RATIO: 0.1
        
BATCH_SIZE: 4

TRAIN:
    PRETRAINED_MODEL_DIR: "/home/aistudio/PaddleSeg/pretrained_model/unet_bn_coco/"
    MODEL_SAVE_DIR: "/home/aistudio/saved_model/unet_optic/"
    SNAPSHOT_EPOCH: 5

TEST:
    TEST_MODEL: "/home/aistudio/saved_model/unet_optic/final"

SOLVER:
    NUM_EPOCHS: 500
    LR: 0.001
    LR_POLICY: "poly"
    OPTIMIZER: "adam"

In [4]
!python /home/aistudio/PaddleSeg/pretrained_model/download_model.py "unet_bn_coco"
Pretrained Model download success!
In [5]
!export CUDA_VISIBLE_DEVICES=0

!python /home/aistudio/PaddleSeg/pdseg/train.py --use_gpu --cfg /home/aistudio/PaddleSeg/configs/unet_optic.yaml --do_eval 
{'AUG': {'AUG_METHOD': 'unpadding',
         'FIX_RESIZE_SIZE': (512, 512),
         'FLIP': False,
         'FLIP_RATIO': 0.5,
         'INF_RESIZE_VALUE': 500,
         'MAX_RESIZE_VALUE': 600,
         'MAX_SCALE_FACTOR': 2.0,
         'MIN_RESIZE_VALUE': 400,
         'MIN_SCALE_FACTOR': 0.5,
         'MIRROR': True,
         'RICH_CROP': {'ASPECT_RATIO': 0.33,
                       'BLUR': False,
                       'BLUR_RATIO': 0.1,
                       'BRIGHTNESS_JITTER_RATIO': 0.2,
                       'CONTRAST_JITTER_RATIO': 0.2,
                       'ENABLE': True,
                       'MAX_ROTATION': 15,
                       'MIN_AREA_RATIO': 0.5,
                       'SATURATION_JITTER_RATIO': 0.2},
         'SCALE_STEP_SIZE': 0.25,
         'TO_RGB': False},
 'BATCH_SIZE': 4,
 'DATALOADER': {'BUF_SIZE': 256, 'NUM_WORKERS': 8},
 'DATASET': {'DATA_DIM': 3,
             'DATA_DIR': '/home/aistudio/',
             'IGNORE_INDEX': 255,
             'IMAGE_TYPE': 'rgb',
             'NUM_CLASSES': 2,
             'PADDING_VALUE': [127.5, 127.5, 127.5],
             'SEPARATOR': ' ',
             'TEST_FILE_LIST': '/home/aistudio/val_list.txt',
             'TEST_TOTAL_IMAGES': 2,
             'TRAIN_FILE_LIST': '/home/aistudio/train_list.txt',
             'TRAIN_TOTAL_IMAGES': 49,
             'VAL_FILE_LIST': '/home/aistudio/val_list.txt',
             'VAL_TOTAL_IMAGES': 2,
             'VIS_FILE_LIST': '/home/aistudio/val_list.txt'},
 'EVAL_CROP_SIZE': (512, 512),
 'FREEZE': {'MODEL_FILENAME': '__model__',
            'PARAMS_FILENAME': '__params__',
            'SAVE_DIR': 'freeze_model'},
 'MEAN': [0.5, 0.5, 0.5],
 'MODEL': {'BN_MOMENTUM': 0.99,
           'DEEPLAB': {'ASPP_WITH_SEP_CONV': True,
                       'BACKBONE': 'xception_65',
                       'BACKBONE_LR_MULT_LIST': None,
                       'DECODER': {'CONV_FILTERS': 256,
                                   'OUTPUT_IS_LOGITS': False,
                                   'USE_SUM_MERGE': False},
                       'DECODER_USE_SEP_CONV': True,
                       'DEPTH_MULTIPLIER': 1.0,
                       'ENABLE_DECODER': True,
                       'ENCODER': {'ADD_IMAGE_LEVEL_FEATURE': True,
                                   'ASPP_CONVS_FILTERS': 256,
                                   'ASPP_RATIOS': None,
                                   'ASPP_WITH_CONCAT_PROJECTION': True,
                                   'ASPP_WITH_SE': False,
                                   'POOLING_CROP_SIZE': None,
                                   'POOLING_STRIDE': [1, 1],
                                   'SE_USE_QSIGMOID': False},
                       'ENCODER_WITH_ASPP': True,
                       'OUTPUT_STRIDE': 16},
           'DEFAULT_EPSILON': 1e-05,
           'DEFAULT_GROUP_NUMBER': 32,
           'DEFAULT_NORM_TYPE': 'bn',
           'FP16': False,
           'HRNET': {'STAGE2': {'NUM_CHANNELS': [40, 80], 'NUM_MODULES': 1},
                     'STAGE3': {'NUM_CHANNELS': [40, 80, 160],
                                'NUM_MODULES': 4},
                     'STAGE4': {'NUM_CHANNELS': [40, 80, 160, 320],
                                'NUM_MODULES': 3}},
           'ICNET': {'DEPTH_MULTIPLIER': 0.5, 'LAYERS': 50},
           'MODEL_NAME': 'unet',
           'MULTI_LOSS_WEIGHT': [1.0],
           'OCR': {'OCR_KEY_CHANNELS': 256, 'OCR_MID_CHANNELS': 512},
           'PSPNET': {'DEPTH_MULTIPLIER': 1, 'LAYERS': 50},
           'SCALE_LOSS': 'DYNAMIC',
           'UNET': {'UPSAMPLE_MODE': 'bilinear'}},
 'NUM_TRAINERS': 1,
 'SLIM': {'KNOWLEDGE_DISTILL': False,
          'KNOWLEDGE_DISTILL_IS_TEACHER': False,
          'KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR': '',
          'NAS_ADDRESS': '',
          'NAS_IS_SERVER': True,
          'NAS_PORT': 23333,
          'NAS_SEARCH_STEPS': 100,
          'NAS_SPACE_NAME': '',
          'NAS_START_EVAL_EPOCH': 0,
          'PREPROCESS': False,
          'PRUNE_PARAMS': '',
          'PRUNE_RATIOS': []},
 'SOLVER': {'BEGIN_EPOCH': 1,
            'CROSS_ENTROPY_WEIGHT': None,
            'DECAY_EPOCH': [10, 20],
            'GAMMA': 0.1,
            'LOSS': ['softmax_loss'],
            'LOSS_WEIGHT': {'BCE_LOSS': 1,
                            'DICE_LOSS': 1,
                            'LOVASZ_HINGE_LOSS': 1,
                            'LOVASZ_SOFTMAX_LOSS': 1,
                            'SOFTMAX_LOSS': 1},
            'LR': 0.001,
            'LR_POLICY': 'poly',
            'LR_WARMUP': False,
            'LR_WARMUP_STEPS': 2000,
            'MOMENTUM': 0.9,
            'MOMENTUM2': 0.999,
            'NUM_EPOCHS': 500,
            'OPTIMIZER': 'adam',
            'POWER': 0.9,
            'WEIGHT_DECAY': 4e-05},
 'STD': [0.5, 0.5, 0.5],
 'TEST': {'TEST_MODEL': '/home/aistudio/saved_model/unet_optic/final'},
 'TRAIN': {'MODEL_SAVE_DIR': '/home/aistudio/saved_model/unet_optic/',
           'PRETRAINED_MODEL_DIR': '/home/aistudio/PaddleSeg/pretrained_model/unet_bn_coco/',
           'RESUME_MODEL_DIR': '',
           'SNAPSHOT_EPOCH': 5,
           'SYNC_BATCH_NORM': False},
 'TRAINER_ID': 0,
 'TRAIN_CROP_SIZE': (512, 512)}
#Device count: 1
batch_size_per_dev: 4
Traceback (most recent call last):
  File "/home/aistudio/PaddleSeg/pdseg/train.py", line 466, in <module>
    main(args)
  File "/home/aistudio/PaddleSeg/pdseg/train.py", line 453, in main
    train(cfg)
  File "/home/aistudio/PaddleSeg/pdseg/train.py", line 237, in train
    train_prog, startup_prog, phase=ModelPhase.TRAIN)
  File "/home/aistudio/PaddleSeg/pdseg/models/model_builder.py", line 133, in build_model
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
  File "<decorator-gen-25>", line 2, in data
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/wrapped_decorator.py", line 25, in __impl__
    return wrapped_func(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py", line 237, in __impl__
    ), "In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and '%s()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode." % func.__name__
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.
四、模型评估
1、测试模型效果
In [6]
!python /home/aistudio/PaddleSeg/pdseg/eval.py --cfg /home/aistudio/PaddleSeg/configs/unet_optic.yaml \
                        --use_gpu \
                        EVAL_CROP_SIZE "(512, 512)"
{'AUG': {'AUG_METHOD': 'unpadding',
         'FIX_RESIZE_SIZE': (512, 512),
         'FLIP': False,
         'FLIP_RATIO': 0.5,
         'INF_RESIZE_VALUE': 500,
         'MAX_RESIZE_VALUE': 600,
         'MAX_SCALE_FACTOR': 2.0,
         'MIN_RESIZE_VALUE': 400,
         'MIN_SCALE_FACTOR': 0.5,
         'MIRROR': True,
         'RICH_CROP': {'ASPECT_RATIO': 0.33,
                       'BLUR': False,
                       'BLUR_RATIO': 0.1,
                       'BRIGHTNESS_JITTER_RATIO': 0.2,
                       'CONTRAST_JITTER_RATIO': 0.2,
                       'ENABLE': True,
                       'MAX_ROTATION': 15,
                       'MIN_AREA_RATIO': 0.5,
                       'SATURATION_JITTER_RATIO': 0.2},
         'SCALE_STEP_SIZE': 0.25,
         'TO_RGB': False},
 'BATCH_SIZE': 4,
 'DATALOADER': {'BUF_SIZE': 256, 'NUM_WORKERS': 8},
 'DATASET': {'DATA_DIM': 3,
             'DATA_DIR': '/home/aistudio/',
             'IGNORE_INDEX': 255,
             'IMAGE_TYPE': 'rgb',
             'NUM_CLASSES': 2,
             'PADDING_VALUE': [127.5, 127.5, 127.5],
             'SEPARATOR': ' ',
             'TEST_FILE_LIST': '/home/aistudio/val_list.txt',
             'TEST_TOTAL_IMAGES': 2,
             'TRAIN_FILE_LIST': '/home/aistudio/train_list.txt',
             'TRAIN_TOTAL_IMAGES': 49,
             'VAL_FILE_LIST': '/home/aistudio/val_list.txt',
             'VAL_TOTAL_IMAGES': 2,
             'VIS_FILE_LIST': '/home/aistudio/val_list.txt'},
 'EVAL_CROP_SIZE': (512, 512),
 'FREEZE': {'MODEL_FILENAME': '__model__',
            'PARAMS_FILENAME': '__params__',
            'SAVE_DIR': 'freeze_model'},
 'MEAN': [0.5, 0.5, 0.5],
 'MODEL': {'BN_MOMENTUM': 0.99,
           'DEEPLAB': {'ASPP_WITH_SEP_CONV': True,
                       'BACKBONE': 'xception_65',
                       'BACKBONE_LR_MULT_LIST': None,
                       'DECODER': {'CONV_FILTERS': 256,
                                   'OUTPUT_IS_LOGITS': False,
                                   'USE_SUM_MERGE': False},
                       'DECODER_USE_SEP_CONV': True,
                       'DEPTH_MULTIPLIER': 1.0,
                       'ENABLE_DECODER': True,
                       'ENCODER': {'ADD_IMAGE_LEVEL_FEATURE': True,
                                   'ASPP_CONVS_FILTERS': 256,
                                   'ASPP_RATIOS': None,
                                   'ASPP_WITH_CONCAT_PROJECTION': True,
                                   'ASPP_WITH_SE': False,
                                   'POOLING_CROP_SIZE': None,
                                   'POOLING_STRIDE': [1, 1],
                                   'SE_USE_QSIGMOID': False},
                       'ENCODER_WITH_ASPP': True,
                       'OUTPUT_STRIDE': 16},
           'DEFAULT_EPSILON': 1e-05,
           'DEFAULT_GROUP_NUMBER': 32,
           'DEFAULT_NORM_TYPE': 'bn',
           'FP16': False,
           'HRNET': {'STAGE2': {'NUM_CHANNELS': [40, 80], 'NUM_MODULES': 1},
                     'STAGE3': {'NUM_CHANNELS': [40, 80, 160],
                                'NUM_MODULES': 4},
                     'STAGE4': {'NUM_CHANNELS': [40, 80, 160, 320],
                                'NUM_MODULES': 3}},
           'ICNET': {'DEPTH_MULTIPLIER': 0.5, 'LAYERS': 50},
           'MODEL_NAME': 'unet',
           'MULTI_LOSS_WEIGHT': [1.0],
           'OCR': {'OCR_KEY_CHANNELS': 256, 'OCR_MID_CHANNELS': 512},
           'PSPNET': {'DEPTH_MULTIPLIER': 1, 'LAYERS': 50},
           'SCALE_LOSS': 'DYNAMIC',
           'UNET': {'UPSAMPLE_MODE': 'bilinear'}},
 'NUM_TRAINERS': 1,
 'SLIM': {'KNOWLEDGE_DISTILL': False,
          'KNOWLEDGE_DISTILL_IS_TEACHER': False,
          'KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR': '',
          'NAS_ADDRESS': '',
          'NAS_IS_SERVER': True,
          'NAS_PORT': 23333,
          'NAS_SEARCH_STEPS': 100,
          'NAS_SPACE_NAME': '',
          'NAS_START_EVAL_EPOCH': 0,
          'PREPROCESS': False,
          'PRUNE_PARAMS': '',
          'PRUNE_RATIOS': []},
 'SOLVER': {'BEGIN_EPOCH': 1,
            'CROSS_ENTROPY_WEIGHT': None,
            'DECAY_EPOCH': [10, 20],
            'GAMMA': 0.1,
            'LOSS': ['softmax_loss'],
            'LOSS_WEIGHT': {'BCE_LOSS': 1,
                            'DICE_LOSS': 1,
                            'LOVASZ_HINGE_LOSS': 1,
                            'LOVASZ_SOFTMAX_LOSS': 1,
                            'SOFTMAX_LOSS': 1},
            'LR': 0.001,
            'LR_POLICY': 'poly',
            'LR_WARMUP': False,
            'LR_WARMUP_STEPS': 2000,
            'MOMENTUM': 0.9,
            'MOMENTUM2': 0.999,
            'NUM_EPOCHS': 500,
            'OPTIMIZER': 'adam',
            'POWER': 0.9,
            'WEIGHT_DECAY': 4e-05},
 'STD': [0.5, 0.5, 0.5],
 'TEST': {'TEST_MODEL': '/home/aistudio/saved_model/unet_optic/final'},
 'TRAIN': {'MODEL_SAVE_DIR': '/home/aistudio/saved_model/unet_optic/',
           'PRETRAINED_MODEL_DIR': '/home/aistudio/PaddleSeg/pretrained_model/unet_bn_coco/',
           'RESUME_MODEL_DIR': '',
           'SNAPSHOT_EPOCH': 5,
           'SYNC_BATCH_NORM': False},
 'TRAINER_ID': 0,
 'TRAIN_CROP_SIZE': (512, 512)}
Traceback (most recent call last):
  File "/home/aistudio/PaddleSeg/pdseg/eval.py", line 178, in <module>
    main()
  File "/home/aistudio/PaddleSeg/pdseg/eval.py", line 174, in main
    evaluate(cfg, **args.__dict__)
  File "/home/aistudio/PaddleSeg/pdseg/eval.py", line 92, in evaluate
    test_prog, startup_prog, phase=ModelPhase.EVAL)
  File "/home/aistudio/PaddleSeg/pdseg/models/model_builder.py", line 133, in build_model
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
  File "<decorator-gen-25>", line 2, in data
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/wrapped_decorator.py", line 25, in __impl__
    return wrapped_func(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py", line 237, in __impl__
    ), "In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and '%s()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode." % func.__name__
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.
In [11]
%matplotlib inline
import matplotlib.pyplot as plt

def display(img_dir):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask','Predicted Mask']
    
    for i in range(len(title)):
        plt.subplot(1, len(img_dir), i+1)
        plt.title(title[i])
        img = plt.imread(img_dir[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()

    随机输入一张胸部X光图片可以得到其肺部图像分割图片

## 五、总结与升华
    
本项目基于PaddleSeg对该数据集进行了简单的处理，用以分割出胸部X光照片中的肺部。因自身实力欠缺，多用借鉴，设计称不上完善，项目仍有许多需要改进的地方，如肺部形状图像分割出后对原图像进行读取和分割。

## 六、个人总结
    
东北大学秦皇岛分校本一在读

初次接触相关内容难免有瑕疵

个人主页链接：

https://aistudio.baidu.com/aistudio/personalcenter/thirdview/1998037
