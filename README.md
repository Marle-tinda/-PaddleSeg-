# -PaddleSeg-
基于PaddleSeg的肺部图像分割
基于PaddleSeg的肺部图像分割
一、项目背景
在示例项目：‘基于PaddleSeg的眼底血管分割——使用飞桨助力医学影像分析’的启发下选择了利用PaddleSeg对数据集‘胸部x光肺部分割数据’进行了图像分割处理。

二、数据介绍
本项目使用的数据集名为‘胸部x光肺部分割数据’，内含两组各51张.png及.jpg格式的图片。
数据集的创建者利用Labelme工具标记X光中的肺部


例图如下 

三、模型训练
1、解压数据集
In [1]
! unzip -oq /home/aistudio/data/data57558/chest.zip -d work/
2、生成图像列表
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
Archive:  work/PaddleSeg.zip
   creating: PaddleSeg/
   creating: PaddleSeg/pretrained_model/
  inflating: PaddleSeg/pretrained_model/download_model.py  
  inflating: PaddleSeg/.copyright.hook  
   creating: PaddleSeg/tutorial/
  inflating: PaddleSeg/tutorial/finetune_icnet.md  
   creating: PaddleSeg/tutorial/imgs/
  inflating: PaddleSeg/tutorial/imgs/optic_icnet.png  
  inflating: PaddleSeg/tutorial/imgs/optic_deeplab.png  
  inflating: PaddleSeg/tutorial/imgs/optic_unet.png  
  inflating: PaddleSeg/tutorial/imgs/optic.png  
  inflating: PaddleSeg/tutorial/imgs/optic_pspnet.png  
  inflating: PaddleSeg/tutorial/imgs/optic_hrnet.png  
  inflating: PaddleSeg/tutorial/finetune_hrnet.md  
  inflating: PaddleSeg/tutorial/finetune_unet.md  
  inflating: PaddleSeg/tutorial/finetune_pspnet.md  
  inflating: PaddleSeg/tutorial/finetune_fast_scnn.md  
  inflating: PaddleSeg/tutorial/finetune_deeplabv3plus.md  
  inflating: PaddleSeg/tutorial/finetune_ocrnet.md  
  inflating: PaddleSeg/LICENSE       
  inflating: PaddleSeg/requirements.txt  
   creating: PaddleSeg/test/
  inflating: PaddleSeg/test/test_utils.py  
   creating: PaddleSeg/test/ci/
  inflating: PaddleSeg/test/ci/test_download_dataset.sh  
  inflating: PaddleSeg/test/ci/check_code_style.sh  
  inflating: PaddleSeg/test/local_test_pet.py  
  inflating: PaddleSeg/test/local_test_cityscapes.py  
   creating: PaddleSeg/test/configs/
  inflating: PaddleSeg/test/configs/unet_pet.yaml  
  inflating: PaddleSeg/test/configs/deeplabv3p_xception65_cityscapes.yaml  
  inflating: PaddleSeg/.pre-commit-config.yaml  
   creating: PaddleSeg/deploy/
   creating: PaddleSeg/deploy/serving/
   creating: PaddleSeg/deploy/serving/tools/
   creating: PaddleSeg/deploy/serving/tools/images/
  inflating: PaddleSeg/deploy/serving/tools/images/2.jpg  
  inflating: PaddleSeg/deploy/serving/tools/images/3.jpg  
  inflating: PaddleSeg/deploy/serving/tools/images/1.jpg  
  inflating: PaddleSeg/deploy/serving/tools/image_seg_client.py  
   creating: PaddleSeg/deploy/serving/seg-serving/
  inflating: PaddleSeg/deploy/serving/seg-serving/CMakeLists.txt  
   creating: PaddleSeg/deploy/serving/seg-serving/proto/
  inflating: PaddleSeg/deploy/serving/seg-serving/proto/CMakeLists.txt  
  inflating: PaddleSeg/deploy/serving/seg-serving/proto/image_seg.proto  
   creating: PaddleSeg/deploy/serving/seg-serving/op/
  inflating: PaddleSeg/deploy/serving/seg-serving/op/image_seg_op.cpp  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/CMakeLists.txt  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/image_seg_op.h  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/write_json_op.cpp  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/write_json_op.h  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/reader_op.h  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/seg_conf.cpp  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/reader_op.cpp  
  inflating: PaddleSeg/deploy/serving/seg-serving/op/seg_conf.h  
   creating: PaddleSeg/deploy/serving/seg-serving/scripts/
  inflating: PaddleSeg/deploy/serving/seg-serving/scripts/start.sh  
   creating: PaddleSeg/deploy/serving/seg-serving/data/
   creating: PaddleSeg/deploy/serving/seg-serving/data/model/
   creating: PaddleSeg/deploy/serving/seg-serving/data/model/paddle/
  inflating: PaddleSeg/deploy/serving/seg-serving/data/model/paddle/fluid_reload_flag  
  inflating: PaddleSeg/deploy/serving/seg-serving/data/model/paddle/fluid_time_file  
   creating: PaddleSeg/deploy/serving/seg-serving/conf/
  inflating: PaddleSeg/deploy/serving/seg-serving/conf/workflow.prototxt  
  inflating: PaddleSeg/deploy/serving/seg-serving/conf/seg_conf.yaml  
  inflating: PaddleSeg/deploy/serving/seg-serving/conf/gflags.conf  
  inflating: PaddleSeg/deploy/serving/seg-serving/conf/seg_conf2.yaml  
  inflating: PaddleSeg/deploy/serving/seg-serving/conf/resource.prototxt  
  inflating: PaddleSeg/deploy/serving/seg-serving/conf/service.prototxt  
  inflating: PaddleSeg/deploy/serving/seg-serving/conf/model_toolkit.prototxt  
  inflating: PaddleSeg/deploy/serving/requirements.txt  
  inflating: PaddleSeg/deploy/serving/COMPILE_GUIDE.md  
  inflating: PaddleSeg/deploy/serving/UBUNTU.md  
  inflating: PaddleSeg/deploy/serving/README.md  
   creating: PaddleSeg/deploy/python/
  inflating: PaddleSeg/deploy/python/requirements.txt  
   creating: PaddleSeg/deploy/python/docs/
  inflating: PaddleSeg/deploy/python/docs/compile_paddle_with_tensorrt.md  
  inflating: PaddleSeg/deploy/python/docs/PaddleSeg_Infer_Benchmark.md  
  inflating: PaddleSeg/deploy/python/README.md  
  inflating: PaddleSeg/deploy/python/infer.py  
   creating: PaddleSeg/deploy/lite/
   creating: PaddleSeg/deploy/lite/example/
  inflating: PaddleSeg/deploy/lite/example/human_2.png  
  inflating: PaddleSeg/deploy/lite/example/human_3.png  
  inflating: PaddleSeg/deploy/lite/example/human_1.png  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/proguard-rules.pro  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/local.properties  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/wrapper/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/wrapper/gradle-wrapper.jar  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradle/wrapper/gradle-wrapper.properties  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradlew  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/.gitignore  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/build.gradle  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/gradlew.bat  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/lite/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/lite/demo/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/androidTest/java/com/baidu/paddle/lite/demo/ExampleInstrumentedTest.java  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/lite/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/lite/demo/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/test/java/com/baidu/paddle/lite/demo/ExampleUnitTest.java  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-mdpi/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-mdpi/ic_launcher.png  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-mdpi/ic_launcher_round.png  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable-v24/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable-v24/ic_launcher_foreground.xml  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-hdpi/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-hdpi/ic_launcher.png  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-hdpi/ic_launcher_round.png  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/drawable/ic_launcher_background.xml  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxxhdpi/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxxhdpi/ic_launcher_round.png  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/layout/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/layout/activity_main.xml  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxhdpi/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxhdpi/ic_launcher.png  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xxhdpi/ic_launcher_round.png  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/colors.xml  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/arrays.xml  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/styles.xml  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/values/strings.xml  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/xml/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/xml/settings.xml  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/menu/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/menu/menu_action_options.xml  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xhdpi/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xhdpi/ic_launcher.png  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-xhdpi/ic_launcher_round.png  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-anydpi-v26/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/res/mipmap-anydpi-v26/ic_launcher_round.xml  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/AndroidManifest.xml  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/AppCompatPreferenceActivity.java  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/config/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/config/Config.java  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/preprocess/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/preprocess/Preprocess.java  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/MainActivity.java  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/Utils.java  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/visual/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/visual/Visualize.java  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/Predictor.java  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/java/com/baidu/paddle/lite/demo/segmentation/SettingsActivity.java  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/images/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/images/human.jpg  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/labels/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/app/src/main/assets/image_segmentation/labels/label_list  
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/
   creating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/wrapper/
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/wrapper/gradle-wrapper.jar  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle/wrapper/gradle-wrapper.properties  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradlew  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/.gitignore  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/build.gradle  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradle.properties  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/gradlew.bat  
  inflating: PaddleSeg/deploy/lite/human_segmentation_demo/settings.gradle  
  inflating: PaddleSeg/deploy/lite/README.md  
  inflating: PaddleSeg/deploy/README.md  
   creating: PaddleSeg/deploy/cpp/
   creating: PaddleSeg/deploy/cpp/tools/
  inflating: PaddleSeg/deploy/cpp/tools/visualize.py  
  inflating: PaddleSeg/deploy/cpp/CMakeLists.txt  
  inflating: PaddleSeg/deploy/cpp/LICENSE  
   creating: PaddleSeg/deploy/cpp/images/
   creating: PaddleSeg/deploy/cpp/images/humanseg/
  inflating: PaddleSeg/deploy/cpp/images/humanseg/demo3.jpeg  
  inflating: PaddleSeg/deploy/cpp/images/humanseg/demo2.jpeg  
  inflating: PaddleSeg/deploy/cpp/images/humanseg/demo2_jpeg_recover.png  
  inflating: PaddleSeg/deploy/cpp/images/humanseg/demo2.jpeg_result.png  
  inflating: PaddleSeg/deploy/cpp/images/humanseg/demo1.jpeg  
  inflating: PaddleSeg/deploy/cpp/INSTALL.md  
   creating: PaddleSeg/deploy/cpp/utils/
  inflating: PaddleSeg/deploy/cpp/utils/utils.h  
  inflating: PaddleSeg/deploy/cpp/utils/seg_conf_parser.h  
   creating: PaddleSeg/deploy/cpp/docs/
  inflating: PaddleSeg/deploy/cpp/docs/vis_result.png  
  inflating: PaddleSeg/deploy/cpp/docs/demo.jpg  
  inflating: PaddleSeg/deploy/cpp/docs/windows_vs2015_build.md  
  inflating: PaddleSeg/deploy/cpp/docs/vis.md  
  inflating: PaddleSeg/deploy/cpp/docs/demo_jpg.png  
  inflating: PaddleSeg/deploy/cpp/docs/windows_vs2019_build.md  
  inflating: PaddleSeg/deploy/cpp/docs/linux_build.md  
  inflating: PaddleSeg/deploy/cpp/README.md  
   creating: PaddleSeg/deploy/cpp/external-cmake/
  inflating: PaddleSeg/deploy/cpp/external-cmake/yaml-cpp.cmake  
  inflating: PaddleSeg/deploy/cpp/demo.cpp  
   creating: PaddleSeg/deploy/cpp/predictor/
  inflating: PaddleSeg/deploy/cpp/predictor/seg_predictor.cpp  
  inflating: PaddleSeg/deploy/cpp/predictor/seg_predictor.h  
   creating: PaddleSeg/deploy/cpp/conf/
  inflating: PaddleSeg/deploy/cpp/conf/humanseg.yaml  
   creating: PaddleSeg/deploy/cpp/preprocessor/
  inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor.cpp  
  inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor_seg.cpp  
  inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor_seg.h  
  inflating: PaddleSeg/deploy/cpp/preprocessor/preprocessor.h  
  inflating: PaddleSeg/deploy/cpp/CMakeSettings.json  
   creating: PaddleSeg/dataset/
  inflating: PaddleSeg/dataset/download_mini_deepglobe_road_extraction.py  
  inflating: PaddleSeg/dataset/download_pet.py  
  inflating: PaddleSeg/dataset/README.md  
  inflating: PaddleSeg/dataset/download_and_convert_voc2012.py  
  inflating: PaddleSeg/dataset/download_cityscapes.py  
  inflating: PaddleSeg/dataset/download_optic.py  
  inflating: PaddleSeg/dataset/convert_voc2012.py  
   creating: PaddleSeg/dygraph/
   creating: PaddleSeg/dygraph/benchmark/
  inflating: PaddleSeg/dygraph/benchmark/hrnet.py  
  inflating: PaddleSeg/dygraph/benchmark/deeplabv3p.py  
   creating: PaddleSeg/dygraph/tools/
  inflating: PaddleSeg/dygraph/tools/conver_cityscapes.py  
  inflating: PaddleSeg/dygraph/tools/voc_augment.py  
   creating: PaddleSeg/dygraph/core/
  inflating: PaddleSeg/dygraph/core/val.py  
  inflating: PaddleSeg/dygraph/core/__init__.py  
  inflating: PaddleSeg/dygraph/core/train.py  
  inflating: PaddleSeg/dygraph/core/infer.py  
   creating: PaddleSeg/dygraph/cvlibs/
  inflating: PaddleSeg/dygraph/cvlibs/__init__.py  
  inflating: PaddleSeg/dygraph/cvlibs/manager.py  
  inflating: PaddleSeg/dygraph/val.py  
   creating: PaddleSeg/dygraph/datasets/
  inflating: PaddleSeg/dygraph/datasets/cityscapes.py  
  inflating: PaddleSeg/dygraph/datasets/ade.py  
  inflating: PaddleSeg/dygraph/datasets/__init__.py  
  inflating: PaddleSeg/dygraph/datasets/dataset.py  
  inflating: PaddleSeg/dygraph/datasets/voc.py  
  inflating: PaddleSeg/dygraph/datasets/optic_disc_seg.py  
  inflating: PaddleSeg/dygraph/__init__.py  
   creating: PaddleSeg/dygraph/utils/
  inflating: PaddleSeg/dygraph/utils/metrics.py  
  inflating: PaddleSeg/dygraph/utils/timer.py  
  inflating: PaddleSeg/dygraph/utils/download.py  
  inflating: PaddleSeg/dygraph/utils/__init__.py  
  inflating: PaddleSeg/dygraph/utils/logger.py  
  inflating: PaddleSeg/dygraph/utils/utils.py  
  inflating: PaddleSeg/dygraph/utils/get_environ_info.py  
   creating: PaddleSeg/dygraph/models/
  inflating: PaddleSeg/dygraph/models/model_utils.py  
  inflating: PaddleSeg/dygraph/models/fcn.py  
  inflating: PaddleSeg/dygraph/models/deeplab.py  
  inflating: PaddleSeg/dygraph/models/unet.py  
  inflating: PaddleSeg/dygraph/models/__init__.py  
   creating: PaddleSeg/dygraph/models/architectures/
  inflating: PaddleSeg/dygraph/models/architectures/layer_utils.py  
  inflating: PaddleSeg/dygraph/models/architectures/hrnet.py  
  inflating: PaddleSeg/dygraph/models/architectures/xception_deeplab.py  
  inflating: PaddleSeg/dygraph/models/architectures/__init__.py  
  inflating: PaddleSeg/dygraph/models/architectures/mobilenetv3.py  
  inflating: PaddleSeg/dygraph/models/architectures/resnet_vd.py  
  inflating: PaddleSeg/dygraph/models/pspnet.py  
  inflating: PaddleSeg/dygraph/README.md  
   creating: PaddleSeg/dygraph/transforms/
  inflating: PaddleSeg/dygraph/transforms/transforms.py  
  inflating: PaddleSeg/dygraph/transforms/__init__.py  
  inflating: PaddleSeg/dygraph/transforms/functional.py  
  inflating: PaddleSeg/dygraph/train.py  
  inflating: PaddleSeg/dygraph/infer.py  
   creating: PaddleSeg/pdseg/
  inflating: PaddleSeg/pdseg/export_model.py  
  inflating: PaddleSeg/pdseg/metrics.py  
   creating: PaddleSeg/pdseg/tools/
  inflating: PaddleSeg/pdseg/tools/create_dataset_list.py  
  inflating: PaddleSeg/pdseg/tools/__init__.py  
  inflating: PaddleSeg/pdseg/tools/labelme2seg.py  
  inflating: PaddleSeg/pdseg/tools/jingling2seg.py  
  inflating: PaddleSeg/pdseg/tools/gray2pseudo_color.py  
  inflating: PaddleSeg/pdseg/check.py  
  inflating: PaddleSeg/pdseg/solver.py  
  inflating: PaddleSeg/pdseg/__init__.py  
   creating: PaddleSeg/pdseg/utils/
  inflating: PaddleSeg/pdseg/utils/collect.py  
  inflating: PaddleSeg/pdseg/utils/config.py  
  inflating: PaddleSeg/pdseg/utils/timer.py  
  inflating: PaddleSeg/pdseg/utils/__init__.py  
  inflating: PaddleSeg/pdseg/utils/fp16_utils.py  
  inflating: PaddleSeg/pdseg/utils/dist_utils.py  
  inflating: PaddleSeg/pdseg/utils/load_model_utils.py  
   creating: PaddleSeg/pdseg/models/
  inflating: PaddleSeg/pdseg/models/model_builder.py  
  inflating: PaddleSeg/pdseg/models/__init__.py  
   creating: PaddleSeg/pdseg/models/modeling/
  inflating: PaddleSeg/pdseg/models/modeling/hrnet.py  
  inflating: PaddleSeg/pdseg/models/modeling/deeplab.py  
  inflating: PaddleSeg/pdseg/models/modeling/unet.py  
  inflating: PaddleSeg/pdseg/models/modeling/icnet.py  
  inflating: PaddleSeg/pdseg/models/modeling/__init__.py  
  inflating: PaddleSeg/pdseg/models/modeling/fast_scnn.py  
  inflating: PaddleSeg/pdseg/models/modeling/pspnet.py  
  inflating: PaddleSeg/pdseg/models/modeling/ocrnet.py  
   creating: PaddleSeg/pdseg/models/libs/
  inflating: PaddleSeg/pdseg/models/libs/__init__.py  
  inflating: PaddleSeg/pdseg/models/libs/model_libs.py  
   creating: PaddleSeg/pdseg/models/backbone/
  inflating: PaddleSeg/pdseg/models/backbone/vgg.py  
  inflating: PaddleSeg/pdseg/models/backbone/mobilenet_v2.py  
  inflating: PaddleSeg/pdseg/models/backbone/__init__.py  
  inflating: PaddleSeg/pdseg/models/backbone/mobilenet_v3.py  
  inflating: PaddleSeg/pdseg/models/backbone/resnet.py  
  inflating: PaddleSeg/pdseg/models/backbone/resnet_vd.py  
  inflating: PaddleSeg/pdseg/models/backbone/xception.py  
  inflating: PaddleSeg/pdseg/vis.py  
  inflating: PaddleSeg/pdseg/reader.py  
  inflating: PaddleSeg/pdseg/loss.py  
  inflating: PaddleSeg/pdseg/data_utils.py  
  inflating: PaddleSeg/pdseg/data_aug.py  
  inflating: PaddleSeg/pdseg/train.py  
  inflating: PaddleSeg/pdseg/eval.py  
  inflating: PaddleSeg/pdseg/lovasz_losses.py  
   creating: PaddleSeg/docs/
  inflating: PaddleSeg/docs/data_aug.md  
   creating: PaddleSeg/docs/imgs/
  inflating: PaddleSeg/docs/imgs/deepglobe.png  
  inflating: PaddleSeg/docs/imgs/usage_vis_demo.jpg  
  inflating: PaddleSeg/docs/imgs/fast-scnn.png  
  inflating: PaddleSeg/docs/imgs/lovasz-hinge.png  
  inflating: PaddleSeg/docs/imgs/loss_comparison.png  
  inflating: PaddleSeg/docs/imgs/lovasz-softmax.png  
  inflating: PaddleSeg/docs/imgs/rangescale.png  
  inflating: PaddleSeg/docs/imgs/qq_group2.png  
  inflating: PaddleSeg/docs/imgs/pspnet2.png  
  inflating: PaddleSeg/docs/imgs/hrnet.png  
  inflating: PaddleSeg/docs/imgs/lovasz-hinge-vis.png  
  inflating: PaddleSeg/docs/imgs/file_list.png  
  inflating: PaddleSeg/docs/imgs/unet.png  
   creating: PaddleSeg/docs/imgs/annotation/
  inflating: PaddleSeg/docs/imgs/annotation/image-4-1.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-6-2.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-4-2.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-10.jpg  
  inflating: PaddleSeg/docs/imgs/annotation/image-11.png  
  inflating: PaddleSeg/docs/imgs/annotation/jingling-5.png  
  inflating: PaddleSeg/docs/imgs/annotation/jingling-4.png  
  inflating: PaddleSeg/docs/imgs/annotation/jingling-1.png  
  inflating: PaddleSeg/docs/imgs/annotation/jingling-3.png  
  inflating: PaddleSeg/docs/imgs/annotation/jingling-2.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-1.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-3.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-2.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-6.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-7.png  
  inflating: PaddleSeg/docs/imgs/annotation/image-5.png  
  inflating: PaddleSeg/docs/imgs/file_list2.png  
  inflating: PaddleSeg/docs/imgs/warmup_with_poly_decay_example.png  
  inflating: PaddleSeg/docs/imgs/data_aug_example.png  
  inflating: PaddleSeg/docs/imgs/dice3.png  
  inflating: PaddleSeg/docs/imgs/dice2.png  
  inflating: PaddleSeg/docs/imgs/softmax_loss.png  
  inflating: PaddleSeg/docs/imgs/gn.png  
  inflating: PaddleSeg/docs/imgs/dice.png  
  inflating: PaddleSeg/docs/imgs/pspnet.png  
  inflating: PaddleSeg/docs/imgs/icnet.png  
  inflating: PaddleSeg/docs/imgs/visualdl_scalar.png  
  inflating: PaddleSeg/docs/imgs/aug_method.png  
  inflating: PaddleSeg/docs/imgs/data_aug_flip_mirror.png  
  inflating: PaddleSeg/docs/imgs/poly_decay_example.png  
  inflating: PaddleSeg/docs/imgs/piecewise_decay_example.png  
  inflating: PaddleSeg/docs/imgs/VOC2012.png  
  inflating: PaddleSeg/docs/imgs/visualdl_image.png  
  inflating: PaddleSeg/docs/imgs/cosine_decay_example.png  
  inflating: PaddleSeg/docs/imgs/data_aug_flow.png  
  inflating: PaddleSeg/docs/imgs/deeplabv3p.png  
  inflating: PaddleSeg/docs/usage.md  
  inflating: PaddleSeg/docs/lovasz_loss.md  
  inflating: PaddleSeg/docs/data_prepare.md  
   creating: PaddleSeg/docs/annotation/
  inflating: PaddleSeg/docs/annotation/labelme2seg.md  
   creating: PaddleSeg/docs/annotation/cityscapes_demo/
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/cityscapes_demo_dataset.yaml  
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/train_list.txt  
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/val_list.txt  
   creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/
   creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/
   creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/stuttgart/
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/stuttgart/stuttgart_000021_000019_gtFine_labelTrainIds.png  
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/train/stuttgart/stuttgart_000072_000019_gtFine_labelTrainIds.png  
   creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/
   creating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/frankfurt/
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/frankfurt/frankfurt_000001_063045_gtFine_labelTrainIds.png  
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/gtFine/val/frankfurt/frankfurt_000001_062250_gtFine_labelTrainIds.png  
   creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/
   creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/
   creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/stuttgart/
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/stuttgart/stuttgart_000072_000019_leftImg8bit.png  
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/train/stuttgart/stuttgart_000021_000019_leftImg8bit.png  
   creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/
   creating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/frankfurt/
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/frankfurt/frankfurt_000001_062250_leftImg8bit.png  
  inflating: PaddleSeg/docs/annotation/cityscapes_demo/leftImg8bit/val/frankfurt/frankfurt_000001_063045_leftImg8bit.png  
  inflating: PaddleSeg/docs/annotation/jingling2seg.md  
   creating: PaddleSeg/docs/annotation/jingling_demo/
  inflating: PaddleSeg/docs/annotation/jingling_demo/jingling.jpg  
   creating: PaddleSeg/docs/annotation/jingling_demo/outputs/
   creating: PaddleSeg/docs/annotation/jingling_demo/outputs/annotations/
  inflating: PaddleSeg/docs/annotation/jingling_demo/outputs/annotations/jingling.png  
  inflating: PaddleSeg/docs/annotation/jingling_demo/outputs/jingling.json  
  inflating: PaddleSeg/docs/annotation/jingling_demo/outputs/class_names.txt  
   creating: PaddleSeg/docs/annotation/labelme_demo/
  inflating: PaddleSeg/docs/annotation/labelme_demo/2011_000025.jpg  
  inflating: PaddleSeg/docs/annotation/labelme_demo/2011_000025.json  
  inflating: PaddleSeg/docs/annotation/labelme_demo/class_names.txt  
  inflating: PaddleSeg/docs/check.md  
  inflating: PaddleSeg/docs/dice_loss.md  
  inflating: PaddleSeg/docs/config.md  
  inflating: PaddleSeg/docs/deploy.md  
  inflating: PaddleSeg/docs/models.md  
   creating: PaddleSeg/docs/configs/
  inflating: PaddleSeg/docs/configs/model_hrnet_group.md  
  inflating: PaddleSeg/docs/configs/model_pspnet_group.md  
  inflating: PaddleSeg/docs/configs/model_deeplabv3p_group.md  
  inflating: PaddleSeg/docs/configs/.gitkeep  
  inflating: PaddleSeg/docs/configs/model_unet_group.md  
  inflating: PaddleSeg/docs/configs/model_group.md  
  inflating: PaddleSeg/docs/configs/test_group.md  
  inflating: PaddleSeg/docs/configs/train_group.md  
  inflating: PaddleSeg/docs/configs/dataloader_group.md  
  inflating: PaddleSeg/docs/configs/model_icnet_group.md  
  inflating: PaddleSeg/docs/configs/freeze_group.md  
  inflating: PaddleSeg/docs/configs/solver_group.md  
  inflating: PaddleSeg/docs/configs/basic_group.md  
  inflating: PaddleSeg/docs/configs/dataset_group.md  
  inflating: PaddleSeg/docs/multiple_gpus_train_and_mixed_precision_train.md  
  inflating: PaddleSeg/docs/model_export.md  
  inflating: PaddleSeg/docs/model_zoo.md  
  inflating: PaddleSeg/docs/loss_select.md  
   creating: PaddleSeg/contrib/
   creating: PaddleSeg/contrib/MechanicalIndustryMeter/
   creating: PaddleSeg/contrib/MechanicalIndustryMeter/imgs/
  inflating: PaddleSeg/contrib/MechanicalIndustryMeter/imgs/1560143028.5_IMG_3091.png  
  inflating: PaddleSeg/contrib/MechanicalIndustryMeter/imgs/1560143028.5_IMG_3091.JPG  
  inflating: PaddleSeg/contrib/MechanicalIndustryMeter/unet_mechanical_meter.yaml  
  inflating: PaddleSeg/contrib/MechanicalIndustryMeter/download_unet_mechanical_industry_meter.py  
  inflating: PaddleSeg/contrib/MechanicalIndustryMeter/download_mini_mechanical_industry_meter.py  
   creating: PaddleSeg/contrib/SpatialEmbeddings/
  inflating: PaddleSeg/contrib/SpatialEmbeddings/config.py  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/models.py  
   creating: PaddleSeg/contrib/SpatialEmbeddings/imgs/
  inflating: PaddleSeg/contrib/SpatialEmbeddings/imgs/kitti_0007_000518_ori.png  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/imgs/kitti_0007_000518_pred.png  
   creating: PaddleSeg/contrib/SpatialEmbeddings/utils/
  inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/util.py  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/data_util.py  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/__init__.py  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/utils/palette.py  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/download_SpatialEmbeddings_kitti.py  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/README.md  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/infer.py  
   creating: PaddleSeg/contrib/SpatialEmbeddings/data/
   creating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/
   creating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/0007/
  inflating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/0007/kitti_0007_000518.png  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/data/kitti/0007/kitti_0007_000512.png  
  inflating: PaddleSeg/contrib/SpatialEmbeddings/data/test.txt  
   creating: PaddleSeg/contrib/RoadLine/
  inflating: PaddleSeg/contrib/RoadLine/config.py  
   creating: PaddleSeg/contrib/RoadLine/imgs/
  inflating: PaddleSeg/contrib/RoadLine/imgs/RoadLine.jpg  
  inflating: PaddleSeg/contrib/RoadLine/imgs/RoadLine.png  
  inflating: PaddleSeg/contrib/RoadLine/__init__.py  
   creating: PaddleSeg/contrib/RoadLine/utils/
  inflating: PaddleSeg/contrib/RoadLine/utils/util.py  
  inflating: PaddleSeg/contrib/RoadLine/utils/__init__.py  
  inflating: PaddleSeg/contrib/RoadLine/utils/palette.py  
  inflating: PaddleSeg/contrib/RoadLine/infer.py  
  inflating: PaddleSeg/contrib/RoadLine/download_RoadLine.py  
  inflating: PaddleSeg/contrib/README.md  
   creating: PaddleSeg/contrib/LaneNet/
  inflating: PaddleSeg/contrib/LaneNet/requirements.txt  
   creating: PaddleSeg/contrib/LaneNet/imgs/
  inflating: PaddleSeg/contrib/LaneNet/imgs/0005_pred_lane.png  
  inflating: PaddleSeg/contrib/LaneNet/imgs/0005_pred_binary.png  
  inflating: PaddleSeg/contrib/LaneNet/imgs/0005_pred_instance.png  
   creating: PaddleSeg/contrib/LaneNet/dataset/
  inflating: PaddleSeg/contrib/LaneNet/dataset/download_tusimple.py  
   creating: PaddleSeg/contrib/LaneNet/utils/
  inflating: PaddleSeg/contrib/LaneNet/utils/config.py  
  inflating: PaddleSeg/contrib/LaneNet/utils/__init__.py  
  inflating: PaddleSeg/contrib/LaneNet/utils/lanenet_postprocess.py  
  inflating: PaddleSeg/contrib/LaneNet/utils/generate_tusimple_dataset.py  
  inflating: PaddleSeg/contrib/LaneNet/utils/dist_utils.py  
  inflating: PaddleSeg/contrib/LaneNet/utils/load_model_utils.py  
   creating: PaddleSeg/contrib/LaneNet/models/
  inflating: PaddleSeg/contrib/LaneNet/models/model_builder.py  
  inflating: PaddleSeg/contrib/LaneNet/models/__init__.py  
   creating: PaddleSeg/contrib/LaneNet/models/modeling/
  inflating: PaddleSeg/contrib/LaneNet/models/modeling/__init__.py  
  inflating: PaddleSeg/contrib/LaneNet/models/modeling/lanenet.py  
  inflating: PaddleSeg/contrib/LaneNet/vis.py  
  inflating: PaddleSeg/contrib/LaneNet/README.md  
  inflating: PaddleSeg/contrib/LaneNet/reader.py  
  inflating: PaddleSeg/contrib/LaneNet/loss.py  
  inflating: PaddleSeg/contrib/LaneNet/data_aug.py  
   creating: PaddleSeg/contrib/LaneNet/configs/
  inflating: PaddleSeg/contrib/LaneNet/configs/lanenet.yaml  
  inflating: PaddleSeg/contrib/LaneNet/train.py  
  inflating: PaddleSeg/contrib/LaneNet/eval.py  
   creating: PaddleSeg/contrib/ACE2P/
  inflating: PaddleSeg/contrib/ACE2P/config.py  
   creating: PaddleSeg/contrib/ACE2P/imgs/
  inflating: PaddleSeg/contrib/ACE2P/imgs/result.jpg  
  inflating: PaddleSeg/contrib/ACE2P/imgs/net.jpg  
  inflating: PaddleSeg/contrib/ACE2P/imgs/117676_2149260.jpg  
  inflating: PaddleSeg/contrib/ACE2P/imgs/117676_2149260.png  
  inflating: PaddleSeg/contrib/ACE2P/download_ACE2P.py  
  inflating: PaddleSeg/contrib/ACE2P/__init__.py  
   creating: PaddleSeg/contrib/ACE2P/utils/
  inflating: PaddleSeg/contrib/ACE2P/utils/util.py  
  inflating: PaddleSeg/contrib/ACE2P/utils/__init__.py  
  inflating: PaddleSeg/contrib/ACE2P/utils/palette.py  
  inflating: PaddleSeg/contrib/ACE2P/README.md  
  inflating: PaddleSeg/contrib/ACE2P/reader.py  
  inflating: PaddleSeg/contrib/ACE2P/infer.py  
   creating: PaddleSeg/contrib/HumanSeg/
  inflating: PaddleSeg/contrib/HumanSeg/video_infer.py  
  inflating: PaddleSeg/contrib/HumanSeg/quant_online.py  
  inflating: PaddleSeg/contrib/HumanSeg/requirements.txt  
  inflating: PaddleSeg/contrib/HumanSeg/val.py  
   creating: PaddleSeg/contrib/HumanSeg/datasets/
  inflating: PaddleSeg/contrib/HumanSeg/datasets/__init__.py  
  inflating: PaddleSeg/contrib/HumanSeg/datasets/dataset.py  
   creating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/
  inflating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/sharedmemory.py  
  inflating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/queue.py  
  inflating: PaddleSeg/contrib/HumanSeg/datasets/shared_queue/__init__.py  
  inflating: PaddleSeg/contrib/HumanSeg/bg_replace.py  
  inflating: PaddleSeg/contrib/HumanSeg/quant_offline.py  
   creating: PaddleSeg/contrib/HumanSeg/utils/
  inflating: PaddleSeg/contrib/HumanSeg/utils/logging.py  
  inflating: PaddleSeg/contrib/HumanSeg/utils/metrics.py  
  inflating: PaddleSeg/contrib/HumanSeg/utils/__init__.py  
  inflating: PaddleSeg/contrib/HumanSeg/utils/humanseg_postprocess.py  
  inflating: PaddleSeg/contrib/HumanSeg/utils/utils.py  
  inflating: PaddleSeg/contrib/HumanSeg/utils/post_quantization.py  
   creating: PaddleSeg/contrib/HumanSeg/models/
  inflating: PaddleSeg/contrib/HumanSeg/models/load_model.py  
  inflating: PaddleSeg/contrib/HumanSeg/models/__init__.py  
  inflating: PaddleSeg/contrib/HumanSeg/models/humanseg.py  
  inflating: PaddleSeg/contrib/HumanSeg/export.py  
  inflating: PaddleSeg/contrib/HumanSeg/README.md  
   creating: PaddleSeg/contrib/HumanSeg/nets/
  inflating: PaddleSeg/contrib/HumanSeg/nets/hrnet.py  
  inflating: PaddleSeg/contrib/HumanSeg/nets/shufflenet_slim.py  
  inflating: PaddleSeg/contrib/HumanSeg/nets/seg_modules.py  
  inflating: PaddleSeg/contrib/HumanSeg/nets/__init__.py  
  inflating: PaddleSeg/contrib/HumanSeg/nets/deeplabv3p.py  
   creating: PaddleSeg/contrib/HumanSeg/nets/backbone/
  inflating: PaddleSeg/contrib/HumanSeg/nets/backbone/mobilenet_v2.py  
  inflating: PaddleSeg/contrib/HumanSeg/nets/backbone/__init__.py  
  inflating: PaddleSeg/contrib/HumanSeg/nets/backbone/xception.py  
  inflating: PaddleSeg/contrib/HumanSeg/nets/libs.py  
   creating: PaddleSeg/contrib/HumanSeg/transforms/
  inflating: PaddleSeg/contrib/HumanSeg/transforms/transforms.py  
  inflating: PaddleSeg/contrib/HumanSeg/transforms/__init__.py  
  inflating: PaddleSeg/contrib/HumanSeg/transforms/functional.py  
  inflating: PaddleSeg/contrib/HumanSeg/train.py  
  inflating: PaddleSeg/contrib/HumanSeg/infer.py  
   creating: PaddleSeg/contrib/HumanSeg/data/
  inflating: PaddleSeg/contrib/HumanSeg/data/background.jpg  
  inflating: PaddleSeg/contrib/HumanSeg/data/download_data.py  
  inflating: PaddleSeg/contrib/HumanSeg/data/human_image.jpg  
   creating: PaddleSeg/contrib/HumanSeg/pretrained_weights/
  inflating: PaddleSeg/contrib/HumanSeg/pretrained_weights/download_pretrained_weights.py  
   creating: PaddleSeg/contrib/RemoteSensing/
  inflating: PaddleSeg/contrib/RemoteSensing/visualize_demo.py  
   creating: PaddleSeg/contrib/RemoteSensing/tools/
  inflating: PaddleSeg/contrib/RemoteSensing/tools/data_analyse_and_check.py  
  inflating: PaddleSeg/contrib/RemoteSensing/tools/cal_norm_coef.py  
  inflating: PaddleSeg/contrib/RemoteSensing/tools/data_distribution_vis.py  
  inflating: PaddleSeg/contrib/RemoteSensing/tools/create_dataset_list.py  
  inflating: PaddleSeg/contrib/RemoteSensing/tools/split_dataset_list.py  
  inflating: PaddleSeg/contrib/RemoteSensing/requirements.txt  
  inflating: PaddleSeg/contrib/RemoteSensing/__init__.py  
   creating: PaddleSeg/contrib/RemoteSensing/utils/
  inflating: PaddleSeg/contrib/RemoteSensing/utils/logging.py  
  inflating: PaddleSeg/contrib/RemoteSensing/utils/metrics.py  
  inflating: PaddleSeg/contrib/RemoteSensing/utils/pretrain_weights.py  
  inflating: PaddleSeg/contrib/RemoteSensing/utils/__init__.py  
  inflating: PaddleSeg/contrib/RemoteSensing/utils/utils.py  
   creating: PaddleSeg/contrib/RemoteSensing/models/
  inflating: PaddleSeg/contrib/RemoteSensing/models/load_model.py  
  inflating: PaddleSeg/contrib/RemoteSensing/models/hrnet.py  
  inflating: PaddleSeg/contrib/RemoteSensing/models/unet.py  
  inflating: PaddleSeg/contrib/RemoteSensing/models/__init__.py  
   creating: PaddleSeg/contrib/RemoteSensing/models/utils/
  inflating: PaddleSeg/contrib/RemoteSensing/models/utils/visualize.py  
  inflating: PaddleSeg/contrib/RemoteSensing/models/base.py  
   creating: PaddleSeg/contrib/RemoteSensing/docs/
   creating: PaddleSeg/contrib/RemoteSensing/docs/imgs/
  inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/dataset.png  
  inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/visualdl.png  
  inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/data_distribution.png  
  inflating: PaddleSeg/contrib/RemoteSensing/docs/imgs/vis.png  
  inflating: PaddleSeg/contrib/RemoteSensing/docs/data_prepare.md  
  inflating: PaddleSeg/contrib/RemoteSensing/docs/transforms.md  
  inflating: PaddleSeg/contrib/RemoteSensing/docs/data_analyse_and_check.md  
  inflating: PaddleSeg/contrib/RemoteSensing/README.md  
   creating: PaddleSeg/contrib/RemoteSensing/nets/
  inflating: PaddleSeg/contrib/RemoteSensing/nets/hrnet.py  
  inflating: PaddleSeg/contrib/RemoteSensing/nets/unet.py  
  inflating: PaddleSeg/contrib/RemoteSensing/nets/__init__.py  
  inflating: PaddleSeg/contrib/RemoteSensing/nets/loss.py  
  inflating: PaddleSeg/contrib/RemoteSensing/nets/libs.py  
   creating: PaddleSeg/contrib/RemoteSensing/transforms/
  inflating: PaddleSeg/contrib/RemoteSensing/transforms/transforms.py  
  inflating: PaddleSeg/contrib/RemoteSensing/transforms/__init__.py  
  inflating: PaddleSeg/contrib/RemoteSensing/transforms/ops.py  
  inflating: PaddleSeg/contrib/RemoteSensing/predict_demo.py  
   creating: PaddleSeg/contrib/RemoteSensing/readers/
  inflating: PaddleSeg/contrib/RemoteSensing/readers/__init__.py  
  inflating: PaddleSeg/contrib/RemoteSensing/readers/reader.py  
  inflating: PaddleSeg/contrib/RemoteSensing/readers/base.py  
  inflating: PaddleSeg/contrib/RemoteSensing/train_demo.py  
  inflating: PaddleSeg/README.md     
  inflating: PaddleSeg/.gitignore    
  inflating: PaddleSeg/.style.yapf   
   creating: PaddleSeg/configs/
  inflating: PaddleSeg/configs/lovasz_hinge_deeplabv3p_mobilenet_road.yaml  
  inflating: PaddleSeg/configs/cityscape_fast_scnn.yaml  
  inflating: PaddleSeg/configs/unet_optic.yaml  
  inflating: PaddleSeg/configs/deepglobe_road_extraction.yaml  
  inflating: PaddleSeg/configs/hrnet_optic.yaml  
  inflating: PaddleSeg/configs/ocrnet_w18_bn_cityscapes.yaml  
  inflating: PaddleSeg/configs/deeplabv3p_mobilenetv3_large_cityscapes.yaml  
  inflating: PaddleSeg/configs/icnet_optic.yaml  
  inflating: PaddleSeg/configs/deeplabv3p_resnet50_vd_cityscapes.yaml  
  inflating: PaddleSeg/configs/deeplabv3p_xception65_optic.yaml  
  inflating: PaddleSeg/configs/deeplabv3p_xception65_cityscapes.yaml  
  inflating: PaddleSeg/configs/deeplabv3p_mobilenet-1-0_pet.yaml  
  inflating: PaddleSeg/configs/fast_scnn_pet.yaml  
  inflating: PaddleSeg/configs/lovasz_softmax_deeplabv3p_mobilenet_pascal.yaml  
  inflating: PaddleSeg/configs/deeplabv3p_mobilenetv2_cityscapes.yaml  
  inflating: PaddleSeg/configs/pspnet_optic.yaml  
   creating: PaddleSeg/slim/
   creating: PaddleSeg/slim/nas/
  inflating: PaddleSeg/slim/nas/train_nas.py  
  inflating: PaddleSeg/slim/nas/model_builder.py  
  inflating: PaddleSeg/slim/nas/mobilenetv2_search_space.py  
  inflating: PaddleSeg/slim/nas/deeplab.py  
  inflating: PaddleSeg/slim/nas/README.md  
  inflating: PaddleSeg/slim/nas/eval_nas.py  
   creating: PaddleSeg/slim/distillation/
  inflating: PaddleSeg/slim/distillation/model_builder.py  
  inflating: PaddleSeg/slim/distillation/README.md  
  inflating: PaddleSeg/slim/distillation/train_distill.py  
  inflating: PaddleSeg/slim/distillation/cityscape_teacher.yaml  
  inflating: PaddleSeg/slim/distillation/cityscape.yaml  
   creating: PaddleSeg/slim/quantization/
  inflating: PaddleSeg/slim/quantization/export_model.py  
   creating: PaddleSeg/slim/quantization/images/
  inflating: PaddleSeg/slim/quantization/images/TransformPass.png  
  inflating: PaddleSeg/slim/quantization/images/ConvertToInt8Pass.png  
  inflating: PaddleSeg/slim/quantization/images/FreezePass.png  
  inflating: PaddleSeg/slim/quantization/images/TransformForMobilePass.png  
  inflating: PaddleSeg/slim/quantization/train_quant.py  
  inflating: PaddleSeg/slim/quantization/eval_quant.py  
  inflating: PaddleSeg/slim/quantization/README.md  
   creating: PaddleSeg/slim/prune/
  inflating: PaddleSeg/slim/prune/train_prune.py  
  inflating: PaddleSeg/slim/prune/README.md  
  inflating: PaddleSeg/slim/prune/eval_prune.py  
   creating: PaddleSeg/.git/
  inflating: PaddleSeg/.git/config   
   creating: PaddleSeg/.git/objects/
   creating: PaddleSeg/.git/objects/pack/
  inflating: PaddleSeg/.git/objects/pack/pack-0b5aa212c914b01e77066dc5729dffedd245ed42.pack  
  inflating: PaddleSeg/.git/objects/pack/pack-0b5aa212c914b01e77066dc5729dffedd245ed42.idx  
   creating: PaddleSeg/.git/objects/info/
  inflating: PaddleSeg/.git/HEAD     
   creating: PaddleSeg/.git/info/
  inflating: PaddleSeg/.git/info/exclude  
   creating: PaddleSeg/.git/logs/
  inflating: PaddleSeg/.git/logs/HEAD  
   creating: PaddleSeg/.git/logs/refs/
   creating: PaddleSeg/.git/logs/refs/heads/
   creating: PaddleSeg/.git/logs/refs/heads/release/
  inflating: PaddleSeg/.git/logs/refs/heads/release/v0.6.0  
   creating: PaddleSeg/.git/logs/refs/remotes/
   creating: PaddleSeg/.git/logs/refs/remotes/origin/
  inflating: PaddleSeg/.git/logs/refs/remotes/origin/HEAD  
  inflating: PaddleSeg/.git/description  
   creating: PaddleSeg/.git/hooks/
  inflating: PaddleSeg/.git/hooks/commit-msg.sample  
  inflating: PaddleSeg/.git/hooks/pre-rebase.sample  
  inflating: PaddleSeg/.git/hooks/pre-commit.sample  
  inflating: PaddleSeg/.git/hooks/applypatch-msg.sample  
  inflating: PaddleSeg/.git/hooks/fsmonitor-watchman.sample  
  inflating: PaddleSeg/.git/hooks/pre-receive.sample  
  inflating: PaddleSeg/.git/hooks/prepare-commit-msg.sample  
  inflating: PaddleSeg/.git/hooks/post-update.sample  
  inflating: PaddleSeg/.git/hooks/pre-applypatch.sample  
  inflating: PaddleSeg/.git/hooks/pre-push.sample  
  inflating: PaddleSeg/.git/hooks/update.sample  
   creating: PaddleSeg/.git/refs/
   creating: PaddleSeg/.git/refs/heads/
   creating: PaddleSeg/.git/refs/heads/release/
  inflating: PaddleSeg/.git/refs/heads/release/v0.6.0  
   creating: PaddleSeg/.git/refs/tags/
   creating: PaddleSeg/.git/refs/remotes/
   creating: PaddleSeg/.git/refs/remotes/origin/
  inflating: PaddleSeg/.git/refs/remotes/origin/HEAD  
  inflating: PaddleSeg/.git/index    
  inflating: PaddleSeg/.git/packed-refs  
  inflating: PaddleSeg/.travis.yml   
/home/aistudio/PaddleSeg
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (1.21.0)
Requirement already satisfied: yapf==0.26.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (0.26.0)
Requirement already satisfied: flake8 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 3)) (4.0.1)
Requirement already satisfied: pyyaml>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (5.1.2)
Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (2.2.0)
Requirement already satisfied: importlib-metadata in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (4.2.0)
Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (0.10.0)
Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.4.10)
Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.3.4)
Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (2.0.1)
Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (16.7.9)
Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r requirements.txt (line 1)) (1.16.0)
Requirement already satisfied: pyflakes<2.5.0,>=2.4.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (2.4.0)
Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (0.6.1)
Requirement already satisfied: pycodestyle<2.9.0,>=2.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r requirements.txt (line 3)) (2.8.0)
Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.0.0)
Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (3.14.0)
Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.19.5)
Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.1)
Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (2.24.0)
Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.5)
Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (8.2.0)
Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (2.2.3)
Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (0.7.1.1)
Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r requirements.txt (line 5)) (0.8.53)
Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.0)
Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (7.0)
Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.11.0)
Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.16.0)
Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r requirements.txt (line 5)) (2019.3)
Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8.0)
Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata->pre-commit->-r requirements.txt (line 1)) (3.7.0)
Requirement already satisfied: typing-extensions>=3.6.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata->pre-commit->-r requirements.txt (line 1)) (4.0.1)
Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.18.0)
Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.9.9)
Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8.2)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.1.0)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.0.7)
Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (0.10.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (1.25.6)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (2019.9.11)
Requirement already satisfied: idna<3,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.8)
Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0->-r requirements.txt (line 5)) (2.0.1)
Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl>=2.0.0->-r requirements.txt (line 5)) (56.2.0)
WARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
You should consider upgrading via the '/opt/conda/envs/python35-paddle120-env/bin/python -m pip install --upgrade pip' command.
4、配置超参数并训练模型
# 数据集配置
DATASET:
    DATA_DIR: "/home/aistudio/"
    NUM_CLASSES: 2
    TEST_FILE_LIST: "/home/aistudio/val_list.txt"
    TRAIN_FILE_LIST: "/home/aistudio/train_list.txt"
    VAL_FILE_LIST: "/home/aistudio/val_list.txt"
    VIS_FILE_LIST: "/home/aistudio/val_list.txt"

# 预训练模型配置
MODEL:
    MODEL_NAME: "unet"
    DEFAULT_NORM_TYPE: "bn"

# 其他配置
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


五、总结与升华
本项目基于PaddleSeg对该数据集进行了简单的处理，用以分割出胸部X光照片中的肺部。因自身实力欠缺，多用借鉴，设计称不上完善，项目仍有许多需要改进的地方，如肺部形状图像分割出后对原图像进行读取和分割。

六、个人总结
东北大学秦皇岛分校本一在读

初次接触相关内容难免有瑕疵

个人主页链接：

https://aistudio.baidu.com/aistudio/personalcenter/thirdview/1998037
