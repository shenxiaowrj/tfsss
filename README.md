# tfsss
我对于transformers相关的一系列学习的笔记
transformers 变形金刚 或者说 变压器可太火了

所以我决定对其源码进行细致的学习和复现
并且对其之后的一系列的相关模型

如 swin transformer

  vision transformer
  
  convNext等进行复现
  
同时还包括一系列transformer的相关应用。

My notes on a series of studies related to transformers
transformers Transformers or transformers are too hot

So I decided to study and reproduce its source code in detail.
And a series of related models after it

such as swin transformer

   vision transformer
   
   convNext, etc. to reproduce
   
It also includes a series of related applications of transformers.



我观看的transformer相关的视频up，以及相关的代码参考有：

跟李沐学AI，

蓝斯诺特，

deep_thoughts 深度学习每日摘要

在此 特别感谢三位大佬的无私奉献和精彩讲解。




更新日志：

2022-04-05 星期二 /Transformer_Example-main/

2022-04-06 星期三 /transformer难点理解与实现

                         /notes
                                 /1,2,3,4,5,6.7.9

                  /Transformer_Example-main
                  
                         /source_functional.py,source_activation.py
                  
2022-04-07 星期四 /transformer难点理解与实现 
                          /notes/8

2022-04-07 星期四 /vision_transformer/VIT难点实现

                                                /1模型架构

                                                /2两种不同的img2patch的实现
                                                  
                                                /3剩下的步骤 上半部分

2022-04-08 星期五 /vision_transformer/VIT难点实现

                                                /3剩下的步骤 下半部分

2022-04-10 星期天 sota模型 实现框架
    视频及参考代码来自于：up:布尔艺数

2022-04-15 星期五 /vision_transformer/vision_tranformer 模型具体实现 并用于训练一个手写数字识别的数据集

2022-04-30 星期六 /swin_trnasformer重建-副本

                                           /swin简洁版

                                           /swin笔记版

                                           /关系矩阵和mask

    (断断徐徐写了一个多星期，下回要分函数写 都写在一块儿 太乱了)
    
2022-05-01~2022-05-04

    看了deedeep_thoughts讲的两个mae的视频
    
    然后完成了源码实现
    
    但是转换尺度 然后对齐的部分没有看懂 很烦烦 而且还没有实际的训练模型
    
    /deep_thoughts  /mae  
    
                          /note
    
                          /argsort2argsort.py
                              
                          /note.md
                              
                          /engine_finetune.py
                          
                          /engine_pretrain.py
                          
                          /main_finetune.py
                          
                          /mian_finetune.py
                          
  提取出未被mask的patch，进入vit模型，将被mask用一个统一的embedding token进行代替 将未被mask的patch和被mask掉的patch还原回原来的顺序的操作没有搞明白
  
  2022-05-07 ~ 2022-05-08
  
  deep_thoughts 52、Excel/Csv文件数据转成PyTorch张量导入模型代码逐行讲解
  
  /deep_thoughts  
  
                  /excel_csv2tensor_topredict
  
                  /csv_exel简洁版.py
  
  2022-05-13 ~ 2022-05-16
  deepthoughts 51、基于PyTorch ResNet18的果蔬分类逐行代码讲解
  
  这个项目使用了mae模型中的库 和 timm库，可以实现标准化训练和输出，可以作为一个模板来使用，适用其它任务，同时我又将其按功能划分为几个部分，分别为main_train_test transforms
  
  backbone engine info args.
  
  同时，这个小项目里还有俩个好用的工具，
  
  split_dataset 划分数据集
  
  stastic_mean_std 对一个图片数据集，求均值和方差，依次用作训练过程中的标准化。
  
  /deep_thoughts   /fruit_classification_based_maecode    
  
                                                          /split_dataset简洁版.py
  
                                                          /statistic_mean_std简洁版.py
                                                          
                                                          /train实用版.py
                                                          
  2022-05-17 ~ 2022-05-21
  deep_thoughts   /46_four_position_embedding     
                                                  
                                                  /简洁版
                                                         /transformer position embedding
                                                         
                                                         /vision transformer position embedding 
                                                         
                                                         /swin transformer posoition embedding
                                                         
                                                         /mae transformer position embedding
                                                         
                                                  /笔记版
                                                         /transformer position embedding
                                                         
                                                         /vision transformer position embedding 
                                                         
                                                         /swin transformer posoition embedding
                                                         
                                                         /mae transformer position embedding
                                        
                                                        
                                                        



                                                 
