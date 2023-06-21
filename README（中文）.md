# Anycost GAN（交互式图像编辑）

### [video](https://youtu.be/_yEziPl9AkM) | [paper](https://arxiv.org/abs/2103.03243) | [website](https://hanlab.mit.edu/projects/anycost-gan/) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mit-han-lab/anycost-gan/blob/master/notebooks/intro_colab.ipynb)

[Anycost GANs for Interactive Image Synthesis and Editing](https://arxiv.org/abs/2103.03243)

[Ji Lin](http://linji.me/), [Richard Zhang](https://richzhang.github.io/), Frieder Ganz, [Song Han](https://songhan.mit.edu/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)

MIT, Adobe Research, CMU

In CVPR 2021

![flexible](https://hanlab.mit.edu/projects/anycost-gan/images/flexible.gif)

交互式图像编辑GAN在不同的计算预算下产生一致的输出。



## 演示

<a href="https://youtu.be/_yEziPl9AkM?t=90"><img src='assets/figures/demo.gif' width=600></a>

在这里，我们可以使用Anycost生成器进行交互式图像编辑。一个完整的生成器需要**~3s**来渲染图像，这对于编辑来说太慢了。而使用Anycost生成器，我们可以以**5倍快的速度提供视觉上类似的预览**。调整后，我们点击“定型”按钮合成高质量的最终输出。 查看 [这里](https://youtu.be/_yEziPl9AkM?t=90) 获得完整的演示。





## 概述

通过使用不同的通道和分辨率配置，Anycost 生成器都可以以不同的计算成本运行。与完整的生成器相比，子生成器实现了高输出一致性，提供了快速预览。

![overview](https://hanlab.mit.edu/projects/anycost-gan/images/overall.jpg)



通过(1)基于采样的多分辨率训练，(2)自适应通道训练，(3)生成器条件鉴别器，我们在不同分辨率和通道下获得了高质量的图像质量和一致性。![method](https://hanlab.mit.edu/projects/anycost-gan/images/method_pad.gif)

## 结果

Anycost GAN(统一通道版本)支持4个分辨率和4个通道比率，产生具有不同图像保真度的视觉一致性图像。

![uniform](https://hanlab.mit.edu/projects/anycost-gan/images/uniform.gif)



在图像投射和编辑过程中保持一致性:

![](https://hanlab.mit.edu/projects/anycost-gan/images/teaser.jpg)

![](https://hanlab.mit.edu/projects/anycost-gan/images/editing.jpg)



## 项目用法

### 项目的开始

- 克隆整个项目:

```bash
git clone https://github.com/mit-han-lab/anycost-gan.git
cd anycost-gan
```

- 安装 PyTorch 1.7 和其他项目所需环境.

我们建议使用Anaconda设置环境:: `conda env create -f environment.yml`

通过项目里面的environment.yml来配置项目所需要的环境。



### 介绍笔记

我们提供了一个jupyter notebook示例来展示如何使用anycost生成器以不同的成本进行图像合成:`notebooks/intro.ipynb`.

我们还提供了一个协作版本的笔记本: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mit-han-lab/anycost-gan/blob/master/notebooks/intro_colab.ipynb)。确保在运行时选项中选择GPU作为加速器。



### 互动演示

我们提供了一个交互式演示，展示了如何使用任何成本的GAN来实现交互式图像编辑。要运行demo.py的代码文件:

```bash
python demo.py
```

如果您的计算机包含CUDA GPU，请尝试使用:
```bash
FORCE_NATIVE=1 python demo.py
```

你可以找到演示的视频记录，点击 [这里](https://youtu.be/_yEziPl9AkM?t=90)。



### 使用预训练模型

要获取预训练的生成器、编码器和编辑方向，请运行:

```python
import models

pretrained_type = 'generator'  # choosing from ['generator', 'encoder', 'boundary']
config_name = 'anycost-ffhq-config-f'  # replace the config name for other models
models.get_pretrained(pretrained_type, config=config_name)
```

我们还提供了用于计算编辑方向的人脸属性分类器(这对于不同的生成器是通用的)。你可以通过运行:

```python
models.get_pretrained('attribute-predictor')
```

属性分类器采用FFHQ格式的人脸图像。



加载 Anycost 生成器后, 我们可以在很大的计算成本范围内运行它。例如:

```python
from models.dynamic_channel import set_uniform_channel_ratio, reset_generator

g = models.get_pretrained('generator', config='anycost-ffhq-config-f')  # anycost uniform
set_uniform_channel_ratio(g, 0.5)  # set channel
g.target_res = 512  # set resolution
out, _ = g(...)  # generate image
reset_generator(g)  # restore the generator
```

有关详细使用方法和*灵活通道*Anycost 生成器，请参阅 `notebooks/intro.ipynb`.



### 框架选择

目前，我们提供以下预训练的生成器，编码器和编辑方向。我们将在未来添加更多。

对于Anycost 生成器，默认情况下，我们引用统一设置。

| config name                    | generator          | encoder            | edit direction     |
| ------------------------------ | ------------------ | ------------------ | ------------------ |
| anycost-ffhq-config-f          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| anycost-ffhq-config-f-flexible | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| anycost-car-config-f           | :heavy_check_mark: |                    |                    |
| stylegan2-ffhq-config-f        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

`stylegan2-ffhq-config-f` 参考从[repo](https://github.com/NVlabs/stylegan2)转换而来的官方StyleGAN2生成器。



### 数据集

我们将[FFHQ](https://github.com/NVlabs/ffhq-dataset)， [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ)和[LSUN Car](https://github.com/fyu/lsun)数据集准备到一个图像目录中，以便它可以轻松地与' torchvision '中的' ImageFolder '一起使用。数据集布局如下:

```
├── PATH_TO_DATASET
│   ├── images
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   ├── ...
```

由于版权问题，您需要从官方网站下载数据并进行相应的处理。

### 评价

我们提供了代码来评估论文中提出的一些指标。一些代码是用[horovod ](https://github.com/horovod/horovod)编写的，以支持分布式评估并降低gpu间通信的成本，从而大大提高了速度。请查看其网站以获得正确的安装

#### Fre ́chet Inception Distance (FID)

在评估FIDs之前，需要使用如下脚本计算真实图像的初始特征:

```bash
python tools/calc_inception.py \
    --resolution 1024 --batch_size 64 -j 16 --n_sample 50000 \
    --save_name assets/inceptions/inception_ffhq_res1024_50k.pkl \
    PATH_TO_FFHQ
```

或者你可以从[这里](https://www.dropbox.com/sh/bc8a7ewlvcxa2cf/AAD8NFzDWKmBDpbLef-gGhRZa?dl=0)下载预计算的inception，并将其放在“assets/inception”下。

然后，您可以通过运行以下命令来评估FID:

```bash
horovodrun -np N_GPU \
    python metrics/fid.py \
    --config anycost-ffhq-config-f \
    --batch_size 16 --n_sample 50000 \
    --inception assets/inceptions/inception_ffhq_res1024_50k.pkl
    # --channel_ratio 0.5 --target_res 512  # optionally using a smaller resolution/channel
```

#### Perceptual Path Lenght (PPL)

同样的，评估PPL用 :

```bash
horovodrun -np N_GPU \
    python metrics/ppl.py \
    --config anycost-ffhq-config-f
```

#### Attribute Consistency

Evaluating the attribute consistency by running:

```bash
horovodrun -np N_GPU \
    python metrics/attribute_consistency.py \
    --config anycost-ffhq-config-f \
    --channel_ratio 0.5 --target_res 512  # config for the sub-generator; necessary
```

#### Encoder Evaluation

通过运行评估属性一致性:

```bash
python metrics/eval_encoder.py \
    --config anycost-ffhq-config-f \
    --data_path PATH_TO_CELEBA_HQ
```



