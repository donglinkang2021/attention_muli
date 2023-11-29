# README

回顾一下李沐的d2l的seq章节——attention章节



## Windows配置环境

```shell
conda create --name d2l python=3.9 -y
```

```shell
conda activate d2l
```

```shell
pip install torch==1.12.0
pip install torchvision==0.13.0
```

```shell
pip install d2l==0.17.6
```

## 服务器配置环境

上传本文件夹到服务器

```bash
scp -r -P 25680 .\seq_Muli root@10.1.3.100:/root/
```

### 下载conda

> 自己新建了一个linkdom文件夹用来存自己的包

```bash
mkdir linkdom
cd linkdom
```

下载miniconda3

```bash
mkdir miniconda3
```

```shell
cd miniconda3
```

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

```shell
bash miniconda.sh -b -u -p miniconda3
```

```shell
rm -rf miniconda.sh
```

配置环境变量

```bash
cd miniconda3/bin
./conda init bash
./conda init zsh
```

```shell
source ~/.bashrc
```

### 其它

配置默认不启动

- 参考[conda退出base　环境 - Lust4Life - 博客园 (cnblogs.com)](https://www.cnblogs.com/g2thend/p/12090918.html#:~:text=方法一： 每次在命令行通过conda deactivate退出base环境回到系统自动的环境 方法二 1，通过将auto_activate_base参数设置为false实现： conda config --set,2，那要进入的话通过conda activate base 3，如果反悔了还是希望base一直留着的话通过conda config --set auto_activate_base true来恢复)

```bash
conda config --set auto_activate_base false
```

退出base环境

```
conda deactivate
```

### 换pip源

把`.config/pip/pip.conf`中的

```sh
[global]
no-cache-dir = true
index-url = https://pypi.org/simple
extra-index-url = https://pypi.ngc.nvidia.com
trusted-host = pypi.ngc.nvidia.com
```

替换成

```sh
[global]
no-cache-dir = true
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
extra-index-url = https://pypi.ngc.nvidia.com
[install]
trusted-host=mirrors.aliyun.com
```

查看是否修改成功

```shell
pip config list 
```

### 安装包

```shell
conda create --name d2l python=3.9 -y
```

```shell
conda activate d2l
```

目前还是装的旧的包，没有装新的包现在，但4090不兼容，所以我们先卸掉了

```bash
(d2l) root@interactive29746:~/linkdom/code# pip list | grep torch
torch                     1.12.0
torchvision               0.13.0
pip uninstall torch
pip uninstall torchvision
```

我们在pytorch官网查到了最新的安装方法（cuda121）

```shell
pip3 install torch torchvision torchaudio
```

```shell
pip install d2l==0.17.6
```

