本教程展示如何搭建clusterfuzz进行本地测试，教程使用的clusterfuzz版本为v2.0.1 (推荐总是使用最新的release版本)。
我的clusterfuzz将搭建在ubuntu18.04 docker容器中。最后提供一个dockerfile作为参考，下面内容基本是将dockerfile中的步骤一步步展开

其实我的步骤基本也是按照[官方教程](https://google.github.io/clusterfuzz)来的，但主要由于国内网络原因，一些地方需要科学上网。
我使用的是clash分流，也是运行在docker容器内，为了简单我的clash容器和ubuntu容器都是设置的network为host（原因是我发现在ubuntu中尽管设置了no_proxy，有的时候还是会代理127.0.0.1，具体原因不清楚，反正为了简单粗暴就先network用host，代理了也不怕，clash中会设置为直连）

## 运行ubuntu18.04

我提供了一个修改软件源后的ubuntu镜像：[ubuntu-cn](https://github.com/sky-bro/ubuntu-cn)，比较方便
所以运行容器docker:

```shell
# 你在克隆仓库的时候也可以按下一步设置下代理，快一些 (几十M)
git clone https://github.com/google/clusterfuzz.git
cd clusterfuzz
# 使用需要的版本，推荐用最新版本，当前是v2.0.1
git checkout -b testv2.0.1 v2.0.1 # 我使用-b创建了一个新分支，随意，也可以不用
docker container run --network host --name clusterfuzz -it -v $(pwd):/clusterfuzz skybro/ubuntu-cn:18.04 # 我只测试了ubuntu18.04
```

## 设置好代理

后续的操作没特别说明的话就都是在容器内了

```shell
# 感觉设置http代理比socks代理要适用性更广一些，我用的clash，默认http代理端口为7890
# 这里因为我上面network给设的host才用127.0.0.1地址
# 具体替换为你自己的代理地址就好
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export no_proxy=127.0.0.0/8 # 这个对于我这host network，设不设都行了，因为我clash中设置了直连（我发现即使设置了no_proxy，有的时候还是会走代理，所以直接network采用的host）
```

## 安装一些软件

### 基础软件

```shell
apt-get update && \
    apt-get upgrade -y && \
    apt-get autoremove -y && \
    apt-get install -y \
        apt-transport-https \
        build-essential \
        curl \
        gdb \
        libbz2-dev \
        libcurl4-openssl-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncurses5-dev \
        libnss3-dev \
        libreadline-dev \
        libssl-dev \
        locales \
        lsb-release \
        net-tools \
        socat \
        sudo \
        unzip \
        util-linux \
        wget \
        zip \
        zlib1g-dev \
        patchelf \
        git \
        vim
```

### google-cloud-sdk

```shell
CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -  && \
    apt-get update && apt-get install -y google-cloud-sdk
```

### Python3.7

```shell
curl -sS https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz | tar -C /tmp -xzv
(cd /tmp/Python-3.7.7 && ./configure --enable-optimizations && make altinstall && rm -rf /tmp/Python-3.7.7)
pip3.7 install --upgrade pip && pip3.7 install wheel && pip3.7 install pipenv
```

### golang

```shell
apt install -y golang
```

### java

```shell
apt install -y openjdk-8-jdk
```

### gcloud依赖

```shell
apt-get install -y \
      google-cloud-sdk-app-engine-go \
      google-cloud-sdk-app-engine-python \
      google-cloud-sdk-app-engine-python-extras \
      google-cloud-sdk \
      google-cloud-sdk-datastore-emulator \
      google-cloud-sdk-pubsub-emulator
```

### python依赖

```shell
cd clusterfuzz
python3.7 -m pipenv --python 3.7
python3.7 -m pipenv sync --dev
```

### 其它依赖

```shell
pipenv shell # 进入python虚拟环境
nodeenv -p --prebuilt # 可能需要反复多尝试几次，应该很快的，慢的话就CTRL+C再重来
# 因为我是root用户，需要下面这两行 ref: https://stackoverflow.com/questions/51811564/sh-1-node-permission-denied
npm config set unsafe-perm true
npm install -g bower polymer-bundler
# bower install
# 同样因为我是root用户，需要加上--allow-root
bower install --allow-root
python butler.py bootstrap
```

## 运行

### 启动虚拟环境

上面已经启动了，就是进入clusterfuzz目录，然后运行`pipenv shell`

```shell
cd /clusterfuzz
pipenv shell
```

### butler.py

butler就是管家的意思，我们通过这个python文件执行各种功能，查看帮助`python butler.py --help`
平时主要用到就两个功能

#### 启动网页(App Engine)

查看帮助: `python butler.py run_server --help`

第一次运行需要加上`-b`选项(`--bootstrap`)，以后就不用了，另外推荐加上`--skip-install-deps` (不加上每次都安装很多东西，感觉是我们之前依赖都已经装过了，我试过加上没事，有问题再去掉这个选项): `python butler.py run_server -b --skip-install-deps`，看到下面内容就运行ok了

```shell
(clusterfuzz) root@manjaro:/clusterfuzz# python butler.py run_server -b --skip-install-deps
Running: pkill -KILL -f "dev_appserver.py"
| Return code is non-zero (-9).
Running: pkill -KILL -f "CloudDatastore.jar"
| Return code is non-zero (-9).
Running: pkill -KILL -f "pubsub-emulator"
| Return code is non-zero (-9).
Running: pkill -KILL -f "run_bot"
| Return code is non-zero (-9).
Created symlink: source: /clusterfuzz/configs/test, target /clusterfuzz/src/appengine/config.
Created symlink: source: /clusterfuzz/src/protos, target /clusterfuzz/src/appengine/protos.
Created symlink: source: /clusterfuzz/src/python, target /clusterfuzz/src/appengine/python.
Running: python polymer_bundler.py (cwd='local')
| Building templates for App Engine...
| App Engine templates built successfully.
Created symlink: source: /clusterfuzz/local/storage/local_gcs, target /clusterfuzz/src/appengine/local_gcs.
Running: gunicorn -b :9000 main:app (cwd='src/appengine')
| [2020-05-19 09:24:25 +0800] [22845] [INFO] Starting gunicorn 20.0.4
| [2020-05-19 09:24:25 +0800] [22845] [INFO] Listening at: http://0.0.0.0:9000 (22845)
| [2020-05-19 09:24:25 +0800] [22845] [INFO] Using worker: sync
| [2020-05-19 09:24:25 +0800] [22855] [INFO] Booting worker with pid: 22855
Bootstrapping datastore...
Running: python butler.py run setup --non-dry-run --local --config-dir=configs/test
| Creating config
| Creating fuzzer afl
| Creating fuzzer libFuzzer
| Creating fuzzer honggfuzz
| Creating fuzzer syzkaller
| Creating template afl
| Creating template engine_asan
| Creating template engine_msan
| Creating template engine_ubsan
| Creating template honggfuzz
| Creating template libfuzzer
| Creating template syzkaller
| Creating template prune
| Done
```

之后的运行都不需要加`-b`选项，直接`python butler.py run_server --skip-install-deps`

#### 运行fuzzing bots

上面的网页(app engine)相当于只是一个交互界面/控制台而已，具体的[fuzzing工作](https://google.github.io/clusterfuzz/architecture/#fuzzing-bots)都还需要运行fuzzing bots才行，fuzzing bots可以运行多个，而且可以单独运行在其它容器中（后面再试，现在直接还在这个容器中运行）

另外，我在docker容器内运行的ubuntu，要多开一个命令行的话可以使用tmux
安装tmux: `apt install tmux`

或者主机上在运行的容器上再开一个shell

```shell
docker container exec -it clusterfuzz bash
root@manjaro:/# cd /clusterfuzz
root@manjaro:/clusterfuzz# pipenv shell
Launching subshell in virtual environment…
root@manjaro:/clusterfuzz#  . /root/.local/share/virtualenvs/clusterfuzz-rAL0Uxhl/bin/activate
(clusterfuzz) root@manjaro:/clusterfuzz#
```

查看运行bots的帮助: `python butler.py run_bot --help`
运行效果如下

```shell
python butler.py run_bot --name bot01 ./bot01  # rename my-bot to anything
```

## 测试

这里的步骤可以参考官方教程，比较详细: [Setting up fuzzing](https://google.github.io/clusterfuzz/setting-up-fuzzing/)

我这里简单地进行部分翻译，下面的操作不用在容器内进行

### 安装libfuzzer

debian系统可参考[LLVM Debian/Ubuntu nightly packages](https://apt.llvm.org/)，我也不清楚到底安装哪些，也没找到好的教程，源码安装感觉不太方便，直接用包管理器应该就行，熟悉这块儿的欢迎留言告诉。

### 编写fuzzer

我这里直接使用这里的[fuzz_me.cc](https://github.com/google/fuzzing/blob/master/tutorial/libFuzzer/fuzz_me.cc):

```c++
#include <stdint.h>
#include <stddef.h>

bool FuzzMe(const uint8_t *Data, size_t DataSize) {
  return DataSize >= 3 &&
      Data[0] == 'F' &&
      Data[1] == 'U' &&
      Data[2] == 'Z' &&
      Data[3] == 'Z';  // :‑<
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  FuzzMe(Data, Size);
  return 0;
}
```

### 上传fuzzer进行测试

编译，压缩

```shell
clang++ -g fuzz_me.cc -o fuzz_me -fsanitize=address,undefined,fuzzer
zip fuzz_me.zip fuzz_me
```

在jobs页面最底下填写ADD NEW JOB表格(具体怎么填看官方文档: [Creating a job type](https://google.github.io/clusterfuzz/setting-up-fuzzing/libfuzzer-and-afl/#creating-a-job-type))，然后点击ADD添加

在Fuzzers页面选择libFuzzer，点击EDIT，勾选刚才的job，提交SUBMIT

### 查看结果

## 常见问题

1. 如果你是直接在本机ubuntu上测试，有些命令需要用sudo才能执行，而环境变量http_proxy和https_proxy很可能不会被继承，及用sudo执行命令很可能不会走代理，你可以参考[How to run “sudo apt-get update” through proxy in commandline?](https://askubuntu.com/questions/7470/how-to-run-sudo-apt-get-update-through-proxy-in-commandline)进行设置:
  即执行visudo，在`Defaults env_reset`所在行下面添加
  `Defaults env_keep="http_proxy https_proxy ftp_proxy DISPLAY XAUTHORITY"`
2. 啊的

## 参考

* 安装及使用主要参考[clusterfuzz官方文档](https://google.github.io/clusterfuzz)
* 安装过程还主要参考了clusterfuzz仓库内的两个文件: [Dockerfile](https://github.com/google/clusterfuzz/blob/master/docker/base/Dockerfile), [install_deps_linux.bash](https://github.com/google/clusterfuzz/blob/master/local/install_deps_linux.bash)
* libFuzzer的使用参考[libFuzzer Tutorial](https://github.com/google/fuzzing/blob/master/tutorial/libFuzzerTutorial.md)和[libfuzzer官方文档](https://bcain-llvm.readthedocs.io/projects/llvm/en/release_39/LibFuzzer/)
