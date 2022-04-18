
## Introduction {#introduction}

KMS[^fn:1] uses a client-server model, to use it, you need to have a KMS host (server) available on your network.

Computers that activate with a KMS host need to have a specific product key (KMS client key, or formally as Microsoft Generic Volume License Key - GVLK).

Volume licensing editions are, by default, KMS clients with no extra configuration needed as the relevant GVLK is already there.


## Get a product key {#get-a-product-key}

Get a product key from [kms client activation product keys](https://docs.microsoft.com/en-us/windows-server/get-started/kms-client-activation-keys) for you running windows edition.

To check your current windows version: run `winver`


## Start KMS server {#start-kms-server}

There are several KMS emulators, choose any:

-   [py-kms](https://github.com/SystemRage/py-kms)
-   [vlmcsd](https://github.com/Wind4/vlmcsd)

I recommend using py-kms with docker:

```shell
docker run -d --name py-kms --restart always -p 1688:1688 pykmsorg/py-kms
```

arguments explained:

-   `-d` run in the background
-   `-name py-kms` container name is `py-kms` (name whatever you want)
-   `--restart always` always restart the container if it's not running (unless manually stopped, but will still restart if docker daemon restarts)
-   `-p 1688:1688` map host port 1688 (left) to the port 1688 (right) in the container
-   `pykmsorg/py-kms` the docker image to run


## activate with slmgr {#activate-with-slmgr}

In your administrator powershell (`win + r`, `powershell`, `Ctrl+Shift+Enter`):

```shell
# uninstall product key
# slmgr.vbs /upk
# set/change the product key
slmgr /ipk W269N-WFGWX-YVC9B-4J6C9-T83GX
# set the kms server address, 1688 is the default port (can be omitted)
slmgr /skms 192.168.122.1:1688
# activate windows
slmgr /ato
# activation status
slmgr /dli
# activation status (verbose)
# slmgr /dlv
```


## Resources {#resources}

-   To install office, choose volume licensing edition, after setting KMS server, it will be activated automatically: [Office Tool Plus](https://otp.landian.vip/zh-cn/)

[^fn:1]: KMS (Key Management Service) activates Microsoft products on a local network.
