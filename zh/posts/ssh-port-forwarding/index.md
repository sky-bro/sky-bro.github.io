
SSH 端口转发（也称 SSH 隧道）允许你通过加密的 SSH 连接安全地路由网络流量。分为两种方向：**本地转发**（`-L`）用于访问远程服务，**远程转发**（`-R`）用于将本地服务暴露给远程机器。

{{< figure src="/images/posts/ssh-port-forwarding/ssh-cover.svg" caption="<span class=\"figure-number\">图 1： </span>SSH 端口转发通过加密隧道连接两台机器" width="90%" >}}

## 远程端口转发（`-R`） {#remote-port-forwarding}

远程端口转发将**远程主机**上的某个端口转发到本地机器指定的地址和端口。

**语法：**

```shell
ssh -R [remote_bind_address:]remote_port:local_host:local_port user@remote_host
```

**示例：** 将远程服务器的 8080 端口转发到本地的 web 服务（localhost:3000）。

```shell
ssh -R 8080:localhost:3000 user@remote_server
```

配置完成后，任何连接到 `remote_server:8080` 的流量都会到达你本地的 `localhost:3000`。

{{< figure src="/images/posts/ssh-port-forwarding/remote-forwarding.svg" caption="<span class=\"figure-number\">图 2： </span>远程端口转发：远程服务器 8080 端口的流量通过 SSH 隧道回传到本地机器的 3000 端口" width="90%" >}}

### 常见场景

- 将本地开发中的服务暴露给远程服务器或同事访问
- 内网穿透：让公网服务器访问你 NAT/防火墙后面的服务
- 远程调试：远程服务器连接你本地的数据库

### 关键参数

- `-R` 指定远程端口转发
- `GatewayPorts yes`（远程服务器 `sshd_config` 中）：允许其他主机访问转发的端口，默认仅绑定 `127.0.0.1`
- `-N` 不执行远程命令，仅做端口转发
- `-f` 后台运行

**示例：** 允许所有远程主机访问：

```shell
ssh -R 0.0.0.0:8080:localhost:3000 user@remote_server
```

{{< alert theme="info" >}}

默认情况下，`-R` 在远程侧仅绑定 `127.0.0.1`，即只有远程机器自身的进程可以访问转发端口。要允许外部访问，需在远程服务器的 `/etc/ssh/sshd_config` 中设置 `GatewayPorts yes`，或显式绑定到 `0.0.0.0`。

{{< /alert >}}

## 本地端口转发（`-L`） {#local-port-forwarding}

本地端口转发将**本地机器**上的某个端口转发到 SSH 服务器可访问的目标地址和端口。

**语法：**

```shell
ssh -L [local_bind_address:]local_port:remote_host:remote_port user@ssh_server
```

**示例：** 通过 SSH 跳板机访问远程数据库。

```shell
ssh -L 5432:db-server.internal:5432 user@jump_host
```

连接成功后，本地的 `localhost:5432` 即等价于跳板机视角的 `db-server.internal:5432`。

{{< figure src="/images/posts/ssh-port-forwarding/local-forwarding.svg" caption="<span class=\"figure-number\">图 3： </span>本地端口转发：你连接本地端口，流量通过 SSH 隧道转发到远程服务" width="90%" >}}

### 常见场景

- 安全访问内网服务（数据库、Redis、管理后台等）
- 通过跳板机连接被防火墙隔离的服务
- 加密本地到远程的通信通道

### 关键参数

- `-L` 指定本地端口转发
- `-N` 不执行远程命令，仅做端口转发
- `-f` 后台运行

**示例：** 仅绑定本地回环（默认行为）：

```shell
ssh -L 8888:api.internal:443 user@gateway
```

## 对比 {#comparison}

| 特性 | Local（`-L`） | Remote（`-R`） |
|---|---|---|
| 方向 | 本地端口 → 远程目标 | 远程端口 → 本地目标 |
| 典型用途 | 通过跳板访问内网服务 | 将本地服务暴露给远程侧 |
| 绑定地址 | 默认 `127.0.0.1` | 默认 `127.0.0.1`（远程侧） |
| 触发条件 | 连接本地端口时触发 | 连接远程端口时触发 |

## 通过 `~/.ssh/config` 持久化配置 {#ssh-config}

通过 `~/.ssh/config` 可以持久化端口转发配置，无需每次手动指定参数。

### 命令行到 config 的映射

| 命令行参数 | config 关键字 | 值格式 |
|---|---|---|
| `-L` | `LocalForward` | `local_port remote_host:remote_port` |
| `-R` | `RemoteForward` | `remote_port local_host:local_port` |

### LocalForward 示例

```text
# ~/.ssh/config
Host db-tunnel
    HostName jump-server.example.com
    User admin
    LocalForward 5432 db-server.internal:5432
    LocalForward 6379 redis.internal:6379
```

使用方式：`ssh db-tunnel`（等同于 `ssh -L 5432:db-server.internal:5432 -L 6379:redis.internal:6379 admin@jump-server.example.com`）。

连接后本地访问 `localhost:5432` 和 `localhost:6379` 即可通过跳板机连接到内网服务。

### RemoteForward 示例

```text
# ~/.ssh/config
Host expose-dev
    HostName public-server.example.com
    User deploy
    RemoteForward 8080 localhost:3000
    RemoteForward 9090 localhost:8080
```

使用方式：`ssh expose-dev`（将远程服务器的 8080/9090 端口分别转发到本地的 3000/8080）。

### 仅转发不登录

```shell
ssh -N db-tunnel      # 仅建立端口转发，不打开远程 shell
ssh -N -f db-tunnel   # 同上，放入后台运行
```

将 `-N` 与 config 别名配合使用，可以快速建立后台隧道。

{{< alert theme="success" >}}

你可以在单个 `Host` 块下定义多个 `LocalForward` 或 `RemoteForward`，一次连接建立多条隧道。

{{< /alert >}}
