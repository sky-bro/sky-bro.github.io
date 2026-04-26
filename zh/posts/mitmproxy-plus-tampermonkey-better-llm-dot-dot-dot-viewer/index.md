
## 引言 {#introduction}

调试 LLM 应用时，我经常想直接看到真实的 HTTP 流量：请求体、模型名、tool call、流式返回片段、token usage，以及最终响应。SDK 日志当然有用，但通常不是太概括，就是太吵。

[mitmproxy](https://mitmproxy.org/) 很适合做这件事。它可以拦截 HTTP、HTTPS 和 WebSocket 流量，并在 `mitmweb` 里展示每个 flow。问题在于展示形式：原始 JSON 信息完整，但当 payload 里有长消息或嵌套 tool call 时，并不好读。

Tampermonkey 正好可以补上这一块。`mitmweb` 本质上就是浏览器里的一个 Web UI，因此我们可以用 userscript 在本地增强页面。本文的方案是：用 mitmproxy 捕获 LLM API 调用，再用 Tampermonkey 脚本把选中的 JSON 响应渲染成更清爽的对话式视图。

## 最终效果 {#what-you-ll-get}

配置完成后，你可以：

- 在少改代码甚至不改代码的情况下拦截 LLM API 调用；
- 实时查看 request、response、header、streaming chunk 和错误信息；
- 把大段 JSON payload 渲染成面向对话阅读的布局；
- 用普通 JavaScript 自定义查看器，而不用去改 mitmproxy 本身。

```mermaid
flowchart LR
    app["App / SDK<br/>Browser, CLI, service<br/>base_url or proxy"]
    proxy["mitmproxy<br/>reverse or forward proxy<br/>listen :8080"]
    provider["LLM Provider<br/>OpenAI-compatible<br/>Anthropic / Gemini / ..."]
    web["mitmweb UI<br/>inspect flows<br/>web :8081"]
    tm["Tampermonkey<br/>local userscript<br/>chat-like payload view"]

    app -->|"HTTP(S) API call"| proxy
    proxy -->|"forward request"| provider
    provider -->|"response / stream"| proxy
    proxy -->|"response"| app
    proxy -.->|"captured flows"| web
    tm -.->|"enhances DOM"| web
```

<span class="figure-number">Figure 1: </span>架构图：mitmproxy 捕获 API 流量，mitmweb 与 Tampermonkey 在本地渲染更好读的查看器。

{{< figure src="/images/posts/mitmproxy-plus-tampermonkey-better-llm-dot-dot-dot-viewer/llm-better-view-example.png" caption="<span class=\"figure-number\">Figure 2: </span>LLM Better View 示例" >}}

## 捕获 LLM API 调用 {#capture-llm-api-calls}

让 mitmproxy 站到应用和 LLM 服务之间有几种方式。我的习惯是先问一个问题：**我能不能控制客户端的出口？**

| 场景 | 推荐模式 | 客户端怎么配 | 适合程度 |
| --- | --- | --- | --- |
| SDK 或网关可以改 `base_url` | reverse proxy | 把 API endpoint 指到 mitmproxy | 最推荐 |
| 命令行、浏览器或服务可以设置代理 | forward proxy | 设置 `HTTP_PROXY` / `HTTPS_PROXY`，并信任 mitmproxy CA | 很常用 |
| 客户端完全不能改，只能动网络 | transparent proxy | 改路由 / NAT / 网关 | 最后手段 |

日常调试 LLM 应用时，我通常优先用 reverse proxy mode。它显式、好理解，而且大部分 SDK 都允许覆盖 API base URL。如果你已经用 LiteLLM、Open WebUI、one-api 这类中间层，reverse proxy 会更自然：让应用继续访问“本地网关”，再让 mitmproxy 站在网关前面。

### Reverse Proxy Mode {#reverse-proxy-mode}

如果你的应用或 SDK 允许配置 API endpoint，比如很多 OpenAI-compatible client 里的 `base_url`，就适合用这种方式。

在这个模式下，应用会把 mitmproxy 当成 API server 访问；mitmproxy 再把请求转发给真正的上游服务。

```shell
mitmweb \
  --mode reverse:https://api.openai.com@8080 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --no-web-open-browser \
  --showhost \
  --set web_password='sky'
```

然后把客户端的 base URL 配成：

```text
http://localhost:8080
```

`mitmweb` 的页面地址是：

```text
http://localhost:8081
```

这里最容易混淆的是端口：`--mode reverse:https://api.openai.com@8080` 里的 `@8080` 是代理/API 访问端口，`--web-port 8081` 是 mitmweb 的 UI 端口。把监听端口写进 `--mode` 里，比单独写一个全局 `--listen-port` 更直观。

如果上游本来就是内网服务，比如 compose 里的 `litellm:4000`，可以直接把 reverse upstream 写成内网服务名：

```shell
mitmweb \
  --mode reverse:http://litellm:4000@4001 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --no-web-open-browser \
  --showhost \
  --set web_password='mitm'
```

这里的 `@4001` 表示这个 reverse mode 自己监听 `4001`。客户端访问 `http://<host>:4001`，mitmproxy 再转发到容器网络里的 `http://litellm:4000`。

### Forward Proxy Mode {#forward-proxy-mode}

如果你能配置浏览器、终端或系统代理，可以使用 forward proxy mode。这个模式下，客户端仍然访问原始 API URL，只是网络流量会经过 mitmproxy。

```shell
mitmweb \
  --mode regular@8080 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --no-web-open-browser \
  --showhost \
  --set web_password='sky'
```

然后把客户端的 HTTP/HTTPS 代理设为 `localhost:8080`。对于命令行工具，通常这样就够了：

```shell
export HTTP_PROXY=http://localhost:8080
export HTTPS_PROXY=http://localhost:8080
```

{{< alert theme="warning" >}}

forward proxy 只设置 `HTTP_PROXY` / `HTTPS_PROXY` 还不够。要解密 HTTPS 请求体和响应体，发起请求的环境必须安装并信任 mitmproxy 生成的本地 CA 证书；否则客户端通常会报证书错误，或者 mitmproxy 只能看到隧道连接而看不到里面的明文 HTTP 内容。

{{< /alert >}}

让流量经过 mitmproxy 后访问 `http://mitm.it`，即可下载对应平台的证书。注意证书要装在真正发起请求的环境里：浏览器就装到浏览器或系统信任库，容器里的 CLI 就装到容器镜像或运行时使用的 CA bundle，Node/Python 等工具如果自带 CA 配置，也要同步指向这个证书。

mitmproxy 也可以同时开多个 mode。例如下面的 compose 示例同时提供：

- `4001`：反向代理到 LiteLLM，适合把 OpenAI-compatible client 的 `base_url` 改成这里；
- `8080`：普通正向代理，适合命令行或浏览器设置 `HTTP_PROXY` / `HTTPS_PROXY`；
- `8081`：mitmweb 可视化界面。

### Docker Compose 部署示例：LiteLLM + mitmproxy {#docker-compose-example-litellm-plus-mitmproxy}

下面这个例子接近我在 TrueNAS 上的部署方式：LiteLLM 负责提供 OpenAI-compatible API，Postgres 持久化 LiteLLM 配置，Prometheus 保留指标，mitmproxy 同时提供 reverse proxy 和 forward proxy。

```yaml
services:
  litellm:
    image: docker.litellm.ai/berriai/litellm:main-stable
    ports:
      - "4000:4000"
    environment:
      DATABASE_URL: "postgresql://llmproxy:change-me-db-password@db:5432/litellm"
      STORE_MODEL_IN_DB: "True"
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test:
        - CMD-SHELL
        - python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:4000/health/liveliness')"
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  mitmproxy:
    image: mitmproxy/mitmproxy:latest
    command: >
      mitmweb
      --mode reverse:http://litellm:4000@4001
      --mode regular@8080
      --no-web-open-browser
      --web-host 0.0.0.0
      --web-port 8081
      --showhost
      --set web_password='mitm'
    ports:
      - "4001:4001"
      - "8080:8080"
      - "8081:8081"
    depends_on:
      litellm:
        condition: service_healthy

  db:
    image: postgres:16
    restart: always
    container_name: litellm_db
    environment:
      POSTGRES_DB: litellm
      POSTGRES_USER: llmproxy
      POSTGRES_PASSWORD: change-me-db-password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -d litellm -U llmproxy"]
      interval: 1s
      timeout: 5s
      retries: 10

  prometheus:
    image: prom/prometheus
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - prometheus_data:/prometheus
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=15d"

volumes:
  prometheus_data:
    driver: local
  postgres_data:
    name: litellm_postgres_data
```

如果只想抓经过 LiteLLM 的 LLM 请求，客户端配置成：

```text
http://<truenas-or-docker-host>:4001
```

如果想让某个命令行工具“原样访问外部 API，但流量经过 mitmproxy”，就设置正向代理：

```shell
export HTTP_PROXY=http://<truenas-or-docker-host>:8080
export HTTPS_PROXY=http://<truenas-or-docker-host>:8080
```

这类正向代理方式同样需要让命令行工具所在环境信任 mitmproxy CA，否则 HTTPS API 调用会卡在证书校验上。

然后打开 `http://<truenas-or-docker-host>:8081` 查看 flow。因为 mitmweb 暴露在 `0.0.0.0`，一定要保留 `web_password`，最好再放在可信局域网或反向代理鉴权之后。

### Transparent Proxy Mode {#transparent-proxy-mode}

当你无法配置客户端，又需要在网络层拦截流量时，可以考虑 transparent mode。它也是侵入性最强的一种方式，所以只有 reverse proxy 和 forward proxy 都不够用时，我才会用它。

一个最小的 Linux 示例大概是：

```shell
# 开启 IP forwarding。
sudo sysctl -w net.ipv4.ip_forward=1

# 把 HTTPS 流量重定向到 mitmproxy。
sudo iptables -t nat -A PREROUTING -p tcp --dport 443 -j REDIRECT --to-port 8080

# 以 transparent mode 启动 mitmproxy。
mitmweb \
  --mode transparent@8080 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --showhost \
  --set web_password='sky'
```

这需要 root 权限、路由配置和证书信任。对于本地 LLM 应用调试，reverse proxy mode 通常要省心得多。

## 安装 Tampermonkey Better LLM View 脚本 {#install-tampermonkey-better-llm-view-script}

我把 userscript 放在这里：[mitmproxy-llm-better-view](https://github.com/sky-bro/mitmproxy-llm-better-view?tab=readme-ov-file#method-2-tampermonkey-script)。你可以直接安装，也可以 fork 后按自己的需求改。

基本步骤如下：

1. 在浏览器中安装 Tampermonkey 扩展。
2. 如果浏览器要求显式允许 user scripts，记得打开对应选项。
3. 打开 `mitmweb-llm-better-view.user.js`，在 Tampermonkey 中安装。
4. 调整 userscript 的 `@include` 或 `@match` 规则，使它匹配你的 mitmweb 地址，例如 `http://localhost:8081/*`。
5. 刷新 mitmweb，选中一个 LLM API flow。

{{< figure src="/images/posts/mitmproxy-plus-tampermonkey-better-llm-dot-dot-dot-viewer/llm-better-view-set-include-rule.png" caption="<span class=\"figure-number\">Figure 3: </span>设置脚本 include 规则" >}}

{{< figure src="/images/posts/mitmproxy-plus-tampermonkey-better-llm-dot-dot-dot-viewer/allow-user-script-for-tampermonkey.png" caption="<span class=\"figure-number\">Figure 4: </span>允许 Tampermonkey 运行 user scripts" >}}

启用后，脚本会观察 mitmweb 当前选中的 flow 详情，并把支持的 LLM JSON payload 转换成更易读的视图。因为它只是一个 userscript，渲染逻辑完全留在你的浏览器本地。你可以加 provider-specific 的解析逻辑，隐藏不关心的字段，或者重点高亮 tool call、token usage 等信息。

## 实用注意事项 {#practical-notes}

有几件事值得特别注意：

- 只拦截你拥有或明确被允许检查的流量。MITM proxy 可能暴露 secret、cookie、API key 和私有 prompt。
- 尽量使用测试 API key。捕获到的请求里经常会包含 bearer token。
- 如果 mitmweb 绑定到 `0.0.0.0`，建议保留 `web_password`；如果只需要本机访问，也可以绑定到 `127.0.0.1`。
- 如果客户端拒绝代理证书，检查它是否使用了自己的 CA bundle，或者是否做了 certificate pinning。
- 对于 streaming response，有时需要同时看原始 event stream 和重组后的 body，具体取决于 provider 和 client 的实现。

## 进阶：用 Addon 清理捕获历史 {#advanced-clean-up-captured-history-with-an-addon}

mitmproxy 支持 Python [addons](https://docs.mitmproxy.org/stable/addons/overview/)。如果你希望捕获列表在调试时保持干净，addon 会很有用。

比如下面这个极小的 addon 会标记那些 host 看起来不像 LLM provider 的 flow：

```python
from mitmproxy import http

LLM_HOST_KEYWORDS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
)


def response(flow: http.HTTPFlow) -> None:
    if not any(keyword in flow.request.pretty_host for keyword in LLM_HOST_KEYWORDS):
        flow.marked = ":skip:"
```

运行方式：

```shell
mitmweb \
  --mode regular@8080 \
  --web-port 8081 \
  --showhost \
  -s keep-llm-flows.py
```

这个例子只是标记 flow，但同一个 hook 也可以用来导出选中的 payload、脱敏敏感 header，或者附加一些元数据，让 mitmweb 的列表更好扫读。

## 总结 {#conclusion}

mitmproxy 本来就是一个很强的网络显微镜。Tampermonkey 则让它的浏览器 UI 变成一个可以按自己工作流塑形的地方。

对于 LLM 调试来说，这个组合尤其顺手：真实 API 流量可以被完整捕获，原始 request 和 response 仍然保留，而你真正需要阅读的部分可以用更友好的形式呈现。最终得到的是一个轻量、本地、可 hack 的查看器，而且不依赖你当天正在调试哪个 SDK 或 UI。
