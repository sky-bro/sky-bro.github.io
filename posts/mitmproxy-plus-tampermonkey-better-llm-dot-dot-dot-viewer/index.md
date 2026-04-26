
## Introduction {#introduction}

When I debug LLM applications, I often want to see the actual HTTP traffic: the request body, model name, tool calls, streaming chunks, token usage, and the final response. SDK logs are useful, but they are usually either too summarized or too noisy.

[mitmproxy](https://mitmproxy.org/) is great for this because it can intercept HTTP, HTTPS, and WebSocket traffic, then show each flow in `mitmweb`. The missing piece is presentation: raw JSON is precise, but not pleasant to read when the payload contains long messages or nested tool calls.

That is where Tampermonkey fits nicely. Since `mitmweb` is just a browser UI, a userscript can enhance the page locally. The setup in this post uses mitmproxy to capture LLM API calls and a Tampermonkey script to render selected JSON responses as a cleaner chat-style view.

## What You'll Get {#what-you-ll-get}

After the setup, you can:

- intercept LLM API calls without changing application code much, or at all;
- inspect requests, responses, headers, streaming chunks, and errors in real time;
- render large JSON payloads in a readable conversation-oriented layout;
- customize the viewer with plain JavaScript instead of patching mitmproxy itself.

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

<span class="figure-number">Figure 1: </span>Architecture: mitmproxy captures the API traffic, while mitmweb and Tampermonkey render a better local viewer.

{{< figure src="/images/posts/mitmproxy-plus-tampermonkey-better-llm-dot-dot-dot-viewer/llm-better-view-example.png" caption="<span class=\"figure-number\">Figure 2: </span>LLM Better View Example" >}}

## Capture LLM API Calls {#capture-llm-api-calls}

There are several ways to put mitmproxy between your application and an LLM provider. The first question I ask is: **can I control the client's network exit point?**

| Scenario | Recommended mode | Client configuration | Fit |
| --- | --- | --- | --- |
| The SDK or gateway can change `base_url` | reverse proxy | Point the API endpoint at mitmproxy | Best default |
| The CLI, browser, or service can use a proxy | forward proxy | Set `HTTP_PROXY` / `HTTPS_PROXY` and trust mitmproxy's CA | Very common |
| The client cannot be changed at all | transparent proxy | Change routing / NAT / gateway rules | Last resort |

For day-to-day LLM development, I normally start with reverse proxy mode. It is explicit, easy to reason about, and most SDKs let you override the API base URL. If you already run a middle layer such as LiteLLM, Open WebUI, or one-api, reverse proxy mode is even more natural: keep the application pointed at a local gateway, then put mitmproxy in front of that gateway.

### Reverse Proxy Mode {#reverse-proxy-mode}

Use this when your application or SDK lets you configure an API endpoint, for example `base_url` in many OpenAI-compatible clients.

In this mode, your app talks to mitmproxy as if it were the API server. mitmproxy forwards the request to the real upstream provider.

```shell
mitmweb \
  --mode reverse:https://api.openai.com@8080 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --no-web-open-browser \
  --showhost \
  --set web_password='sky'
```

Then configure the client to use:

```text
http://localhost:8080
```

Open `mitmweb` at:

```text
http://localhost:8081
```

The important detail is that the `@8080` in `--mode reverse:https://api.openai.com@8080` is the proxy/API-facing port, while `--web-port 8081` is the mitmweb UI port. Keeping the listen port inside `--mode` is easier to read than using a separate global `--listen-port`.

If the upstream is an internal service, such as `litellm:4000` inside a compose network, you can point the reverse upstream at the service name:

```shell
mitmweb \
  --mode reverse:http://litellm:4000@4001 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --no-web-open-browser \
  --showhost \
  --set web_password='mitm'
```

The `@4001` part means this reverse mode listens on `4001`. The client talks to `http://<host>:4001`, and mitmproxy forwards the request to `http://litellm:4000` inside the container network.

### Forward Proxy Mode {#forward-proxy-mode}

Use this when you can configure the browser, terminal, or system proxy. In this mode, the client keeps using the original API URL, but the network traffic is routed through mitmproxy.

```shell
mitmweb \
  --mode regular@8080 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --no-web-open-browser \
  --showhost \
  --set web_password='sky'
```

Then configure the client to use `localhost:8080` as its HTTP/HTTPS proxy. For command-line tools, this is often enough:

```shell
export HTTP_PROXY=http://localhost:8080
export HTTPS_PROXY=http://localhost:8080
```

{{< alert theme="warning" >}}

For forward proxy mode, setting `HTTP_PROXY` / `HTTPS_PROXY` is not enough by itself. To decrypt HTTPS request and response bodies, the environment that sends the traffic must install and trust mitmproxy's generated local CA certificate. Otherwise, the client will usually fail certificate validation, or mitmproxy will only see the tunnel connection without the plaintext HTTP content inside it.

{{< /alert >}}

Visit `http://mitm.it` while routed through mitmproxy to download the certificate for your platform. Install it where the request is actually made: a browser needs the browser or system trust store, a CLI inside a container needs the container image or runtime CA bundle, and tools such as Node or Python may need their own CA configuration pointed at that certificate.

mitmproxy can also run multiple modes at once. The compose example below exposes:

- `4001`: reverse proxy to LiteLLM, useful when an OpenAI-compatible client can change `base_url`;
- `8080`: regular forward proxy, useful for CLIs or browsers with `HTTP_PROXY` / `HTTPS_PROXY`;
- `8081`: the mitmweb UI.

### Docker Compose Example: LiteLLM + mitmproxy {#docker-compose-example-litellm-plus-mitmproxy}

This is close to the way I deploy it on TrueNAS: LiteLLM provides an OpenAI-compatible API, Postgres persists LiteLLM state, Prometheus keeps metrics, and mitmproxy exposes both reverse proxy and forward proxy entry points.

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

If you only want to capture requests that go through LiteLLM, configure the client base URL as:

```text
http://<truenas-or-docker-host>:4001
```

If you want a CLI tool to keep using the original upstream API while routing traffic through mitmproxy, use the forward proxy port:

```shell
export HTTP_PROXY=http://<truenas-or-docker-host>:8080
export HTTPS_PROXY=http://<truenas-or-docker-host>:8080
```

This forward proxy setup also requires the CLI environment to trust mitmproxy's CA, otherwise HTTPS API calls will fail at certificate validation.

Then open `http://<truenas-or-docker-host>:8081` to inspect flows. Because mitmweb is bound to `0.0.0.0`, keep `web_password` enabled and preferably put it on a trusted LAN or behind another authenticated reverse proxy.

### Transparent Proxy Mode {#transparent-proxy-mode}

Transparent mode is useful when you need network-level interception and cannot configure the client. It is also the most invasive option, so I only reach for it when reverse or forward proxy mode is not enough.

A minimal Linux sketch looks like this:

```shell
# Enable IP forwarding.
sudo sysctl -w net.ipv4.ip_forward=1

# Redirect HTTPS traffic to mitmproxy.
sudo iptables -t nat -A PREROUTING -p tcp --dport 443 -j REDIRECT --to-port 8080

# Run mitmproxy in transparent mode.
mitmweb \
  --mode transparent@8080 \
  --web-host 0.0.0.0 \
  --web-port 8081 \
  --showhost \
  --set web_password='sky'
```

This needs root privileges, routing setup, and certificate trust. For local LLM app debugging, reverse proxy mode is usually much calmer.

## Install the Tampermonkey Better LLM View Script {#install-tampermonkey-better-llm-view-script}

I keep the userscript here: [mitmproxy-llm-better-view](https://github.com/sky-bro/mitmproxy-llm-better-view?tab=readme-ov-file#method-2-tampermonkey-script). You can install it directly, fork it, or treat it as a starting point for your own viewer.

The basic steps are:

1. Install the Tampermonkey extension in your browser.
2. Enable browser support for user scripts if your browser requires it.
3. Open `mitmweb-llm-better-view.user.js` and install it in Tampermonkey.
4. Adjust the userscript `@include` or `@match` rule so it matches your mitmweb address, for example `http://localhost:8081/*`.
5. Reload mitmweb and select an LLM API flow.

{{< figure src="/images/posts/mitmproxy-plus-tampermonkey-better-llm-dot-dot-dot-viewer/llm-better-view-set-include-rule.png" caption="<span class=\"figure-number\">Figure 3: </span>Set the script include rule" >}}

{{< figure src="/images/posts/mitmproxy-plus-tampermonkey-better-llm-dot-dot-dot-viewer/allow-user-script-for-tampermonkey.png" caption="<span class=\"figure-number\">Figure 4: </span>Allow user scripts for Tampermonkey" >}}

Once enabled, the script watches mitmweb's selected flow details and transforms supported LLM JSON payloads into a more readable view. Because this is a userscript, the rendering logic stays local to your browser. You can add provider-specific parsing, hide fields you do not care about, or highlight things like tool calls and token usage.

## Practical Notes {#practical-notes}

A few things are worth keeping in mind:

- Do this only for traffic you own or are allowed to inspect. A MITM proxy can expose secrets, cookies, API keys, and private prompts.
- Use a test API key when possible. Captured requests often include bearer tokens in headers.
- Keep `web_password` enabled if mitmweb is bound to `0.0.0.0`, or bind it to `127.0.0.1` if you only need local access.
- If a client refuses the proxy certificate, check whether it uses its own CA bundle or certificate pinning.
- For streaming responses, you may need to inspect both the raw event stream and the reconstructed body, depending on the provider and client.

## Advanced: Clean Up Captured History with an Addon {#advanced-clean-up-captured-history-with-an-addon}

mitmproxy supports Python [addons](https://docs.mitmproxy.org/stable/addons/overview/). An addon is useful when you want the capture list to stay focused while you work.

For example, this tiny addon keeps only flows whose host looks related to an LLM provider:

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

Run it with:

```shell
mitmweb \
  --mode regular@8080 \
  --web-port 8081 \
  --showhost \
  -s keep-llm-flows.py
```

That example only marks flows, but the same hook can export selected payloads, redact sensitive headers, or attach metadata that makes the mitmweb list easier to scan.

## Conclusion {#conclusion}

mitmproxy is already a strong network microscope. Tampermonkey turns its browser UI into a place you can shape for your own workflow.

For LLM debugging, that combination is especially handy: capture the real API traffic, keep the raw request and response available, and render the parts you actually read in a friendlier format. The result is lightweight, local, hackable, and independent of whichever SDK or UI you happen to be debugging that day.
