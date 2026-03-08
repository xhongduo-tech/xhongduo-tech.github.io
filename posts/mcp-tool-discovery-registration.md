## 核心结论

MCP 的工具发现机制，核心不是“客户端提前写死一组可调用函数”，而是把“工具定义”本身变成协议数据，在运行时由服务端暴露给客户端。这个流程至少分成两段：

1. 客户端调用 `tools/list`，读取当前服务端暴露的工具清单。
2. 客户端基于清单中的元数据，选择工具，再通过 `tools/call` 发起调用。

这里的“工具清单”不是只有名字。一个可被正确调用的 MCP 工具，最少要让客户端知道三类信息：

| 字段 | 含义 | 为什么必须有 |
| --- | --- | --- |
| `name` | 工具标识符 | 客户端调用时要明确点名调用哪个工具 |
| `description` | 工具用途说明 | 帮助人类或模型判断“这个工具大概做什么” |
| `inputSchema` | 输入参数结构 | 帮助客户端在调用前检查参数是否合法 |

其中 `inputSchema` 可以直接理解成“参数说明书”。它回答的是：这个工具接收什么参数、每个参数是什么类型、哪些字段必须提供、对象结构如何嵌套。没有这一层，客户端只能靠猜。

这套设计的真正价值，不在“工具能不能被调用”，而在“新工具上线时，客户端能不能不改代码就接入”。静态工具定义把接口信息编译进客户端，新增一个工具通常意味着修改客户端代码、重新发布版本、重新部署；MCP 则把工具元数据放在服务端，由客户端通过统一协议动态发现。只要客户端支持 `tools/list` 和 `tools/call`，理论上就能连接任意符合协议的工具服务。

它的代价也很明确。动态发现并不是免费能力，至少要承担四类额外成本：

| 成本 | 具体表现 |
| --- | --- |
| 发现开销 | 第一次使用前通常要多一次 `tools/list` |
| 校验开销 | 客户端要根据 `inputSchema` 检查参数 |
| 缓存开销 | 工具会变化时，要维护本地缓存与失效逻辑 |
| 权限开销 | 对高风险工具，往往还要在调用前做确认 |

因此，MCP 买到的是扩展性、统一性和可插拔性，付出的是一点延迟和一些实现复杂度。这个交换在“工具集合经常变化”的系统里通常是值得的，在“工具永远固定不变”的系统里则未必划算。

---

## 问题定义与边界

要讨论 MCP 的工具发现与注册机制，先把问题压缩成一句话：

**当服务端的工具集合会变化时，客户端怎样在“不提前知道所有工具”的前提下，仍然能正确发现并调用它们？**

这个问题看起来像接口设计，实际上同时涉及三个层面：

| 层面 | 关心的问题 |
| --- | --- |
| 发现 | 当前有哪些工具可用 |
| 理解 | 每个工具的作用和参数结构是什么 |
| 调用 | 用什么格式把参数发送给服务端并拿回结果 |

如果工具集合长期稳定，比如一个内部系统永远只有 `get_user` 和 `create_ticket` 两个动作，那么静态 SDK 往往更便宜。因为客户端在编译期就已经知道接口长什么样，运行时只需要直接调用本地封装，不必先问一遍“你现在有哪些工具”。

但一旦工具集合开始变化，静态方式就会迅速变重。常见变化包括：

| 变化类型 | 例子 |
| --- | --- |
| 新增工具 | 今天没有 `create_refund`，明天新增 |
| 删除工具 | 某个灰度工具下线 |
| 参数变更 | `user_id` 改成 `account_id`，或新增必填字段 |
| 租户差异 | A 租户看到 5 个工具，B 租户看到 20 个工具 |
| 权限差异 | 同一个服务，不同角色可调用的工具不同 |

这时客户端会遇到三个直接问题：

| 问题 | 静态工具定义 | MCP 动态发现 |
| --- | --- | --- |
| 新增工具 | 要改客户端代码 | 服务端更新后可被重新发现 |
| 参数变化 | 容易版本不一致 | 由 `inputSchema` 同步暴露 |
| 多服务接入 | 每个服务都要写适配层 | 统一走 `tools/list` / `tools/call` |

这里的边界也必须说清楚，否则很容易把 MCP 说成“万能协议”。

第一，MCP 解决的是**接口发现与调用格式统一**，不是“模型自动懂业务”。客户端即使拿到了完整工具列表，也不等于它就一定能选对工具、填对参数、理解副作用。协议只负责把“能力长什么样”表达清楚，不负责替你做业务判断。

第二，MCP 不等于零成本。动态发现至少多出一个列工具步骤，总耗时可以粗略写成：

$$
T_{dynamic}=T_{initialize}+T_{list}+T_{validate}+T_{call}
$$

静态绑定通常更接近：

$$
T_{static}=T_{initialize}+T_{call}
$$

因此两者的差值可以近似写成：

$$
\Delta T \approx T_{list}+T_{validate}
$$

在低延迟敏感系统里，这个差值不能忽略；在高扩展性优先的系统里，这个差值通常可以接受。

第三，MCP 也不替你做安全判断。协议可以告诉客户端“有一个叫 `delete_file` 的工具”，但它不会自动替你判断“这个调用该不该执行”。对删除、支付、写文件、发邮件之类的操作，客户端或宿主应用仍然应当保留人工确认、策略过滤和权限检查。

第四，MCP 不是“服务端一推送，客户端自动全懂”。`notifications/tools/list_changed` 的含义是“工具列表发生变化”，不是“下面直接附送一份完整新列表”。客户端仍然需要在收到通知后重新调用 `tools/list`。

如果用一个不失真的玩具类比，可以把它理解成“先看菜单，再下单”。你不能先说“给我来一个 A 套餐”，因为你还不知道这家店今天有没有 A 套餐，也不知道 A 套餐现在需要选哪些配菜。正确顺序是先看菜单，再确认规则，再下单。

---

## 核心机制与推导

MCP 建立在 JSON-RPC 2.0 之上。JSON-RPC 可以理解为：用一个固定 JSON 外壳，表达“我要调用某个远程方法，并带上一组参数”。它的基础消息结构很简单：

| 字段 | 含义 |
| --- | --- |
| `jsonrpc` | 协议版本，通常是 `"2.0"` |
| `method` | 要调用的方法名 |
| `params` | 方法参数 |
| `id` | 请求标识，用于把响应和请求对应起来 |

MCP 没有重新发明一套消息封装，而是在 JSON-RPC 这个壳里定义自己的方法名和语义，例如：

| MCP 方法/通知 | 作用 |
| --- | --- |
| `initialize` | 初始化连接，交换能力与协议信息 |
| `tools/list` | 获取当前工具列表 |
| `tools/call` | 调用某个具体工具 |
| `notifications/tools/list_changed` | 通知客户端工具列表已变化 |

完整流程通常可以拆成四步。

| 阶段 | 请求方 | 关键字段 | 作用 |
| --- | --- | --- | --- |
| 初始化 | Client | `initialize`、`capabilities` | 建立协议会话，交换支持能力 |
| 发现 | Client | `tools/list`、`cursor` | 获取工具清单，必要时分页 |
| 调用 | Client | `tools/call`、`name`、`arguments` | 按名字调用某个工具 |
| 刷新 | Server | `notifications/tools/list_changed` | 提醒客户端本地缓存应失效 |

为什么一定要拆成“先发现、再调用”两段，而不是允许客户端直接上来就 `tools/call`？

因为调用一个未知工具前，客户端必须先知道三件事：

1. 这个工具的**名字**是什么。
2. 这个工具的**用途**是什么。
3. 这个工具的**输入结构**是否合法。

其中第 3 点最关键，因为这决定了调用是否可验证。假设客户端没有 `inputSchema`，就只能依赖字符串描述去猜参数：

- 字段名可能猜错。
- 必填字段可能漏传。
- 类型可能不匹配。
- 嵌套对象结构可能不对。

一旦猜错，最轻是返回参数错误，最重是触发业务副作用错误。

这个逻辑可以写成一个很直接的依赖关系：

$$
\text{Correct Call} \Rightarrow \text{Known Name} \land \text{Known Semantics} \land \text{Known Input Shape}
$$

而在 MCP 里，这三项信息并不由客户端本地静态持有，而是由服务端在运行时提供。因此：

$$
\text{Dynamic Invocation} \Rightarrow \text{Discovery Before Call}
$$

继续展开，可以得到 MCP 调用正确性的最小链路：

$$
\text{tools/list} \rightarrow \text{schema validation} \rightarrow \text{tools/call}
$$

这就是为什么动态发现不是一个“可选优化”，而是协议思想的一部分。客户端如果跳过 `tools/list`，等于跳过了运行时接口协商。

这里再看 `tools/list_changed` 的设计。它不是直接把新工具清单推给客户端，而是发一个“列表变了”的通知。这个设计有两个工程上的好处：

| 设计选择 | 好处 |
| --- | --- |
| 只推送“已变化”信号 | 通知负载轻，不必每次都发送大对象 |
| 真正数据仍由 `tools/list` 拉取 | 请求响应路径清晰，更容易重试、分页和缓存 |

也就是说，MCP 把“变更通知”和“数据获取”拆开了。通知负责告诉你“缓存可能过期了”，真正的数据仍然通过幂等的读取接口来拿。这比“服务端直接推一整份新菜单”更容易实现一致性。

再看“注册”这件事。在工程实现里，所谓工具注册，通常不是指向某个中心注册平台报备，而是指**服务端把某个处理函数与一份工具元数据绑定起来，并让它出现在 `tools/list` 结果里**。因此“注册”本质上是服务端内部动作，“发现”则是客户端对外可见动作。

可以把两者关系写成：

$$
\text{Registration on Server} \Rightarrow \text{Visibility in tools/list}
$$

但反过来不一定成立。原因很简单：某个工具虽然在服务端代码里已经注册了，也可能因为权限、租户隔离、灰度策略或环境配置，不出现在当前客户端看到的 `tools/list` 结果里。

真实工程场景里，这一点很常见。比如一个风控 Agent 同时连接多个服务：

| 服务 | 暴露工具 |
| --- | --- |
| 账户服务 | `query_balance` |
| 风控服务 | `freeze_account` |
| 工单服务 | `risk_review/create_case` |

如果采用静态 SDK，客户端需要提前集成三套接口；如果采用 MCP，客户端只需要支持统一协议，然后在连接建立后读取当前可见工具。某个灰度工具上线时，服务端更新注册表，随后发出 `notifications/tools/list_changed`，客户端重新拉取即可。

---

## 代码实现

下面给出一个可直接运行的最小 Python 示例，目标不是完整复刻官方 MCP SDK，而是把“注册、发现、校验、调用、变更通知”这条主链路完整串起来。

这个例子修复了常见示例里的两个问题：

1. 不只是“能跑通 happy path”，而是包含失败分支和缓存刷新分支。
2. 不是只做字符串拼接，而是实现了最小可验证的 Schema 校验逻辑。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


JsonDict = Dict[str, Any]
ToolHandler = Callable[[JsonDict], Any]


class MCPError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Tool:
    name: str
    description: str
    input_schema: JsonDict
    handler: ToolHandler


class MCPServer:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._version = 0

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: JsonDict,
        handler: ToolHandler,
    ) -> None:
        self._tools[name] = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self._version += 1

    def unregister_tool(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            self._version += 1

    @property
    def version(self) -> int:
        return self._version

    def handle_request(self, request: JsonDict) -> JsonDict:
        method = request.get("method")
        request_id = request.get("id")

        try:
            if method == "tools/list":
                params = request.get("params", {})
                cursor = params.get("cursor")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": self._tools_list(cursor=cursor),
                }

            if method == "tools/call":
                params = request.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})
                result = self._tools_call(name=name, arguments=arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result,
                }

            raise MCPError(-32601, f"Method not found: {method}")

        except MCPError as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            }

    def tools_list_changed_notification(self) -> JsonDict:
        return {
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed",
            "params": {
                "version": self._version,
            },
        }

    def _tools_list(self, cursor: Optional[int] = None, page_size: int = 50) -> JsonDict:
        tool_values = list(self._tools.values())
        start = int(cursor or 0)
        end = start + page_size
        page = tool_values[start:end]

        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in page
            ],
            "nextCursor": str(end) if end < len(tool_values) else None,
            "version": self._version,
        }

    def _tools_call(self, name: Any, arguments: JsonDict) -> JsonDict:
        if not isinstance(name, str) or not name:
            raise MCPError(-32602, "Invalid params: tool name is required")

        tool = self._tools.get(name)
        if tool is None:
            raise MCPError(-32601, f"Unknown tool: {name}")

        validate_input(arguments, tool.input_schema)

        try:
            output = tool.handler(arguments)
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "isError": True,
            }

        return {
            "content": [{"type": "text", "text": str(output)}],
            "isError": False,
        }


class MCPClient:
    def __init__(self, server: MCPServer) -> None:
        self.server = server
        self.tool_cache: Dict[str, JsonDict] = {}
        self.cached_version: Optional[int] = None
        self._next_id = 1

    def refresh_tools(self) -> None:
        tools: List[JsonDict] = []
        cursor: Optional[str] = None

        while True:
            response = self._request(
                "tools/list",
                {"cursor": cursor} if cursor is not None else {},
            )
            result = response["result"]
            tools.extend(result["tools"])
            cursor = result.get("nextCursor")
            self.cached_version = result["version"]
            if cursor is None:
                break

        self.tool_cache = {tool["name"]: tool for tool in tools}

    def on_notification(self, notification: JsonDict) -> None:
        if notification.get("method") == "notifications/tools/list_changed":
            new_version = notification.get("params", {}).get("version")
            if new_version != self.cached_version:
                self.refresh_tools()

    def call_tool(self, name: str, arguments: JsonDict) -> JsonDict:
        tool = self.tool_cache.get(name)
        if tool is None:
            raise RuntimeError(f"Tool not found in local cache: {name}")

        validate_input(arguments, tool["inputSchema"])

        response = self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )

        if "error" in response:
            raise RuntimeError(response["error"]["message"])

        return response["result"]

    def _request(self, method: str, params: JsonDict) -> JsonDict:
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        self._next_id += 1
        return self.server.handle_request(request)


def validate_input(arguments: JsonDict, schema: JsonDict) -> None:
    if schema.get("type") != "object":
        raise MCPError(-32602, "Only object input schema is supported in this demo")

    if not isinstance(arguments, dict):
        raise MCPError(-32602, "Invalid params: arguments must be an object")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        if field not in arguments:
            raise MCPError(-32602, f"Missing required field: {field}")

    for field, value in arguments.items():
        spec = properties.get(field)
        if spec is None:
            continue  # 这个最小示例允许额外字段存在

        expected_type = spec.get("type")
        if expected_type == "string" and not isinstance(value, str):
            raise MCPError(-32602, f"Field '{field}' must be string")
        if expected_type == "integer" and not isinstance(value, int):
            raise MCPError(-32602, f"Field '{field}' must be integer")
        if expected_type == "number" and not isinstance(value, (int, float)):
            raise MCPError(-32602, f"Field '{field}' must be number")
        if expected_type == "boolean" and not isinstance(value, bool):
            raise MCPError(-32602, f"Field '{field}' must be boolean")


def main() -> None:
    server = MCPServer()

    server.register_tool(
        name="get_weather",
        description="Return a fake weather report by city name.",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["location"],
        },
        handler=lambda args: f"{args['location']} weather for {args.get('days', 1)} day(s): sunny",
    )

    client = MCPClient(server)
    client.refresh_tools()

    ok = client.call_tool("get_weather", {"location": "Tokyo", "days": 2})
    assert ok["isError"] is False
    assert ok["content"][0]["text"] == "Tokyo weather for 2 day(s): sunny"

    try:
        client.call_tool("get_weather", {})
    except Exception as exc:
        assert "Missing required field: location" in str(exc)

    server.register_tool(
        name="sum_numbers",
        description="Add two integers.",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=lambda args: args["a"] + args["b"],
    )

    client.on_notification(server.tools_list_changed_notification())

    added = client.call_tool("sum_numbers", {"a": 7, "b": 8})
    assert added["content"][0]["text"] == "15"

    print("All demo checks passed.")


if __name__ == "__main__":
    main()
```

如果把这段代码保存为 `mcp_demo.py` 并运行：

```bash
python3 mcp_demo.py
```

预期输出是：

```text
All demo checks passed.
```

这个最小示例对应了 MCP 机制中的五个核心动作：

| 示例动作 | 对应协议思想 |
| --- | --- |
| `register_tool(...)` | 服务端注册工具 |
| `refresh_tools()` | 客户端执行 `tools/list`，建立本地缓存 |
| `validate_input(...)` | 客户端或服务端依据 Schema 做输入检查 |
| `call_tool(...)` | 客户端执行 `tools/call` |
| `tools_list_changed_notification()` | 服务端通知客户端缓存失效 |

这个示例仍然是“最小版”，它故意省略了一些真实实现里常见的能力，例如：

| 省略项 | 为什么省略 | 真实工程里怎么做 |
| --- | --- | --- |
| 完整 JSON Schema | 手写会把示例淹没 | 使用正式 Schema 校验器 |
| `initialize` 握手 | 这里聚焦工具发现主线 | 建立连接时先协商版本和能力 |
| 传输层 | 这里直接函数调用模拟请求响应 | 真实环境可能是 stdio、HTTP、WebSocket |
| 权限系统 | 示例目标是解释协议骨架 | 真实系统应在调用前后做权限检查 |
| 输出 Schema | 文章重点是输入发现与调用 | 工程上可继续补充输出约束 |

对新手来说，最容易混淆的是“注册”和“发现”的方向：

| 动作 | 发生在谁这一侧 | 本质 |
| --- | --- | --- |
| 注册工具 | 服务端 | 把处理逻辑和元数据加入工具表 |
| 发现工具 | 客户端 | 读取当前工具表的可见视图 |
| 调用工具 | 客户端发起，服务端执行 | 按名字和参数执行具体动作 |

因此，注册不是客户端做的事，客户端做的是发现和调用。客户端不需要知道服务端内部怎么维护注册表，它只需要知道：`tools/list` 能读到什么，`tools/call` 能调用什么。

---

## 工程权衡与常见坑

动态发现最大的收益是扩展性，最大的成本是状态协调。只看 happy path 时，MCP 会显得非常简单；一旦进入真实工程，问题通常都出在缓存、校验、权限和描述语义上。

先看最常见的一组坑：

| 常见坑 | 现象 | 根因 | 处理方式 |
| --- | --- | --- | --- |
| 缓存过期 | 客户端看不到新工具，或继续调用已下线工具 | 本地缓存未及时刷新 | 监听 `notifications/tools/list_changed`，收到后重新 `tools/list` |
| 不做参数校验 | 服务端返回 `Invalid params`，或业务层直接报错 | 客户端只看描述，不看 Schema | 按 `inputSchema` 在调用前做校验 |
| 过度相信描述文本 | 模型误判工具用途，调用危险操作 | `description` 不是强约束 | 把描述视为辅助信息，不作为安全依据 |
| 忽略权限确认 | 删除、支付、写文件等操作风险高 | 协议统一不等于自动安全 | 在 UI 或代理层增加人工批准 |
| 工具名不稳定 | 客户端缓存命中失败或逻辑混乱 | 服务端频繁改名 | 保持名字稳定，语义变化走版本或新工具名 |
| 分页没处理 | 只能看到前 N 个工具 | 客户端默认以为列表一次返回完 | 正确处理 `cursor` / `nextCursor` |
| 多租户隔离不清 | A 用户看到 B 用户工具 | 服务端可见性控制没做好 | `tools/list` 结果必须按会话和权限裁剪 |

这里有一个非常容易被忽略的工程事实：**动态发现不等于每次调用前都重新 `tools/list`。**  
如果真这么做，延迟和负载会明显变差。更合理的常见策略是：

1. 初始化完成后拉一次工具列表。
2. 本地缓存工具元数据。
3. 收到 `notifications/tools/list_changed` 后失效缓存并重新拉取。
4. 只有在显式刷新、错误恢复或重连时，才主动重新发现。

这个策略本质上是在平衡两个目标：

$$
\text{Freshness} \leftrightarrow \text{Latency}
$$

如果把“工具信息的新鲜度”记作 `F`，把“每次调用额外延迟”记作 `L`，那么几种常见策略可以粗略比较为：

| 策略 | 新鲜度 `F` | 额外延迟 `L` | 适用情况 |
| --- | --- | --- | --- |
| 每次调用前都 `tools/list` | 高 | 高 | 工具变化极频繁且调用量低 |
| 初始化拉一次 + 通知刷新 | 中高 | 低 | 最常见，也最均衡 |
| 永久缓存不刷新 | 低 | 低 | 只适合工具集合几乎不变的场景 |

另一个典型误区，是把 `description` 当成“机器可执行规则”。它不是。  
`description` 的作用是帮助人类或模型理解工具用途，像是：

- “根据城市名获取天气”
- “创建风控复核工单”
- “删除指定目录下的文件”

这些文本说明能帮助选择工具，但不能替代参数约束，也不能替代权限控制。真正可验证的是结构化约束，例如 `inputSchema`；真正可审计的是策略系统；真正可确认的是人工审批或显式授权。

也就是说：

$$
\text{description} \neq \text{validation rule}
$$

更准确地说，三者职责不同：

| 机制 | 主要作用 |
| --- | --- |
| `description` | 解释“它大概做什么” |
| `inputSchema` | 约束“参数必须长什么样” |
| 权限/审批系统 | 控制“这次是否允许执行” |

如果把这三层混为一谈，系统就会出现两个典型后果：

1. 可用性差：模型能看到工具，但总是填错参数。
2. 安全性差：模型能猜对参数，却越权执行危险动作。

对于新手来说，还要特别注意“Schema 校验做在客户端还是服务端”这个问题。正确答案不是二选一，而是两边都应该做，只是目标不同：

| 位置 | 作用 |
| --- | --- |
| 客户端校验 | 尽早拦截明显错误，减少无效请求 |
| 服务端校验 | 作为最终防线，防止绕过客户端的非法输入 |

因此，一个更稳妥的工程公式是：

$$
\text{Safe Call} = \text{Client Validation} + \text{Server Validation} + \text{Permission Check}
$$

少任何一层，都可能在真实系统里出问题。

---

## 替代方案与适用边界

MCP 的 `tools/list` + `tools/call` 并不是所有场景下的唯一答案。它适合的是“工具集合变化频繁、跨服务接入多、希望统一协议”的环境；如果场景前提不成立，其他方案可能更便宜。

第一种替代方案是静态 SDK。  
如果工具固定、客户端数量少、团队可以同步升级，那么静态 SDK 仍然非常合理。它的优势很直接：

| 优势 | 原因 |
| --- | --- |
| 延迟低 | 没有额外发现请求 |
| 类型清晰 | 可在编译期生成类型和接口 |
| 调试简单 | 接口变化都在代码层显式体现 |

它的问题在于，一旦服务端工具频繁变化，客户端升级成本会快速上升。

第二种替代方案，是“单入口工具”模式。  
也就是不把多个动作暴露为多个工具，而是暴露一个总入口，比如 `execute_action`，再把真正动作名塞进参数里。这样做的好处是工具枚举很小，表面上减少了发现成本；但它把原本分散的复杂度，全都压进一个参数对象里。

例如：

```json
{
  "name": "execute_action",
  "arguments": {
    "action": "freeze_account",
    "payload": {
      "account_id": "A123"
    }
  }
}
```

这种设计的问题不是不能用，而是它会带来新的代价：

| 问题 | 原因 |
| --- | --- |
| 工具语义变重 | 一个工具承载了很多完全不同的动作 |
| Schema 变复杂 | 参数里又套动作分发逻辑 |
| 权限边界变模糊 | 到底是给了一个工具权限，还是给了一组动作权限 |
| 可观测性变差 | 日志层面不容易直接看出执行了什么动作 |

因此，这种模式更适合高度定制的平台系统，而不是通用工具生态。

如果把几种方案并排比较，可以得到一个更清晰的选择表：

| 方案 | 优势 | 劣势 | 适用场景 |
| --- | --- | --- | --- |
| 静态 SDK | 延迟低、类型清晰、实现简单 | 扩展差、升级重、服务耦合高 | 工具少且长期稳定 |
| MCP 动态发现 | 扩展强、跨服务统一、运行时可插拔 | 多一次发现、要处理缓存和校验 | 多工具、常变化、插件化场景 |
| 单入口工具 | 调用轮次少、外表简单 | 单工具语义过重，安全边界更难定义 | 高度定制的平台型系统 |

还可以进一步用一个简单决策规则来判断是否值得上 MCP：

| 条件 | 倾向选择 |
| --- | --- |
| 工具经常新增、下线、灰度发布 | MCP |
| 不同租户或角色看到的工具不同 | MCP |
| 客户端无法频繁发版 | MCP |
| 工具集合长期固定不变 | 静态 SDK |
| 调用延迟极其敏感 | 静态 SDK 或更窄的定制接口 |
| 动作极其复杂且统一封装更重要 | 单入口工具或工作流型接口 |

所以，`tools/list` + `tools/call` 不是“唯一正确答案”，而是“当工具集合不稳定时，通常更划算的答案”。  
它解决的是动态接入和协议统一问题，不是所有工具系统都必须追求的目标。

---

## 参考资料

- Model Context Protocol Specification, Overview, 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25
- Model Context Protocol Tools, 2025-06-18: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
- Model Context Protocol Lifecycle, 2025-03-26: https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle
- Model Context Protocol Pagination, 2025-03-26: https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/pagination
- JSON-RPC 2.0 Specification: https://www.jsonrpc.org/specification
- JSON Schema: Core Concepts and Validation Overview: https://json-schema.org/
