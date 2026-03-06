# Charm TUI Architecture — Visual Reference

Last updated: 2026-03-05

---

## 1. End-to-End System Overview

How the user's keypress travels from terminal to LLM and back.

```mermaid
flowchart TB
    subgraph Terminal["User Terminal"]
        User["User Input<br/>(keypress / slash command)"]
    end

    subgraph GoBinary["Go Binary — lobster-tui"]
        Main["main.go:17<br/>runChat()"]
        Run["run.go:19<br/>chat.Run()"]
        Model["model.go:207<br/>NewModel()"]
        Update["model.go:284<br/>Update() loop"]
        Handler["protocol.Handler<br/>StartReadLoop()"]
        Views["views.go<br/>renderHeader / renderMessage"]
        Theme["theme.go<br/>Lipgloss Styles"]

        Main --> Run --> Model --> Update
        Update --> Views
        Views --> Theme
        Update <--> Handler
    end

    subgraph PythonProcess["Python Process"]
        Launcher["go_tui_launcher.py:540<br/>launch_go_tui_chat()"]
        Bridge["_LightBridge:138<br/>send() / recv_event()"]
        EventLoop["_go_tui_event_loop:435<br/>dispatch by msg_type"]
        HandleQuery["_handle_user_query:458<br/>client.query(stream=True)"]
        HandleSlash["_handle_slash_command:503<br/>ProtocolOutputAdapter"]
        HandleComp["_handle_completion_request"]

        Launcher --> Bridge
        Launcher --> EventLoop
        EventLoop --> HandleQuery
        EventLoop --> HandleSlash
        EventLoop --> HandleComp
    end

    subgraph CoreEngine["Lobster Core Engine"]
        Client["client.py:184<br/>query() / _stream_query()"]
        Graph["graph.py<br/>create_bioinformatics_graph()"]
        Hook["context_management.py:235<br/>pre_model_hook"]
        Supervisor["supervisor agent"]
        Specialists["specialist agents<br/>(21 across 10 packages)"]
        Store["InMemoryStore<br/>delegation results"]
    end

    User --> Update
    Handler <-->|"JSON-lines<br/>over FD pipes"| Bridge
    HandleQuery --> Client
    Client --> Graph
    Graph --> Hook
    Hook --> Supervisor
    Supervisor --> Specialists
    Specialists -.->|"dual-write"| Store
    Client -->|"content_delta<br/>agent_change<br/>complete"| HandleQuery
    HandleQuery -->|"bridge.send()"| Bridge
    HandleSlash -->|"OutputAdapter.print*()"| Bridge
```

---

## 2. IPC Protocol — Message Flow

All 25 message types and their direction across the FD pipe boundary.

```mermaid
flowchart LR
    subgraph GoToPython["Go --> Python"]
        direction TB
        GI1["input<br/>(free-text query)"]
        GI2["slash_command<br/>(command + args)"]
        GI3["completion_request<br/>(prefix + context)"]
        GI4["form_response"]
        GI5["confirm_response"]
        GI6["select_response"]
        GI7["cancel"]
        GI8["quit"]
        GI9["resize"]
        GI10["handshake<br/>(version + terminal size)"]
    end

    subgraph Pipe["FD Pipes<br/>JSON-lines"]
        direction TB
        P2G["parent-to-go (p2g)"]
        G2P["go-to-parent (g2p)"]
    end

    subgraph PythonToGo["Python --> Go"]
        direction TB
        PO1["text (streamed chunk)"]
        PO2["markdown (rendered block)"]
        PO3["code (language + content)"]
        PO4["table (headers + rows)"]
        PO5["form (title + fields)"]
        PO6["confirm (question)"]
        PO7["select (options)"]
        PO8["progress (pct + label)"]
        PO9["alert (level + message)"]
        PO10["spinner (active flag)"]
        PO11["status (status-bar text)"]
        PO12["clear"]
        PO13["done (end-of-turn)"]
        PO14["agent_transition<br/>(from/to/reason)"]
        PO15["modality_loaded"]
        PO16["tool_execution<br/>(start/finish/error)"]
        PO17["suspend / resume"]
        PO18["ready"]
        PO19["heartbeat"]
    end

    GoToython -- "g2p pipe" --> Pipe
    Pipe -- "p2g pipe" --> PythonToGo
```

---

## 3. Slash Command Routing

How a slash command travels from Go through Python dispatch to protocol output.

```mermaid
flowchart TD
    GoInput["Go: user types /workspace list"]
    GoModel["model.go:737-763<br/>slash dispatch switch"]

    GoModel -->|"/help, /clear,<br/>/exit, /data"| GoNative["Go-native handler<br/>(no Python)"]
    GoModel -->|"/dashboard,<br/>/tree (degraded)"| GoDegraded["Go sends degraded<br/>fallback message"]
    GoModel -->|"everything else"| GoBridge["Send slash_command<br/>via protocol"]

    GoBridge --> EventLoop["_go_tui_event_loop:447<br/>msg_type == slash_command"]
    EventLoop --> HandleSlash["_handle_slash_command:503"]

    HandleSlash --> Spinner["bridge.send spinner ON"]
    HandleSlash --> ImportOA["Import ProtocolOutputAdapter<br/>(lazy, ~1-2s first time)"]
    ImportOA --> CreateOA["output = ProtocolOutputAdapter(bridge.send)"]
    CreateOA --> ImportExec["Import _execute_command<br/>(heavy module)"]
    ImportExec --> Dispatch["_execute_command(full_cmd,<br/>client, output=output)"]

    Dispatch --> Router{"slash_commands.py:1557<br/>command router"}
    Router -->|"/workspace *"| WS["workspace handler"]
    Router -->|"/config *"| CFG["config handler"]
    Router -->|"/metadata *"| META["metadata handler"]
    Router -->|"/queue *"| QUEUE["queue handler"]
    Router -->|"/pipeline *"| PIPE["pipeline handler"]
    Router -->|"/tokens"| TOK["tokens handler"]
    Router -->|"30+ commands"| OTHER["other handlers"]

    WS --> OA["OutputAdapter methods"]
    CFG --> OA
    META --> OA
    QUEUE --> OA
    PIPE --> OA
    TOK --> OA
    OTHER --> OA

    OA -->|"output.print()"| ProtoText["bridge.send('text') or<br/>bridge.send('alert')"]
    OA -->|"output.print_table()"| ProtoTable["bridge.send('table')"]
    OA -->|"output.print_code_block()"| ProtoCode["bridge.send('code')"]

    HandleSlash --> SpinnerOff["bridge.send spinner OFF<br/>+ bridge.send done"]

    GoInput --> GoModel
```

---

## 4. OutputAdapter Hierarchy

The 4 frontend adapters and their rendering targets.

```mermaid
classDiagram
    class OutputAdapter {
        <<abstract>>
        +print(message, style)
        +print_table(table_data)
        +print_code_block(code, language)
        +confirm(question) bool
        +prompt(question, default) str
    }

    class ConsoleOutputAdapter {
        -console: Rich Console
        +print() Rich Panel/Text
        +print_table() Rich Table
        +confirm() Rich Confirm
        +prompt() Rich Prompt
        +print_code_block() Rich Syntax
    }

    class JsonOutputAdapter {
        -messages: list
        -tables: list
        +print() append to messages
        +print_table() append to tables
        +confirm() return False
        +prompt() return default
    }

    class DashboardOutputAdapter {
        -results_display: ResultsDisplay
        +print() results_display.append
        +print_table() results_display.append
    }

    class ProtocolOutputAdapter {
        -_send: Callable (bridge.send)
        +print() send text or alert
        +print_table() send table msg
        +print_code_block() send code msg
        +confirm() return False (non-interactive)
        +prompt() return default (non-interactive)
    }

    OutputAdapter <|-- ConsoleOutputAdapter : "Classic CLI (Rich)"
    OutputAdapter <|-- JsonOutputAdapter : "JSON / automation"
    OutputAdapter <|-- DashboardOutputAdapter : "Cloud UI"
    OutputAdapter <|-- ProtocolOutputAdapter : "Go TUI (protocol)"
```

---

## 5. Streaming Query Pipeline

How a natural language query flows through the LangGraph agent system back to the TUI.

```mermaid
sequenceDiagram
    participant Go as Go TUI
    participant Bridge as _LightBridge
    participant Loop as _go_tui_event_loop
    participant Client as LobsterClient
    participant Graph as LangGraph
    participant Hook as pre_model_hook
    participant Sup as Supervisor
    participant Spec as Specialist Agent
    participant Store as InMemoryStore

    Go->>Bridge: {"type":"input","payload":{"content":"analyze RNA-seq"}}
    Bridge->>Loop: recv_event() returns input msg
    Loop->>Loop: _handle_user_query()
    Loop->>Bridge: send("spinner", active=true)
    Loop->>Client: query(text, stream=True)
    Client->>Graph: graph.stream(input, config,<br/>stream_mode=["messages","updates"],<br/>subgraphs=True)

    Note over Graph,Hook: Each supervisor LLM call
    Graph->>Hook: pre_model_hook(state, store)
    Hook->>Hook: trim_messages(strategy="last")
    Hook->>Hook: _fix_orphaned_tool_messages()
    Hook->>Store: store.search() for key index
    Hook->>Hook: inject key index into SystemMessage
    Hook-->>Graph: {llm_input_messages, store_keys}

    Graph->>Sup: invoke supervisor LLM
    Sup->>Spec: delegate to specialist
    Spec-->>Store: store_delegation_result() [dual-write]
    Spec-->>Graph: return result

    Graph-->>Client: yield (namespace, "messages", AIMessageChunk)
    Client-->>Loop: yield {"type":"content_delta","delta":"..."}
    Loop->>Bridge: send("text", {"content": delta})
    Bridge->>Go: render streamed text

    Graph-->>Client: yield (namespace, "messages", final)
    Client-->>Loop: yield {"type":"agent_change","agent":"research_agent"}
    Loop->>Bridge: send("agent_transition", {...})
    Bridge->>Go: show agent badge

    Client-->>Loop: yield {"type":"complete", token_usage, duration}
    Loop->>Bridge: send("done") + send("status", usage_text)
    Loop->>Bridge: send("spinner", active=false)
    Bridge->>Go: end-of-turn, show status bar
```

---

## 6. Launch Sequence

The 7-phase startup that achieves zero-latency TUI appearance.

```mermaid
sequenceDiagram
    participant CLI as lobster chat --ui go
    participant Py as Python Launcher
    participant Go as Go Binary
    participant Bridge as _LightBridge

    CLI->>Py: launch_go_tui_chat()

    rect rgb(230, 245, 230)
        Note over Py,Go: Phase 1: Spawn (line 577-622)
        Py->>Py: Create FD pipes (p2g, g2p)
        Py->>Go: Popen(lobster-tui chat --proto-fd-in --proto-fd-out)
        Note over Go: Go TUI visible immediately (~50ms)
        Py->>Bridge: _LightBridge(proc, g2p_r, p2g_w)
    end

    rect rgb(230, 235, 250)
        Note over Py,Go: Phase 2: Handshake (line 624-633)
        Go->>Bridge: {"type":"handshake","payload":{"version":1,"width":120,"height":40}}
        Bridge->>Py: recv_event(timeout=5.0)
    end

    rect rgb(250, 245, 230)
        Note over Py,Go: Phase 3: Loading spinner (line 635-640)
        Py->>Bridge: send("spinner", {"active":true,"label":"Initializing agents"})
        Bridge->>Go: Show spinner
    end

    rect rgb(245, 230, 230)
        Note over Py,Go: Phase 4: Heartbeat (line 642-652)
        Py->>Py: Start heartbeat thread (every 3s)
        loop Every 3 seconds
            Py->>Bridge: send("heartbeat", {"ts": ...})
        end
    end

    rect rgb(240, 240, 250)
        Note over Py,Go: Phase 5: Heavy imports (line 654-698)
        Py->>Py: Import session_infra (slow, ~1-2s)
        Py->>Py: init_client() -> create agent graph
        Py->>Py: Wire ProtocolCallbackHandler
        Py->>Bridge: send("status", {"text": "Provider: openai"})
    end

    rect rgb(230, 250, 240)
        Note over Py,Go: Phase 6: Signal ready (line 711-729)
        Py->>Py: Stop heartbeat, restore stdio
        Py->>Bridge: send("spinner", {"active": false})
        Py->>Bridge: send("ready", {})
        Bridge->>Go: Unlock input field
    end

    rect rgb(250, 250, 230)
        Note over Py,Go: Phase 7: Event loop (line 732-734)
        Py->>Py: _go_tui_event_loop(bridge, client)
        Note over Py: Blocks on recv_event() forever
    end
```

---

## 7. Context Management & Store Architecture

How the pre_model_hook prevents context overflow while preserving agent results.

```mermaid
flowchart TD
    subgraph PerLLMCall["Every Supervisor LLM Call"]
        Hook["pre_model_hook(state, store)"]
        ReadStore["store.search(AGENT_RESULTS_NAMESPACE)"]
        BuildKeys["Build store_keys dict<br/>{agent_name: store_key}"]
        Trim["trim_messages()<br/>strategy=last, include_system=True"]
        FixOrphans["_fix_orphaned_tool_messages()<br/>strip ToolMsg without AIMsg partner"]
        InjectIndex["Inject key index into SystemMessage<br/>(LLM always sees available keys)"]
        Return["Return {llm_input_messages, store_keys}"]

        Hook --> ReadStore --> BuildKeys --> Trim --> FixOrphans --> InjectIndex --> Return
    end

    subgraph BudgetCalc["Budget Resolution"]
        CW["context_window<br/>(from provider config)"]
        Reserve["output_reserve = CW * 0.15"]
        ToolSchema["tool_schema_tokens<br/>(measured from tool definitions)"]
        Budget["budget = CW - reserve - tools<br/>floor at 4096"]

        CW --> Reserve --> Budget
        ToolSchema --> Budget
    end

    subgraph DualWrite["Delegation Dual-Write"]
        AgentInvoke["_invoke_and_store()"]
        Result["agent result (full text)"]
        StateMsg["Append to state.messages"]
        StoreWrite["store_delegation_result()<br/>write to InMemoryStore"]
        Retrieve["retrieve_agent_result tool<br/>(20K char cap, AQUADIF UTILITY)"]

        AgentInvoke --> Result
        Result --> StateMsg
        Result --> StoreWrite
        StoreWrite -.->|"on-demand"| Retrieve
    end

    Budget --> Hook
    DualWrite -.->|"keys survive trimming"| ReadStore
```

---

## 8. Parity Matrix — Command Classification

Current state of Go TUI command coverage (30 commands tracked).

```mermaid
pie title Command Parity Status (30 total)
    "Parity (24)" : 24
    "Degraded-Explicit (6)" : 6
    "Blocked (0)" : 0
```

```mermaid
flowchart LR
    subgraph Native["Go-Native (no Python)"]
        N1["/help"]
        N2["/clear"]
        N3["/exit"]
        N4["/data"]
    end

    subgraph Bridged["Bridged via Protocol (20 parity)"]
        B1["/session"]
        B2["/status"]
        B3["/tokens"]
        B4["/workspace *"]
        B5["/queue *"]
        B6["/metadata *"]
        B7["/config *"]
        B8["/pipeline *"]
        B9["/save"]
        B10["/restore"]
        B11["/read"]
        B12["/open"]
        B13["/reset"]
        B14["/files"]
        B15["/export"]
        B16["/plots"]
        B17["/plot"]
        B18["/describe"]
        B19["/modalities"]
        B20["/vector-search"]
    end

    subgraph Degraded["Degraded-Explicit (6)"]
        D1["/tree --> /files fallback"]
        D2["/dashboard --> classic only"]
        D3["/status-panel --> /status"]
        D4["/workspace-info --> /workspace"]
        D5["/analysis-dash --> /plots+/metadata"]
        D6["/progress --> explicit fallback"]
    end
```

---

## 9. Test Infrastructure

How tests validate protocol correctness and visual parity.

```mermaid
flowchart TD
    subgraph GoldenTests["Golden Transcript Tests"]
        GT["test_slash_command_golden_transcripts.py"]
        DummyClient["_DummyClient<br/>(deterministic mock data)"]
        ProtoOA["ProtocolOutputAdapter<br/>(captures events)"]
        ExecCmd["_execute_command(cmd, client, output)"]
        Compare["Compare actual vs golden JSON"]
        GoldenDir["tests/golden/slash_commands/*.json<br/>(9 files: help, session, status,<br/>tokens, workspace_list, queue,<br/>metadata_publications,<br/>config_provider, pipeline_list)"]

        GT --> DummyClient --> ProtoOA --> ExecCmd --> Compare
        Compare <-->|"assert ==" | GoldenDir
    end

    subgraph SmokeTests["Protocol Smoke Tests"]
        ST["test_go_tui_protocol_smoke_script.py"]
        Script["scripts/go_tui_protocol_smoke.py"]
        SubProc["subprocess.run(script)"]
        Assert["assert exit_code == 0<br/>assert 'GO_TUI_PROTOCOL_SMOKE_OK'"]

        ST --> SubProc --> Script --> Assert
    end

    subgraph RegressionTests["Regression Tests"]
        RT1["test_go_tui_launcher_completions.py<br/>(completion_request/response)"]
        RT2["test_slash_commands_go_tui_regressions.py<br/>(/open, /restore, /config, /save,<br/>/clear, /exit protocol safety)"]
    end

    subgraph GoTests["Go Unit Tests"]
        GoT["go test ./internal/chat<br/>go test ./internal/protocol"]
        ModelTest["model_test.go<br/>(/exit confirm flow)"]

        GoT --> ModelTest
    end

    subgraph Update["Golden Update Flow"]
        Env["LOBSTER_UPDATE_GOLDENS=1"]
        Regen["pytest regenerates golden JSON"]
        Review["Developer reviews diff"]
        Commit["Commit only intentional changes"]

        Env --> Regen --> Review --> Commit
    end
```

---

## 10. Phase Roadmap

Current and planned phases for Go TUI completion.

```mermaid
gantt
    title Charm TUI Implementation Phases
    dateFormat YYYY-MM-DD

    section Foundation
    Phase 0 - Init Wizard           :done, p0, 2026-03-04, 1d
    Phase 1 - Protocol + Streaming  :done, p1, after p0, 1d
    Phase 2 - Rendering + UX       :done, p2, after p1, 1d
    Phase 3 - Protocol Handlers    :done, p3, after p2, 1d
    Phase 4 - Slash + Autocomplete :done, p4, after p3, 1d
    Phase 4.2 - Parity Hardening   :done, p42, after p4, 1d

    section Current
    Phase 5 - Rich Parity Migration :active, p5, after p42, 3d

    section Future
    Phase 6A - HITL Python Infra    :p6a, after p5, 2d
    Phase 6B - HITL Protocol Ext    :p6b, after p6a, 2d
    Phase 6C - BioCharm Components  :p6c, after p6b, 3d
    Phase 7 - Distribution          :p7, after p6c, 2d
    Phase 8 - SSH + Cloud           :p8, after p7, 3d
```

---

## File Index (Key Files)

| Layer | File | Purpose |
|-------|------|---------|
| Go entry | `lobster-tui/cmd/lobster-tui/main.go:17` | CLI dispatcher, `runChat()` |
| Go model | `lobster-tui/internal/chat/model.go:207` | BubbleTea Model, Update loop, slash routing |
| Go views | `lobster-tui/internal/chat/views.go` | Header, message, summary renderers |
| Go run | `lobster-tui/internal/chat/run.go` | `Run()` entry, protocol handler setup |
| Go protocol | `lobster-tui/internal/protocol/types.go` | 25 message types + payload structs |
| Go theme | `lobster-tui/internal/theme/theme.go` | Colors, 30+ Lipgloss Styles |
| Py launcher | `lobster/cli_internal/go_tui_launcher.py:540` | 7-phase launch, bridge, event loop |
| Py adapter | `lobster/cli_internal/commands/output_adapter.py:331` | ProtocolOutputAdapter (bridge.send wrapper) |
| Py dispatch | `lobster/cli_internal/commands/heavy/slash_commands.py:1557` | `_execute_command()` router |
| Py client | `lobster/core/client.py:322` | `_stream_query()` generator |
| Py context | `lobster/agents/context_management.py:235` | `create_supervisor_pre_model_hook()` |
| Py graph | `lobster/agents/graph.py` | Graph builder, delegation tools, store wiring |
| Py callback | `lobster/ui/callbacks/protocol_callback.py` | LangGraph callback -> protocol events |
| Test golden | `tests/integration/test_slash_command_golden_transcripts.py` | Protocol output snapshot testing |
| Test smoke | `tests/integration/test_go_tui_protocol_smoke_script.py` | Binary launch + protocol handshake |
| Test regr. | `tests/unit/cli/test_slash_commands_go_tui_regressions.py` | Protocol safety regressions |
