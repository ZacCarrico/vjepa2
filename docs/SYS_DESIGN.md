## Batch Inference Architecture

``` mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'12px'}}}%%
flowchart LR
    SCHED[Scheduler] --> A[Batch<br/>Ingestion]
    A --> B[Preprocessing<br/>Service]
    B --> E[Classification<br/>Service]
    E --> F[(Database)]
    E --> ERR[Error<br/>Queue]

    style SCHED fill:#e3f2fd,stroke:#1976d2
    style A fill:#e3f2fd,stroke:#1976d2
    style E fill:#e8f5e8,stroke:#4caf50
    style F fill:#fff3e0,stroke:#f57c00
    style ERR fill:#fff3e0,stroke:#f57c00
```

## Streaming Inference Architecture

``` mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'12px'}}}%%
flowchart LR
    A[Video<br/>Source] --> B[Preprocessing<br/>Service]
    B --> C[Classification<br/>Service]
    C --> CB[Async<br/>Callback]
    C --> DB[(Database)]

    style A fill:#e3f2fd,stroke:#1976d2
    style C fill:#e8f5e8,stroke:#4caf50
    style CB fill:#fff3e0,stroke:#f57c00
    style DB fill:#fff3e0,stroke:#f57c00
```

## RESTful Architecture

``` mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'12px'}}}%%
flowchart LR
    A[Client] --> B[REST<br/>API]
    B --> C[Preprocessing<br/>Service]
    C --> D[Classification<br/>Service]
    D --> E[Response]
    D --> F[(Database)]
    E --> A

    style A fill:#e3f2fd,stroke:#1976d2
    style B fill:#e3f2fd,stroke:#1976d2
    style D fill:#e8f5e8,stroke:#4caf50
    style E fill:#fff3e0,stroke:#f57c00
    style F fill:#fff3e0,stroke:#f57c00
```

## Event-Driven Architecture

``` mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'12px'}}}%%
flowchart LR
    A[Event<br/>Sources] --> B[Topic]
    B --> C[Event<br/>Handler]
    C --> D[Preprocessing<br/>Service]
    D --> E[Classification<br/>Service]
    E --> F[Async<br/>Callback]
    E --> G[(Database)]

    style A fill:#e3f2fd,stroke:#1976d2
    style B fill:#e3f2fd,stroke:#1976d2
    style C fill:#e3f2fd,stroke:#1976d2
    style E fill:#e8f5e8,stroke:#4caf50
    style F fill:#fff3e0,stroke:#f57c00
    style G fill:#fff3e0,stroke:#f57c00
```

## Combined Architecture Overview

``` mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'18px'}}}%%
flowchart TB
    RESTCLIENT[Client or<br/>Camera]

    subgraph Batch["𝗕𝗔𝗧𝗖𝗛"]
        SCHED[Scheduler] --> BATCH[Batch<br/>Ingestion]
    end

    subgraph RESTful["𝗥𝗘𝗦𝗧𝗙𝗨𝗟"]
        RESTAPI[REST<br/>API]
    end

    subgraph EventDriven["𝗘𝗩𝗘𝗡𝗧-𝗗𝗥𝗜𝗩𝗘𝗡"]
        EVENTS[Event<br/>Sources] --> TOPIC[Topic] --> HANDLER[Event<br/>Handler]
    end

    subgraph Streaming["𝗦𝗧𝗥𝗘𝗔𝗠𝗜𝗡𝗚"]
        STREAM[Video<br/>Source]
    end


    RESTCLIENT --> RESTAPI
    RESTCLIENT --> EVENTS
    RESTCLIENT --> STREAM

    BATCH --> PRE[Preprocessing<br/>Service]
    RESTAPI --> PRE
    HANDLER --> PRE
    STREAM --> PRE

    PRE --> CLASS[Classification<br/>Service]

    CLASS --> DB[(Database)]
    CLASS --> ERRQ[Error<br/>Queue]
    CLASS --> CB[Async<br/>Callback]
    CLASS --> RESP[Response]
    RESP --> RESTCLIENT

    style SCHED fill:#e3f2fd,stroke:#1976d2
    style BATCH fill:#e3f2fd,stroke:#1976d2
    style STREAM fill:#e3f2fd,stroke:#1976d2
    style RESTCLIENT fill:#e3f2fd,stroke:#1976d2
    style RESTAPI fill:#e3f2fd,stroke:#1976d2
    style EVENTS fill:#e3f2fd,stroke:#1976d2
    style TOPIC fill:#e3f2fd,stroke:#1976d2
    style HANDLER fill:#e3f2fd,stroke:#1976d2
    style PRE fill:#e8f5e8,stroke:#4caf50
    style CLASS fill:#c8e6c9,stroke:#388e3c
    style DB fill:#fff3e0,stroke:#f57c00
    style ERRQ fill:#fff3e0,stroke:#f57c00
    style CB fill:#fff3e0,stroke:#f57c00
    style RESP fill:#fff3e0,stroke:#f57c00
```