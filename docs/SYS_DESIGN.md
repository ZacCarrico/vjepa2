``` mermaid
flowchart LR
    A[Raw Video Ingestion] --> B[Preprocessing Service]
    B -->|Decode & Normalize| C[Video Frames / Chunks]
    B -->|Optional: Embeddings| D[Lightweight Features]
    A --> S[(Raw Video Storage)]
    C --> E[Classification Service]
    D --> E
    E --> F[Predictions & Labels DB]

    subgraph Storage
        S
        C
        D
    end
