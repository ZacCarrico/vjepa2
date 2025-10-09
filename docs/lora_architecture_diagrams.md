# V-JEPA 2 LoRA Fine-tuning Architecture

This document shows the architecture changes for V-JEPA 2 LoRA fine-tuning on a **UCF-101 subset with 10 classes** (not the full 101 classes).

## V-JEPA 2 General Training for Classification
```mermaid
graph LR
    A[Video frames] --> B[V-JEPA 2 Encoder]
    B --> C[Video Embeddings]
    C --> D[VJEPA2AttentivePooler]
    D --> E[Classification Head]
    E --> F[Class Logits]
    F --> G[Cross-Entropy Loss]

    H[Ground Truth Labels] --> G
    G --> I[Backpropagation]
    I --> J[Parameter Updates]

    style B fill:#e3f2fd,stroke:#1976d2
    style E fill:#e8f5e8,stroke:#4caf50
    style G fill:#fff3e0,stroke:#f57c00
```

## V-JEPA 2 Inference
```mermaid
graph LR
    A[Input Video] --> B[V-JEPA 2 Encoder]
    B --> C[Video Embeddings]
    C --> D[VJEPA2AttentivePooler]
    D --> E[Classification Head]
    E --> F[Class Logits]
    F --> G[ArgMax -> Predicted Class]
```

## V-JEPA 2 With Classification Output (Without LoRA)
```mermaid
graph LR
    A[Input Video] --> B[V-JEPA 2 Encoder]
    B --> C[Attention Layers]
    C --> D[query_proj Linear]
    C --> E[key_proj Linear]
    C --> F[value_proj Linear]
    C --> G[out_proj Linear]
    D --> H[Multi-Head Attention]
    E --> H
    F --> H
    G --> I[Layer Output]
    H --> I
    I --> J[More Encoder Layers...]
    J --> K[Classification Head]
    K --> L[Output Logits]

    style D fill:#e1f5fe,stroke:#0288d1
    style E fill:#e1f5fe,stroke:#0288d1
    style F fill:#e1f5fe,stroke:#0288d1
    style G fill:#e1f5fe,stroke:#0288d1
    style K fill:#e8f5e8,stroke:#4caf50
```

## V-JEPA 2 Model with LoRA Adapters
```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'12px'}}}%%
graph LR
    A[Input<br/>Video] --> B[V-JEPA 2<br/>Encoder]
    B --> C[Attention<br/>Layers]

    subgraph "Q/K/V Projections"
        D[AdaptedLinear<br/>query/key/value_proj]
        D1[Original<br/>query/key/value_proj<br/>FROZEN]
        D2[LoRA<br/>query/key/value_proj<br/>TRAINABLE]
        D --> D1
        D --> D2
        D1 --> D3[+]
        D2 --> D3
    end

    subgraph "Output Projection"
        G[AdaptedLinear<br/>out_proj]
        G1[Original<br/>out_proj<br/>FROZEN]
        G2[LoRA<br/>out_proj<br/>TRAINABLE]
        G --> G1
        G --> G2
        G1 --> G3[+]
        G2 --> G3
    end

    C --> D
    D3 --> H[Multi-Head<br/>Attention]
    H --> G
    G3 --> I[Layer<br/>Output]
    I --> J[More Encoder<br/>Layers...]
    J --> K[Classification<br/>Head<br/>TRAINABLE]
    K --> L[Output<br/>Logits]

    style D1 fill:#ffcdd2,stroke:#c62828
    style G1 fill:#ffcdd2,stroke:#c62828
    style D2 fill:#c8e6c9,stroke:#388e3c
    style G2 fill:#c8e6c9,stroke:#388e3c
    style K fill:#c8e6c9,stroke:#388e3c
```

## Weight Update Comparison: Regular Fine-tuning vs LoRA
```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'14px'}}}%%
graph TB
    subgraph Regular["Weight update in regular finetuning"]
        I1[Inputs<br/>d] --> W1[Pretrained<br/>weights<br/>W]
        I1 --> DW1[Weight<br/>update<br/>ΔW]
        W1 --> ADD1[+]
        DW1 --> ADD1
        ADD1 --> O1[Outputs]

        style W1 fill:#808080,stroke:#424242,color:#fff
        style DW1 fill:#1e88e5,stroke:#0d47a1,color:#fff
        style I1 fill:#000,stroke:#000,color:#fff
        style O1 fill:#000,stroke:#000,color:#fff
    end

    subgraph LoRA["Weight update in LoRA"]
        I2[Inputs x<br/>d] --> W2[Pretrained<br/>weights<br/>W]
        I2 --> A[Matrix A<br/>d → r]
        A --> B[Matrix B<br/>r → d]
        W2 --> ADD2[+]
        B --> ADD2
        ADD2 --> O2[Outputs]

        NOTE1["LoRA matrices A and B<br/>approximate ΔW = BA"]
        NOTE2["Inner dimension r<br/>is a hyperparameter"]

        style W2 fill:#808080,stroke:#424242,color:#fff
        style A fill:#1e88e5,stroke:#0d47a1,color:#fff
        style B fill:#1e88e5,stroke:#0d47a1,color:#fff
        style I2 fill:#000,stroke:#000,color:#fff
        style O2 fill:#000,stroke:#000,color:#fff
        style NOTE1 fill:#fff,stroke:#e0e0e0,color:#000
        style NOTE2 fill:#fff,stroke:#e0e0e0,color:#000
    end
```

## LoRA Attention Layer (Simplified for Slides)
```mermaid
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'12px'}}}%%
graph LR
    subgraph "Q/K/V Projections"
        A[Original<br/>FROZEN]
        B[LoRA<br/>TRAINABLE]
        A --> C[+]
        B --> C
    end

    C --> D[Multi-Head<br/>Attention]

    subgraph "Output Projection"
        E[Original<br/>FROZEN]
        F[LoRA<br/>TRAINABLE]
        E --> G[+]
        F --> G
    end

    D --> E
    D --> F
    G --> H[...]

    style A fill:#ffcdd2,stroke:#c62828
    style E fill:#ffcdd2,stroke:#c62828
    style B fill:#c8e6c9,stroke:#388e3c
    style F fill:#c8e6c9,stroke:#388e3c
```

## LoRA Layer Internal Structure
```mermaid
graph LR
    A[Input x] --> B[Original Linear Layer<br/>W_0 FROZEN]
    A --> C[LoRA Path]

    C --> D[LoRA A Matrix<br/>in_features → rank]
    D --> E[Dropout]
    E --> F[LoRA B Matrix<br/>rank → out_features]
    F --> G[× scaling factor<br/>α/r]

    B --> H[+]
    G --> H
    H --> I[Output]

    style B fill:#ffcdd2,stroke:#c62828
    style D fill:#c8e6c9,stroke:#388e3c
    style F fill:#c8e6c9,stroke:#388e3c
    style G fill:#fff3e0,stroke:#f57c00
```

### LoRA Mathematical Formulation

The LoRA adaptation follows this equation:
```
h = W₀x + ΔWx = W₀x + BAx × (α/r)
```

Where:
- `W₀` = Original pre-trained weight matrix (frozen)
- `ΔW = BA` = Low-rank decomposition of weight update
- `B` = LoRA B matrix (rank → out_features)
- `A` = LoRA A matrix (in_features → rank)
- `α` = LoRA alpha parameter (32.0)
- `r` = LoRA rank (16)
- `α/r` = scaling factor (2.0)

### Matrix Dimensions Example
For a typical attention projection layer in V-JEPA 2:
```
Input x:           [batch_size, seq_len, 1024]
W₀ (frozen):       [1024, 1024]
A matrix:          [1024, 16]     # in_features → rank
B matrix:          [16, 1024]     # rank → out_features
BA decomposition:  [1024, 1024]   # same as W₀
```

## Classification Head Replacement (Without LoRA)
```mermaid
graph LR
    A[V-JEPA 2 Encoder Output] --> B[Original Classification Head<br/>Pre-trained for SSV2]
    B --> C[SSV2 Classes<br/>174 outputs]

    A --> D[New Classification Head<br/>Trainable]
    D --> E[UCF-101 Subset Classes<br/>10 outputs]

    style B fill:#ffcdd2,stroke:#c62828
    style C fill:#ffcdd2,stroke:#c62828
    style D fill:#c8e6c9,stroke:#388e3c
    style E fill:#c8e6c9,stroke:#388e3c
```

## Key Changes Summary
- **216 LoRA modules** applied to attention layers (query_proj, key_proj, value_proj, out_proj)
- **Original attention weights frozen** (red boxes)
- **LoRA adapters trainable** (green boxes) with rank=16, α=32.0
- **Classification head replaced** and trainable for UCF-101 subset (10 classes vs original SSV2 174 classes)
- **Massive parameter reduction**: Only LoRA + classification head parameters are trainable
- **Classification head parameters**: ~10,240 parameters (1024 hidden × 10 classes + 10 bias terms)
- **LoRA parameters**: ~501,760 parameters across all attention modules
- **Total trainable**: ~512,000 parameters vs 375M+ total model parameters
