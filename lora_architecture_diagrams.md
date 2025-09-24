# LoRA Model Architecture Changes

## Original V-JEPA 2 Model (Without LoRA)
```mermaid
graph TD
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

    style D fill:#e1f5fe
    style E fill:#e1f5fe
    style F fill:#e1f5fe
    style G fill:#e1f5fe
    style K fill:#e8f5e8
```

## V-JEPA 2 Model with LoRA Adapters
```mermaid
graph TD
    A[Input Video] --> B[V-JEPA 2 Encoder]
    B --> C[Attention Layers]
    C --> D[AdaptedLinear query_proj]
    C --> E[AdaptedLinear key_proj]
    C --> F[AdaptedLinear value_proj]
    C --> G[AdaptedLinear out_proj]

    D --> D1[Original query_proj<br/>FROZEN]
    D --> D2[LoRA query_proj<br/>TRAINABLE]
    D1 --> D3[+]
    D2 --> D3

    E --> E1[Original key_proj<br/>FROZEN]
    E --> E2[LoRA key_proj<br/>TRAINABLE]
    E1 --> E3[+]
    E2 --> E3

    F --> F1[Original value_proj<br/>FROZEN]
    F --> F2[LoRA value_proj<br/>TRAINABLE]
    F1 --> F3[+]
    F2 --> F3

    G --> G1[Original out_proj<br/>FROZEN]
    G --> G2[LoRA out_proj<br/>TRAINABLE]
    G1 --> G3[+]
    G2 --> G3

    D3 --> H[Multi-Head Attention]
    E3 --> H
    F3 --> H
    G3 --> I[Layer Output]
    H --> I
    I --> J[More Encoder Layers...]
    J --> K[Classification Head<br/>TRAINABLE]
    K --> L[Output Logits]

    style D1 fill:#ffcdd2
    style E1 fill:#ffcdd2
    style F1 fill:#ffcdd2
    style G1 fill:#ffcdd2
    style D2 fill:#c8e6c9
    style E2 fill:#c8e6c9
    style F2 fill:#c8e6c9
    style G2 fill:#c8e6c9
    style K fill:#c8e6c9
```

## LoRA Layer Internal Structure
```mermaid
graph TD
    A[Input x] --> B[Original Linear Layer<br/>W_0 FROZEN]
    A --> C[LoRA Path]

    C --> D[LoRA A Matrix<br/>in_features → rank]
    D --> E[Dropout]
    E --> F[LoRA B Matrix<br/>rank → out_features]
    F --> G[× scaling factor<br/>α/r]

    B --> H[+]
    G --> H
    H --> I[Output]

    style B fill:#ffcdd2
    style D fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#fff3e0
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
graph TD
    A[V-JEPA 2 Encoder Output] --> B[Original Classification Head<br/>Pre-trained for SSV2]
    B --> C[SSV2 Classes<br/>174 outputs]

    A --> D[New Classification Head<br/>Trainable]
    D --> E[UCF-101 Classes<br/>101 outputs]

    style B fill:#ffcdd2
    style C fill:#ffcdd2
    style D fill:#c8e6c9
    style E fill:#c8e6c9
```

## Key Changes Summary
- **216 LoRA modules** applied to attention layers (query_proj, key_proj, value_proj, out_proj)
- **Original attention weights frozen** (red boxes)
- **LoRA adapters trainable** (green boxes) with rank=16, α=32.0
- **Classification head replaced** and trainable for UCF-101 (101 classes vs original SSV2 174 classes)
- **Massive parameter reduction**: Only LoRA + classification head parameters are trainable