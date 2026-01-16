# Disease Standardization Tutorial Diagram

**Purpose**: Show 5-level fuzzy matching for tutorial video (Phase 4)

**Placement**: Tutorial video at ~5:30 timestamp (during disease standardization explanation)

---

## Diagram: 5-Level Disease Matching

```mermaid
graph LR
    subgraph "Input (Messy Labels)"
        A["colorectal cancer"]
        B["Stage III colorectal cancer"]
        C["UC"]
        D["Crohn disease"]
        E["healthy control"]
    end

    subgraph "Matching Strategy"
        L1["Level 1: Exact Match"]
        L2["Level 2: Contains"]
        L3["Level 3: Reverse Contains"]
        L4["Level 4: Token Match"]
        L5["Level 5: Unmapped"]
    end

    subgraph "Output (Standardized)"
        O1["crc"]
        O2["crc"]
        O3["uc"]
        O4["cd"]
        O5["healthy"]
    end

    A -->|Level 1| L1 --> O1
    B -->|Level 2| L2 --> O2
    C -->|Level 1| L1 --> O3
    D -->|Level 2| L2 --> O4
    E -->|Level 4| L4 --> O5

    style A fill:#ffebee,stroke:#c62828
    style B fill:#ffebee,stroke:#c62828
    style C fill:#ffebee,stroke:#c62828
    style D fill:#ffebee,stroke:#c62828
    style E fill:#ffebee,stroke:#c62828

    style L1 fill:#fff3e0,stroke:#f57c00
    style L2 fill:#fff3e0,stroke:#f57c00
    style L3 fill:#fff3e0,stroke:#f57c00
    style L4 fill:#fff3e0,stroke:#f57c00
    style L5 fill:#fff3e0,stroke:#f57c00

    style O1 fill:#c8e6c9,stroke:#2e7d32
    style O2 fill:#c8e6c9,stroke:#2e7d32
    style O3 fill:#c8e6c9,stroke:#2e7d32
    style O4 fill:#c8e6c9,stroke:#2e7d32
    style O5 fill:#c8e6c9,stroke:#2e7d32
```

---

## Alternative: Simpler Table View (Recommended for Video)

**If Mermaid is too complex, show as animated table:**

| Input (Messy) | Matching Level | Output (Standard) |
|---------------|----------------|-------------------|
| `"colorectal cancer"` | Level 1: Exact ✓ | **crc** |
| `"Stage III colorectal cancer"` | Level 2: Contains "colorectal cancer" ✓ | **crc** |
| `"UC"` | Level 1: Exact ✓ | **uc** |
| `"Crohn disease"` | Level 2: Contains "Crohn" ✓ | **cd** |
| `"healthy control"` | Level 4: Token "healthy" ✓ | **healthy** |

**Animation**: Reveal rows sequentially (1 per 2 seconds = 10 seconds total)

---

## Video Narration Script (for this diagram)

**Timing**: 5:30-5:50 (20 seconds)

> "The service uses 5-level fuzzy matching to standardize disease labels.
>
> Level 1 catches exact matches like 'UC' to 'uc'.
>
> Level 2 handles variations like 'Stage III colorectal cancer' by checking if the label contains known disease terms.
>
> This continues through 5 levels until all labels are standardized or marked unmapped.
>
> In our case, all 89 samples achieved 100% mapping success."

---

**Recommendation**: Use **table animation** instead of Mermaid for clearer video presentation.
