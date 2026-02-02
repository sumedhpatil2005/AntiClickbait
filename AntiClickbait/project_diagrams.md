# AntiClickbait System Diagrams

## 1. High-Level System Architecture
This diagram illustrates the separation of concerns between the Chrome Extension (Frontend), Flask API (Backend), and the AI Components.

```mermaid
graph TD
    subgraph Client [Client Side - Chrome Extension]
        UI[YouTube UI Overlay]
        CS[Content Script]
        BG[Background Service Worker]
    end

    subgraph Server [Backend API - Flask]
        API[API Gateway /predict]
        
        subgraph Core [Core Logic]
            ML[LightGBM Model]
            LLM[Llama 3 Cerebras]
            TR[Transcript Fetcher]
            ENS[Ensemble Logic]
        end
    end

    subgraph External [External Services]
        YT[YouTube Data API]
        TAPI[TranscriptAPI.com]
        CER[Cerebras Cloud]
    end

    %% Flows
    UI -->|DOM Mutation| CS
    CS -->|Video ID| BG
    BG -->|POST /predict| API
    
    API -->|Metadata| YT
    API -->|Transcript| TAPI
    
    API -->|Features| ML
    API -->|Prompt| LLM
    LLM -.->|Inference| CER
    
    ML -->|Prob Score| ENS
    LLM -->|Flags & Analysis| ENS
    TR -->|Text| LLM
    
    ENS -->|Final Verdict| API
    API -->|JSON Response| BG
    BG -->|Badge Update| UI

    %% Styling
    classDef client fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef server fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef ext fill:#fce4ec,stroke:#880e4f,stroke-width:2px;
    
    class UI,CS,BG client;
    class API,ML,LLM,TR,ENS server;
    class YT,TAPI,CER ext;
```

## 2. API Sequence Diagram (Request Flow)
Detailed step-by-step flow of handling a prediction request.

```mermaid
sequenceDiagram
    participant Ext as Chrome Extension
    participant API as Flask Backend
    participant YT as YouTube API
    participant ML as LightGBM
    participant LLM as Llama 3
    participant TR as Transcript API
    
    Ext->>API: POST /predict (video_id)
    activate API
    
    API->>YT: Fetch Video Metadata (Title, Stats)
    YT-->>API: Metadata JSON
    
    par Parallel Analysis
        API->>ML: Extract Features -> Predict
        ML-->>API: Probability (e.g., 0.85)
        
        API->>LLM: Metadata Analysis (Prompt)
        LLM-->>API: Initial Flags (e.g., "Full Movie")
    end
    
    alt Risk Detected (ML > 0.5 or Flags Found)
        API->>TR: Fetch Transcript
        alt Transcript Found
            TR-->>API: Full Text
            API->>LLM: Deep Verification (Verify Promises)
            LLM-->>API: Confirmed/Debunked + Timestamp
        else No Transcript
            API-->>API: Fallback to Metadata Analysis
        end
    end
    
    API->>API: Ensemble Logic (Weighted Avg / Veto)
    API-->>Ext: JSON Response {verdict, confidence, reason}
    deactivate API
    
    Ext->>Ext: Render Badge & Tooltip
```

## 3. ML Feature Engineering Pipeline
How raw video data is transformed into the 55 features used by LightGBM.

```mermaid
flowchart LR
    Input[Raw Video Data]
    
    subgraph TextProc [Text Processing]
        Clean[Clean Text]
        TFIDF[TF-IDF Vectorization]
        Sent[Sentiment Analysis]
    end
    
    subgraph FeatEng [Feature Engineering]
        Kw[Keyword Counters]
        Caps[CAPS Ratio]
        Emoji[Emoji Count]
        Structure[Structure Check]
    end
    
    subgraph MetaFeat [Metadata Features]
        Ratio[Like/View Ratio]
        Dur[Duration Log]
        Views[View Log]
    end
    
    Input --> Clean --> TFIDF
    Input --> FeatEng
    Input --> MetaFeat
    
    TFIDF --> Matrix
    FeatEng --> Matrix
    MetaFeat --> Matrix
    
    Matrix[Feature Matrix - sparse] --> Model[LightGBM Classifier]
    Model --> Prob[Clickbait Probability]
    
    style Input fill:#d1c4e9
    style Model fill:#c8e6c9
    style Prob fill:#ffccbc
```

## 4. Hybrid Ensemble Decision Logic
The "Brain" that decides the final verdict.

```mermaid
flowchart TD
    Start([Ensemble Input]) --> ML_Score[ML Probability]
    Start --> LLM_Res[LLM Flags]
    
    CheckFlags{Has Red Flags?}
    LLM_Res --> CheckFlags
    
    CheckFlags -- "Yes (Piracy/Scam)" --> ForceRisk[Force MISLEADING 0.95]
    
    CheckFlags -- No --> CheckPromise{Promise Verified?}
    
    CheckPromise -- "Yes (Key Moment Found)" --> Veto[Force TRUSTWORTHY 0.1]
    
    CheckPromise -- No --> CheckConf{LLM High Confidence?}
    
    CheckConf -- "Yes (> 80% Safe)" --> Veto
    CheckConf -- "No / Ambiguous" --> Weighted[Weighted Average]
    
    ML_Score --> Weighted
    
    ForceRisk --> Result([Final Verdict])
    Veto --> Result
    Weighted --> Result
    
    style ForceRisk fill:#ffcdd2,stroke:#b71c1c
    style Veto fill:#c8e6c9,stroke:#1b5e20
    style Weighted fill:#fff9c4,stroke:#fbc02d
```
