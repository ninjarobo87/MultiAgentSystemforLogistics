# Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **Max 3 loops** | Each iteration costs ~2K-4K tokens. 3 retries = max ~12K tokens for extraction cycles. Prevents runaway costs on unparseable documents. |
| **Compilation error accumulation** | Uses `Annotated[list, operator.add]` for append-only error logs. Each retry carries forward ALL previous errors so the extraction agent learns from mistakes. |
| **Semantic validation via LLM** | Rule-based checks catch format errors; LLM catches logical inconsistencies (e.g., "Maersk" carrier with "express air" service type). |
| **RouterAgent as entry gate** | Prevents unnecessary extraction on pre-structured data (API webhooks, system-to-system). Saves tokens and latency. |
| **Structured CompilationError** | Each error includes `resolution_hint` - this is injected into the extraction prompt on retry, giving the LLM targeted guidance. |
| **Hard state typing** | `TypedDict` ensures compile-time safety. No runtime state corruption between agents. |
