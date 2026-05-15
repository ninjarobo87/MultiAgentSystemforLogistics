# Multi-Agent State Machine for Logistics Company using LangGraph

## Architecture Overview

This system handles incoming logistics payloads such as shipment orders, bills of lading (BOLs), customs declarations, and tracking updates. A central `RouterAgent` evaluates the payload and routes it to either:

- `DataExtractionAgent` for parsing unstructured logistics documents.
- `ValidationAgent` for verifying extracted shipment data against business rules.

The state machine is built with LangGraph and uses a combination of structured validation and LLM-guided routing.

## Processing Flow

- Input is received as a raw payload.
- `RouterAgent` decides if the payload is raw/unstructured or already structured/pre-extracted.
- If extraction is needed, `DataExtractionAgent` extracts shipment details into structured fields.
- `ValidationAgent` then validates the extracted data against business logic and compliance rules.
- The system supports a retry loop with up to 3 iterations to avoid token bleeding.

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOGISTICS MULTI-AGENT STATE MACHINE                  │
│                                                                         │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐             │
│  │  START  │───▶│ RouterAgent  │───▶│ DataExtractionAgent  │           │
│  └─────────┘    └──────────────┘    └─────────────────────┘             │
│                        │                       │                        │
│                        │                       ▼                        │
│                        │              ┌─────────────────┐               │
│                        └─────────────▶│ ValidationAgent │              │
│                                       └─────────────────┘               │
│                                          │           │                  │
│                                     PASS │           │ FAIL             │
│                                          ▼           ▼                  │
│                                    ┌─────────┐  ┌──────────┐            │
│                                    │   END   │  │ Loop ≤3? │            │
│                                    └─────────┘  └─────────┘             │
│                                                   │        │            │
│                                              YES  │        │ NO         │
│                                                   ▼        ▼            │
│                                          ┌──────────┐ ┌─────────┐       │
│                                          │Extract+  │ │FAIL_END │       │
│                                          │ErrorLog  │ └─────────┘       │
│                                          └──────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```
