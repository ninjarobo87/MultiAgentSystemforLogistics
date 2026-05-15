# Token Bleeding Prevention Strategy

The logistics multi-agent state machine uses a strict retry cap to control token usage and keep costs predictable.

## Retry Token Budget

- Loop 1: ~3K tokens (initial extraction + validation)
- Loop 2: ~4K tokens (re-extraction with 1 error log + validation)
- Loop 3: ~5K tokens (re-extraction with 2 error logs + validation)

## Hard Stop Policy

- After Loop 3, the workflow transitions to `failure_handler` if validation still fails.
- This prevents:
  - Token bleeding
  - Infinite loops
  - Cost runaway

## Why it matters

Without this cap, adversarial or garbage input could loop indefinitely and consume unbounded tokens, creating a production cost disaster. The hard stop at 3 iterations ensures the system fails safely and escalates for human review.
