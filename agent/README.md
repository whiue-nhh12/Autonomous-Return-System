# Agent package layout

```text
agent/
├── orchestration/
│   ├── graph.py              Manager Agent graph
│   ├── routing.py            intent routing and parallel fan-out
│   ├── state.py              LangGraph state and result reducer
│   └── nodes/                one module per orchestration node
└── agents/
    ├── policy/               implemented policy retrieval agent
    │   ├── agent.py          retrieval worker
    │   ├── graph.py          policy workflow
    │   └── state/            policy-local retrieval state
    ├── order/                read-only, user-scoped order worker
    └── return_decision/      deterministic return-eligibility decision worker
```

Shared cross-agent contracts live in `agent/core/state/shared.py`. Worker-local state
belongs in that worker's `state/` package. Keep orchestration nodes focused on
coordination; business rules and persistence belong in their domain services.

Order lookup is read-only and user-scoped. Return eligibility now combines the
order and policy workers and records a typed proposal in shared state. The
default decision worker requires a reviewed structured policy rule provider
before it can return `approve` or `reject`; retrieved policy passages alone
produce `manual_review`. It does not persist decisions or create refunds.
Return submission remains unimplemented. The old `agent.policy_agent` and
`agent.manager_agent` import paths are temporary compatibility wrappers; new
code should import from `agent.agents.policy` and `agent.orchestration`.
