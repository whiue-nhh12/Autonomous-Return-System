# Manager Agent

`ManagerAgentGraph` creates one graph run per user turn. It routes the request to
the configured policy worker, or returns an explicit `not_configured` result
for child services that are not implemented. Eligibility requests fan out to
the policy and user-scoped order workers, then pass their results to the return
decision worker. The decision worker only auto-approves or rejects when an
injected, reviewed structured rule provider enables automated decisions.
Retrieved policy passages alone result in `manual_review`. The worker creates
a proposal in shared state and does not persist a decision or initiate a refund.

Clarification ends the current graph run. Send the user's next message as a new
request. The manager generates a fresh `request_id` and can reuse the supplied
`conversation_id`.

## Authenticated API boundary

Build a `ManagerAgentGraph`, then pass it to
`backend.api.app.create_app(manager)` to create the FastAPI application. The
endpoint is `POST /agent/requests`. Its JSON body accepts `message`,
optional `language`, and optional `conversation_id`. It rejects extra fields,
including `user_id`. The manager receives `user_id` only from the authenticated
principal, never from the request body.

Configure the external OIDC provider with:

```text
AUTH_JWKS_URL=https://identity.example/.well-known/jwks.json
AUTH_JWT_ISSUER=https://identity.example/
AUTH_JWT_AUDIENCE=automous-return-api
AUTH_USER_ID_CLAIM=user_id
```

The issuer must place the internal application user's immutable UUID in the
configured claim (`user_id` by default). Authentication verifies an RS256
signature from the configured JWKS, issuer, audience, expiration, issued-at
time, subject, and UUID claim. The UUID is then resolved to an active database
user. Missing authentication configuration fails closed with HTTP 503; invalid
tokens and unknown or inactive users receive HTTP 401.

Shared request state is defined in `agent/core/state/shared.py`. The auth service and
injectable `get_current_user` dependency are in `service/auth/`; JWT/JWKS
verification is isolated in `backend/core/security/`; JWT settings and the
authenticated principal schemas live in `backend/schemas/auth/`. `backend/api/auth.py` is only a thin export for the
dependency, while manager request and response schemas are in `schemas/manager.py`.

No token issuer, login endpoint, or identity mapping table exists in this
repository. Provision tokens through the application's OIDC identity provider
with the configured internal-user-ID claim.
