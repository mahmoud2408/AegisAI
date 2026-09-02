# API

Phase status: API contract proposal only. Endpoints are not implemented yet.

Base path:

```text
/api/v1
```

Planned endpoints:

- `GET /health`
- `GET /metrics`
- `GET /logs`
- `GET /traces`
- `GET /anomalies`
- `GET /predictions`
- `POST /forecast`
- `GET /incidents`
- `GET /incidents/{id}`
- `POST /investigations`
- `GET /investigations/{id}`
- `POST /rag/search`
- `POST /agent/investigate`
- `GET /models`

Planned real-time support:

- WebSocket stream for live alerts and investigation status updates

API requirements:

- Pydantic request and response schemas
- Structured errors
- Proper HTTP status codes
- No internal stack traces in client responses
- Authentication and authorization extension points
- Rate limiting for expensive AI endpoints where practical
- OpenAPI documentation generated from real schemas
