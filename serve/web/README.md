# bnk dashboard

Live analytics for the bnk engine, built with React, Tailwind and [shadcn/ui](https://ui.shadcn.com/)
(components in `src/components/ui`, added with `npx shadcn add ...`).

- `run.sh` builds it automatically (`npm ci && npm run build` into `dist/`, served by `serve/server.py`).
- `npm run dev` serves it with hot reload and proxies `/api` and `/v1` to a running server (`BNK_URL`, default
  `http://localhost:8080`).

Data comes from `GET /api/stream` (server-sent events): `init` once, then `live` (~4/s engine snapshot), `sample`
(1/s rates), `request` (finished requests) and `log`. See `src/lib/telemetry.ts`.
