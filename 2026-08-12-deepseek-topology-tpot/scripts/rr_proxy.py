#!/usr/bin/env python3
"""极简轮询代理 —— 聚合对照臂 aggR2 用。

为什么需要它:第一轮的聚合臂用的是单实例 TP2,与 PD 臂(2×TP1)的差异
不止「聚合 vs 分离」,还多了一层 TP 通信开销。本代理把 2 个独立的 TP1 副本
组成一个入口,使聚合臂与 PD 臂在**并行度上完全对齐**,拆掉那个混淆。

设计取舍(必须写进结论):这是**无共享队列**的轮询,
没有 work-stealing —— 对聚合臂略偏悲观。真实生产 LB 通常更聪明。
"""
import argparse
import asyncio
import itertools

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "backends": app.state.backends}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    backend = next(app.state.rr)
    url = f"http://{backend}/v1/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    try:
        req = app.state.client.build_request(
            request.method, url, content=body, headers=headers
        )
        resp = await app.state.client.send(req, stream=True)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e), "backend": backend}, status_code=502)

    if "text/event-stream" in resp.headers.get("content-type", ""):
        async def gen():
            async for chunk in resp.aiter_raw():
                yield chunk
            await resp.aclose()

        return StreamingResponse(
            gen(), status_code=resp.status_code, media_type=resp.headers.get("content-type")
        )

    content = await resp.aread()
    await resp.aclose()
    return Response(
        content=content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--backends", nargs="+", required=True)
    args = ap.parse_args()

    app.state.backends = args.backends
    app.state.rr = itertools.cycle(args.backends)
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0),
        limits=httpx.Limits(max_connections=512, max_keepalive_connections=256),
    )
    print(f"Initialized round-robin over {len(args.backends)} backends: {args.backends}", flush=True)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
