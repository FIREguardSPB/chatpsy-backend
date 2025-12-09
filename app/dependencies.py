"""FastAPI dependencies."""
from fastapi import Request


def get_client_ip(request: Request) -> str:
    """
    Достаём IP:
    - сначала из X-Forwarded-For (ngrok / прокси),
    - если нет — из request.client.host.
    """
    xff = request.headers.get("x-forwarded-for") or request.headers.get(
        "X-Forwarded-For"
    )
    if xff:
        # может быть "ip1, ip2, ip3" — берём первый
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"
