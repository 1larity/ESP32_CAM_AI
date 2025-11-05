#!/usr/bin/env python3
"""
discover_cams.py — Discover ESP32-CAMs on the local LAN.

Strategy
- Determine local /24 automatically via a UDP socket trick (override with --cidr).
- Probe http://IP/ping (your firmware replies "pong").
- Optionally probe :81/stream to confirm MJPEG boundary and auth requirements.
- Concurrency with ThreadPoolExecutor. Safe timeouts.

Usage
  python discover_cams.py                # auto /24 from primary NIC
  python discover_cams.py --cidr 192.168.1.0/24
  python discover_cams.py --check-stream
  python discover_cams.py --workers 256 --timeout 0.6
"""
from __future__ import annotations
import argparse
import ipaddress
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def guess_primary_ipv4() -> str | None:
    """Pick the local IPv4 by opening a UDP socket to a public IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

def default_cidr() -> str:
    return "192.168.1.0/24"

# def default_cidr() -> str:
#     ip = guess_primary_ipv4()
#     if not ip:
#         return "192.168.1.0/24"
#     # assume /24
#     parts = ip.split(".")
#     return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"


def probe_host(ip: str, timeout: float, check_stream: bool, token: str | None) -> dict | None:
    base = f"http://{ip}"
    session = requests.Session()
    session.headers.update({"User-Agent": "ESP32-CAM-Discovery/1.0"})
    # 1) /ping
    try:
        r = session.get(f"{base}/ping", timeout=timeout)
        text = (r.text or "").strip().lower()
        if r.status_code == 200 and text.startswith("pong"):
            info = {
                "ip": ip,
                "ping": True,
                "auth": False,
                "stream_ok": None,
                "notes": "",
            }
            # try / (may 401 if auth on)
            try:
                r0 = session.get(base + "/", timeout=timeout)
                if r0.status_code == 401:
                    info["auth"] = True
                elif r0.ok:
                    # maybe capture title if present
                    t = r0.text.lower()
                    if "<title" in t:
                        info["notes"] = "web ui reachable"
            except Exception:
                pass

            # 2) :81/stream (optional)
            if check_stream:
                stream_url = f"http://{ip}:81/stream"
                if token:
                    sep = "&" if "?" in stream_url else "?"
                    stream_url = f"{stream_url}{sep}token={token}"
                try:
                    r1 = session.get(stream_url, stream=True, timeout=timeout)
                    if r1.status_code == 401:
                        info["stream_ok"] = False
                        info["auth"] = True
                    elif r1.ok:
                        ctype = r1.headers.get("Content-Type", "")
                        # read a small chunk to confirm boundary, then close
                        try:
                            next(r1.iter_content(chunk_size=512))
                        except Exception:
                            pass
                        info["stream_ok"] = ("multipart/x-mixed-replace" in ctype.lower())
                    else:
                        info["stream_ok"] = False
                except Exception:
                    info["stream_ok"] = False
            return info
    except requests.exceptions.RequestException:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Discover ESP32-CAM devices on LAN")
    ap.add_argument("--cidr", default=default_cidr(), help="CIDR to scan, e.g. 192.168.1.0/24")
    ap.add_argument("--workers", type=int, default=128, help="Max concurrent probes")
    ap.add_argument("--timeout", type=float, default=0.8, help="Per-request timeout seconds")
    ap.add_argument("--check-stream", action="store_true", help="Also probe :81/stream")
    ap.add_argument("--token", default=None, help="Optional Base64 user:pass token for /stream")
    args = ap.parse_args()

    try:
        net = ipaddress.ip_network(args.cidr, strict=False)
    except Exception as e:
        print(f"Invalid CIDR: {e}", file=sys.stderr)
        sys.exit(2)

    hosts = [str(ip) for ip in net.hosts()]
    print(f"Scanning {len(hosts)} hosts in {args.cidr}…")
    t0 = time.time()

    found = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(probe_host, ip, args.timeout, args.check_stream, args.token): ip
            for ip in hosts
        }
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                found.append(res)

    dt = time.time() - t0
    if not found:
        print("No ESP32-CAMs found.")
        print(f"Done in {dt:.1f}s")
        return

    # Output table
    print("\nDiscovered devices:")
    print("{:<15} {:<6} {:<6} {}".format("IP", "PING", "STRM", "NOTES/AUTH"))
    print("-" * 54)
    for d in sorted(found, key=lambda x: x["ip"]):
        ping = "ok" if d["ping"] else "-"
        if d["stream_ok"] is None:
            strm = "-"
        else:
            strm = "ok" if d["stream_ok"] else "fail"
        notes = d["notes"]
        if d["auth"]:
            notes = (notes + " | auth").strip(" |")
        print("{:<15} {:<6} {:<6} {}".format(d["ip"], ping, strm, notes))

    print(f"\nDone in {dt:.1f}s")


if __name__ == "__main__":
    main()
