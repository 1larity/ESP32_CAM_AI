from __future__ import annotations

import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET


WSA_UDP_ADDR = ("239.255.255.250", 3702)
PROBE_TYPE = "dn:NetworkVideoTransmitter"
NS = {
    "soap": "http://www.w3.org/2003/05/soap-envelope",
    "wsd": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
}


@dataclass(slots=True)
class OnvifDiscoveryResult:
    xaddr: str
    epr: Optional[str]
    scopes: List[str]
    ip: str

    @property
    def host(self) -> str:
        p = urlparse(self.xaddr)
        return p.hostname or self.ip

    @property
    def port(self) -> Optional[int]:
        p = urlparse(self.xaddr)
        return p.port


def _build_probe() -> bytes:
    mid = f"uuid:{uuid.uuid4()}"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Envelope xmlns="http://www.w3.org/2003/05/soap-envelope"
          xmlns:wsd="http://schemas.xmlsoap.org/ws/2005/04/discovery"
          xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <Header>
    <MessageID>{mid}</MessageID>
    <To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</To>
    <Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</Action>
  </Header>
  <Body>
    <wsd:Probe>
      <wsd:Types>{PROBE_TYPE}</wsd:Types>
    </wsd:Probe>
  </Body>
</Envelope>
""".encode("utf-8")


def _parse_probe_match(payload: bytes, src_ip: str) -> Optional[OnvifDiscoveryResult]:
    try:
        root = ET.fromstring(payload)
    except Exception:
        return None

    def find_text(path: str) -> Optional[str]:
        el = root.find(path, NS)
        return el.text.strip() if el is not None and el.text else None

    xaddrs_raw = find_text(".//wsd:ProbeMatch/wsd:XAddrs")
    if not xaddrs_raw:
        return None
    xaddrs = xaddrs_raw.split()
    if not xaddrs:
        return None

    epr = find_text(".//wsd:ProbeMatch/wsd:EndpointReference/wsd:Address")
    scopes_raw = find_text(".//wsd:ProbeMatch/wsd:Scopes") or ""
    scopes = [s for s in scopes_raw.split() if s]

    return OnvifDiscoveryResult(
        xaddr=xaddrs[0],
        epr=epr,
        scopes=scopes,
        ip=src_ip,
    )


def discover_onvif(timeout: float = 2.0, retries: int = 1, *, stop_event: Optional[threading.Event] = None, max_results: int = 64) -> List[OnvifDiscoveryResult]:
    """
    WS-Discovery probe for ONVIF devices on the local network.
    """
    stop = stop_event or threading.Event()
    seen: set[str] = set()
    results: list[OnvifDiscoveryResult] = []
    probe = _build_probe()

    for _ in range(max(1, retries)):
        if stop.is_set():
            break
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.settimeout(timeout)
            sock.sendto(probe, WSA_UDP_ADDR)
            t0 = time.time()
            while time.time() - t0 < timeout:
                if stop.is_set():
                    break
                try:
                    data, (ip, _) = sock.recvfrom(65535)
                except socket.timeout:
                    break
                res = _parse_probe_match(data, ip)
                if res is None:
                    continue
                if res.xaddr in seen:
                    continue
                seen.add(res.xaddr)
                results.append(res)
                if len(results) >= max_results:
                    stop.set()
                    break
        finally:
            try:
                sock.close()
            except Exception:
                pass
        if stop.is_set():
            break
    return results
