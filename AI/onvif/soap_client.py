from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests


SOAP_ENV = "http://www.w3.org/2003/05/soap-envelope"
NS = {
    "s": SOAP_ENV,
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "trt": "http://www.onvif.org/ver10/media/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
}


class OnvifError(Exception):
    pass


class OnvifAuthError(OnvifError):
    pass


@dataclass(slots=True)
class OnvifProfile:
    token: str
    name: str


@dataclass(slots=True)
class OnvifCapabilities:
    media_xaddr: Optional[str]


@dataclass(slots=True)
class OnvifDeviceInfo:
    manufacturer: Optional[str]
    model: Optional[str]
    firmware: Optional[str]
    serial: Optional[str]


class OnvifClient:
    """
    Lightweight ONVIF SOAP client using requests only (digest auth supported).
    """

    def __init__(
        self,
        xaddr: str,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.xaddr = xaddr
        self.username = username
        self.password = password
        self.session = session or requests.Session()

    # ------------------ public ------------------ #
    def get_capabilities(self) -> OnvifCapabilities:
        body = """
        <tds:GetCapabilities xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
          <tds:Category>All</tds:Category>
        </tds:GetCapabilities>
        """
        root = self._post_soap(
            self.xaddr,
            body,
            action="http://www.onvif.org/ver10/device/wsdl/GetCapabilities",
        )
        media_xaddr = (
            self._find_text(root, ".//tds:Capabilities/tds:Media/tt:XAddr") or None
        )
        return OnvifCapabilities(media_xaddr=media_xaddr)

    def get_device_information(self) -> OnvifDeviceInfo:
        body = """
        <tds:GetDeviceInformation xmlns:tds="http://www.onvif.org/ver10/device/wsdl" />
        """
        root = self._post_soap(
            self.xaddr,
            body,
            action="http://www.onvif.org/ver10/device/wsdl/GetDeviceInformation",
        )
        return OnvifDeviceInfo(
            manufacturer=self._find_text(
                root, ".//tds:GetDeviceInformationResponse/tds:Manufacturer"
            ),
            model=self._find_text(root, ".//tds:GetDeviceInformationResponse/tds:Model"),
            firmware=self._find_text(
                root, ".//tds:GetDeviceInformationResponse/tds:FirmwareVersion"
            ),
            serial=self._find_text(
                root, ".//tds:GetDeviceInformationResponse/tds:SerialNumber"
            ),
        )

    def get_profiles(self, media_xaddr: Optional[str] = None) -> List[OnvifProfile]:
        body = """
        <trt:GetProfiles xmlns:trt="http://www.onvif.org/ver10/media/wsdl" />
        """
        root = self._post_soap(
            media_xaddr or self.xaddr,
            body,
            action="http://www.onvif.org/ver10/media/wsdl/GetProfiles",
        )
        profiles: list[OnvifProfile] = []
        for p in root.findall(".//trt:Profiles", NS):
            token = p.attrib.get("token")
            name_el = p.find("./tt:Name", NS)
            name = (
                name_el.text.strip()
                if name_el is not None and name_el.text
                else token or "Profile"
            )
            if token:
                profiles.append(OnvifProfile(token=token, name=name))
        return profiles

    def get_stream_uri(
        self, profile_token: str, media_xaddr: Optional[str] = None
    ) -> Optional[str]:
        # Try TCP first to avoid UDP-only failures; fall back to RTSP default.
        for protocol in ("TCP", "RTSP"):
            body = f"""
            <trt:GetStreamUri xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
              <trt:StreamSetup>
                <tt:Stream>RTP-Unicast</tt:Stream>
                <tt:Transport>
                  <tt:Protocol>{protocol}</tt:Protocol>
                </tt:Transport>
              </trt:StreamSetup>
              <trt:ProfileToken>{profile_token}</trt:ProfileToken>
            </trt:GetStreamUri>
            """
            try:
                root = self._post_soap(
                    media_xaddr or self.xaddr,
                    body,
                    action="http://www.onvif.org/ver10/media/wsdl/GetStreamUri",
                )
                uri = self._find_text(
                    root, ".//trt:GetStreamUriResponse/trt:MediaUri/tt:Uri"
                )
                if uri:
                    return uri
            except OnvifError:
                continue
        return None

    # ------------------ internals ------------------ #
    def _post_soap(self, url: str, body: str, action: Optional[str] = None) -> ET.Element:
        envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="{SOAP_ENV}">
  <s:Body>
    {body}
  </s:Body>
</s:Envelope>
"""
        auth = None
        if self.username and self.password:
            # Digest is most common; caller can re-instantiate with different auth if needed.
            auth = requests.auth.HTTPDigestAuth(self.username, self.password)
        ct = 'application/soap+xml; charset=utf-8'
        if action:
            ct = f'{ct}; action="{action}"'
        headers = {"Content-Type": ct}
        if action:
            headers["SOAPAction"] = action
        try:
            resp = self.session.post(
                url,
                data=envelope,
                headers=headers,
                timeout=(3, 5),
                auth=auth,
            )
        except requests.exceptions.ReadTimeout as e:
            raise OnvifError(f"timeout talking to {urlparse(url).hostname}") from e
        except Exception as e:
            raise OnvifError(f"request failed to {url}") from e

        if resp.status_code == 401:
            raise OnvifAuthError("authentication required")
        if not (200 <= resp.status_code < 300):
            detail = resp.text
            detail = detail[:300] + ("..." if len(detail) > 300 else "")
            raise OnvifError(f"bad status {resp.status_code} ({detail})")
        try:
            root = ET.fromstring(resp.content)
        except Exception as e:
            raise OnvifError("invalid XML response") from e
        return root

    @staticmethod
    def _find_text(root: ET.Element, path: str) -> Optional[str]:
        el = root.find(path, NS)
        return el.text.strip() if el is not None and el.text else None


__all__ = [
    "SOAP_ENV",
    "NS",
    "OnvifError",
    "OnvifAuthError",
    "OnvifProfile",
    "OnvifCapabilities",
    "OnvifDeviceInfo",
    "OnvifClient",
]

