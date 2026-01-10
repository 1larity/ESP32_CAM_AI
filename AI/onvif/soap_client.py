from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import base64
import hashlib
import os
from datetime import datetime, timezone
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import requests


SOAP_ENV = "http://www.w3.org/2003/05/soap-envelope"
WSSE_NS = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU_NS = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
WSSE_PW_DIGEST_TYPE = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-username-token-profile-1.0#PasswordDigest"
)
WSSE_PW_TEXT_TYPE = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-username-token-profile-1.0#PasswordText"
)
WSSE_BASE64_ENCODING = (
    "http://docs.oasis-open.org/wss/2004/01/"
    "oasis-200401-wss-soap-message-security-1.0#Base64Binary"
)
NS = {
    "s": SOAP_ENV,
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "trt": "http://www.onvif.org/ver10/media/wsdl",
    "tt": "http://www.onvif.org/ver10/schema",
    "tptz": "http://www.onvif.org/ver20/ptz/wsdl",
}


class OnvifError(Exception):
    pass


class OnvifAuthError(OnvifError):
    pass


class OnvifHttpError(OnvifError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = int(status_code)
        self.detail = str(detail)
        super().__init__(f"bad status {self.status_code} ({self.detail})")


@dataclass(slots=True)
class OnvifProfile:
    token: str
    name: str


@dataclass(slots=True)
class OnvifCapabilities:
    media_xaddr: Optional[str]
    ptz_xaddr: Optional[str]


@dataclass(slots=True)
class OnvifDeviceInfo:
    manufacturer: Optional[str]
    model: Optional[str]
    firmware: Optional[str]
    serial: Optional[str]


@dataclass(slots=True)
class OnvifPreset:
    token: str
    name: Optional[str] = None


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

    def _wsse_username_token_header(self) -> str:
        """
        Add WS-Security UsernameToken (PasswordDigest).

        Many ONVIF devices require this even when HTTP auth is also used.
        """
        user = self.username or ""
        pwd = self.password or ""
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nonce = os.urandom(16)
        digest = base64.b64encode(
            hashlib.sha1(nonce + created.encode("utf-8") + pwd.encode("utf-8")).digest()
        ).decode("ascii")
        nonce_b64 = base64.b64encode(nonce).decode("ascii")

        return f"""
  <s:Header>
    <wsse:Security s:mustUnderstand="1" xmlns:wsse="{WSSE_NS}" xmlns:wsu="{WSU_NS}">
      <wsse:UsernameToken>
        <wsse:Username>{escape(user)}</wsse:Username>
        <wsse:Password Type="{WSSE_PW_DIGEST_TYPE}">{digest}</wsse:Password>
        <wsse:Nonce EncodingType="{WSSE_BASE64_ENCODING}">{nonce_b64}</wsse:Nonce>
        <wsu:Created>{created}</wsu:Created>
      </wsse:UsernameToken>
    </wsse:Security>
  </s:Header>
"""

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
        paths_media = (
            ".//tds:Capabilities/tds:Media/tt:XAddr",
            ".//tds:Capabilities/tt:Media/tt:XAddr",
        )
        paths_ptz = (
            ".//tds:Capabilities/tds:PTZ/tt:XAddr",
            ".//tds:Capabilities/tt:PTZ/tt:XAddr",
        )
        media_xaddr = None
        for p in paths_media:
            media_xaddr = self._find_text(root, p)
            if media_xaddr:
                break
        ptz_xaddr = None
        for p in paths_ptz:
            ptz_xaddr = self._find_text(root, p)
            if ptz_xaddr:
                break
        return OnvifCapabilities(media_xaddr=media_xaddr or None, ptz_xaddr=ptz_xaddr or None)

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

    def ptz_continuous_move(
        self,
        profile_token: str,
        *,
        ptz_xaddr: str,
        pan: float = 0.0,
        tilt: float = 0.0,
        zoom: float = 0.0,
        timeout_s: float = 0.6,
    ) -> None:
        pan = float(max(-1.0, min(1.0, pan)))
        tilt = float(max(-1.0, min(1.0, tilt)))
        zoom = float(max(-1.0, min(1.0, zoom)))
        timeout_s = float(max(0.1, min(10.0, timeout_s)))
        # Some cameras reject fractional xs:duration values; use whole seconds.
        timeout = f"PT{max(1, int(round(timeout_s)))}S"

        vel_parts = []
        if pan != 0.0 or tilt != 0.0:
            vel_parts.append(
                (
                    f'<tt:PanTilt x="{pan:.3f}" y="{tilt:.3f}" '
                    f'space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace" />'
                )
            )
        if zoom != 0.0:
            vel_parts.append(
                (
                    f'<tt:Zoom x="{zoom:.3f}" '
                    f'space="http://www.onvif.org/ver10/tptz/ZoomSpaces/VelocityGenericSpace" />'
                )
            )
        vel_xml = "\n".join(vel_parts)

        body = f"""
        <tptz:ContinuousMove xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
                             xmlns:tt="http://www.onvif.org/ver10/schema">
          <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
          <tptz:Velocity>
            {vel_xml}
          </tptz:Velocity>
          <tptz:Timeout>{timeout}</tptz:Timeout>
        </tptz:ContinuousMove>
        """
        self._post_soap(
            ptz_xaddr,
            body,
            action="http://www.onvif.org/ver20/ptz/wsdl/ContinuousMove",
        )

    def ptz_relative_move(
        self,
        profile_token: str,
        *,
        ptz_xaddr: str,
        pan: float = 0.0,
        tilt: float = 0.0,
        zoom: float = 0.0,
    ) -> None:
        """
        RelativeMove step (translation). Useful fallback for cameras that reject ContinuousMove.
        """
        pan = float(max(-1.0, min(1.0, pan)))
        tilt = float(max(-1.0, min(1.0, tilt)))
        zoom = float(max(-1.0, min(1.0, zoom)))

        trans_parts = []
        if pan != 0.0 or tilt != 0.0:
            trans_parts.append(
                (
                    f'<tt:PanTilt x="{pan:.3f}" y="{tilt:.3f}" '
                    f'space="http://www.onvif.org/ver10/tptz/PanTiltSpaces/TranslationGenericSpace" />'
                )
            )
        if zoom != 0.0:
            trans_parts.append(
                (
                    f'<tt:Zoom x="{zoom:.3f}" '
                    f'space="http://www.onvif.org/ver10/tptz/ZoomSpaces/TranslationGenericSpace" />'
                )
            )
        trans_xml = "\n".join(trans_parts)

        body = f"""
        <tptz:RelativeMove xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
                           xmlns:tt="http://www.onvif.org/ver10/schema">
          <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
          <tptz:Translation>
            {trans_xml}
          </tptz:Translation>
        </tptz:RelativeMove>
        """
        self._post_soap(
            ptz_xaddr,
            body,
            action="http://www.onvif.org/ver20/ptz/wsdl/RelativeMove",
        )

    def ptz_stop(
        self,
        profile_token: str,
        *,
        ptz_xaddr: str,
        pan_tilt: bool = True,
        zoom: bool = True,
    ) -> None:
        pan_tilt_xml = "true" if pan_tilt else "false"
        zoom_xml = "true" if zoom else "false"
        body = f"""
        <tptz:Stop xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
          <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
          <tptz:PanTilt>{pan_tilt_xml}</tptz:PanTilt>
          <tptz:Zoom>{zoom_xml}</tptz:Zoom>
        </tptz:Stop>
        """
        self._post_soap(
            ptz_xaddr,
            body,
            action="http://www.onvif.org/ver20/ptz/wsdl/Stop",
        )

    def ptz_goto_home(self, profile_token: str, *, ptz_xaddr: str) -> None:
        body = f"""
        <tptz:GotoHomePosition xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
          <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
        </tptz:GotoHomePosition>
        """
        self._post_soap(
            ptz_xaddr,
            body,
            action="http://www.onvif.org/ver20/ptz/wsdl/GotoHomePosition",
        )

    def ptz_get_presets(self, profile_token: str, *, ptz_xaddr: str) -> List[OnvifPreset]:
        body = f"""
        <tptz:GetPresets xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
          <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
        </tptz:GetPresets>
        """
        root = self._post_soap(
            ptz_xaddr,
            body,
            action="http://www.onvif.org/ver20/ptz/wsdl/GetPresets",
        )
        presets: list[OnvifPreset] = []
        for pe in root.findall(".//tptz:GetPresetsResponse/tptz:Preset", NS):
            tok = pe.attrib.get("token") or None
            if not tok:
                continue
            name_el = pe.find("./tt:Name", NS)
            name = (
                name_el.text.strip()
                if name_el is not None and name_el.text
                else None
            )
            presets.append(OnvifPreset(token=tok, name=name))
        return presets

    def ptz_supports_zoom(self, profile_token: Optional[str] = None, *, ptz_xaddr: str) -> bool:
        """
        Best-effort capability probe for whether this PTZ device exposes a zoom axis.

        Some devices advertise ZoomSpaces in GetNodes even when no zoom is usable. When a
        profile token is available, prefer GetStatus and only report zoom if the status
        response includes a Zoom position element.
        """
        body = """
        <tptz:GetNodes xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" />
        """
        root = self._post_soap(
            ptz_xaddr,
            body,
            action="http://www.onvif.org/ver20/ptz/wsdl/GetNodes",
        )
        has_zoom_spaces = False
        for spaces in root.findall(".//{*}SupportedPTZSpaces"):
            for el in spaces.iter():
                tag = getattr(el, "tag", "")
                if isinstance(tag, str) and "Zoom" in tag:
                    has_zoom_spaces = True
                    break
            if has_zoom_spaces:
                break
        if not has_zoom_spaces:
            return False

        if profile_token:
            body = f"""
            <tptz:GetStatus xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
              <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
            </tptz:GetStatus>
            """
            try:
                root = self._post_soap(
                    ptz_xaddr,
                    body,
                    action="http://www.onvif.org/ver20/ptz/wsdl/GetStatus",
                )
            except OnvifError:
                # Conservative: if we cannot confirm a zoom axis via status, hide zoom controls.
                return False

            zoom_el = root.find(
                ".//tptz:GetStatusResponse/tptz:PTZStatus/tt:Position/tt:Zoom",
                NS,
            )
            if zoom_el is not None:
                return True
            # Fallback for odd namespace/layouts
            if root.find(".//{*}PTZStatus//{*}Position//{*}Zoom") is not None:
                return True
            return False

        # Fallback: no profile token available, so rely on the spaces hint.
        return True

    def ptz_goto_preset(
        self,
        profile_token: str,
        preset_token: str,
        *,
        ptz_xaddr: str,
    ) -> None:
        body = f"""
        <tptz:GotoPreset xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
          <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
          <tptz:PresetToken>{preset_token}</tptz:PresetToken>
        </tptz:GotoPreset>
        """
        self._post_soap(
            ptz_xaddr,
            body,
            action="http://www.onvif.org/ver20/ptz/wsdl/GotoPreset",
        )

    # ------------------ internals ------------------ #
    def _post_soap(self, url: str, body: str, action: Optional[str] = None) -> ET.Element:
        header = ""
        if self.username and self.password:
            header = self._wsse_username_token_header()

        envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="{SOAP_ENV}">{header}
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
            host = urlparse(url).hostname or url
            raise OnvifAuthError(f"authentication required ({host})")
        if not (200 <= resp.status_code < 300):
            detail = ""
            # Try to surface SOAP Fault reason (often more useful than raw XML).
            try:
                root = ET.fromstring(resp.content)
                fault = root.find(".//{*}Fault")
                if fault is not None:
                    reason_el = fault.find(".//{*}Reason/{*}Text")
                    if reason_el is not None and reason_el.text:
                        detail = reason_el.text.strip()
                    if not detail:
                        fs = fault.find(".//{*}faultstring")
                        if fs is not None and fs.text:
                            detail = fs.text.strip()
                    if not detail:
                        # Last resort: collapse all fault text nodes.
                        parts = [t.strip() for t in fault.itertext() if t and t.strip()]
                        if parts:
                            detail = " ".join(parts)
            except Exception:
                detail = ""

            if not detail:
                try:
                    detail = (resp.text or "").strip()
                except Exception:
                    detail = ""
            if not detail:
                detail = "unknown error"
            if len(detail) > 400:
                detail = detail[:400] + "..."
            raise OnvifHttpError(resp.status_code, detail)
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
    "OnvifHttpError",
    "OnvifProfile",
    "OnvifCapabilities",
    "OnvifDeviceInfo",
    "OnvifPreset",
    "OnvifClient",
]
