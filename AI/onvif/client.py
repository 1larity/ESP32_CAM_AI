from __future__ import annotations

import importlib.util
import os
import site
import sys
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests

# Cache pip-installed onvif-zeep modules separately so we can use them without shadowing the local AI/onvif package.
_PIP_ONVIF_MODULES: dict[str, object] = {}


def _load_onvif_zeep_camera():
    """
    Explicitly load onvif-zeep from site-packages, avoiding the local AI/onvif package.
    Returns ONVIFCamera class or None if not available.
    """
    if _PIP_ONVIF_MODULES:
        mod = _PIP_ONVIF_MODULES.get("onvif")
        if mod:
            return getattr(mod, "ONVIFCamera", None)

    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except Exception:
        pass

    ai_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    original_path = list(sys.path)
    original_onvif_modules = {k: v for k, v in sys.modules.items() if k == "onvif" or k.startswith("onvif.")}

    for base in candidates:
        if not base or not os.path.isdir(base):
            continue
        search_roots = {base, os.path.join(base, "Lib", "site-packages")}
        for root in search_roots:
            init_path = os.path.join(root, "onvif", "__init__.py")
            if not os.path.isfile(init_path):
                continue
            try:
                if os.path.samefile(init_path, os.path.join(ai_dir, "onvif", "__init__.py")):
                    continue
            except Exception:
                pass

            # Prepare an isolated module set so the pip package can import its siblings.
            backup_modules = {k: v for k, v in sys.modules.items() if k == "onvif" or k.startswith("onvif.")}
            for k in list(backup_modules.keys()):
                sys.modules.pop(k, None)
            try:
                cleaned_path = [root] + [p for p in original_path if os.path.abspath(p) != ai_dir]
                sys.path = cleaned_path
                spec = importlib.util.spec_from_file_location("onvif", init_path)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                module.__path__ = [os.path.dirname(init_path)]
                sys.modules["onvif"] = module
                spec.loader.exec_module(module)
                camera = getattr(module, "ONVIFCamera", None)
                if camera:
                    pip_modules = {k: v for k, v in sys.modules.items() if k == "onvif" or k.startswith("onvif.")}
                    _PIP_ONVIF_MODULES.update(pip_modules)
                    return camera
            except Exception:
                continue
            finally:
                sys.path = original_path
                for k in list(sys.modules.keys()):
                    if k == "onvif" or k.startswith("onvif."):
                        sys.modules.pop(k, None)
                sys.modules.update(backup_modules)

    # Restore original modules if nothing was loaded.
    for k in list(sys.modules.keys()):
        if k == "onvif" or k.startswith("onvif."):
            sys.modules.pop(k, None)
    sys.modules.update(original_onvif_modules)
    return None

try:
    # Heavy optional dependency for full ONVIF support (pip package onvif-zeep)
    ONVIFCamera = _load_onvif_zeep_camera()
except Exception:  # pragma: no cover
    ONVIFCamera = None

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

    def __init__(self, xaddr: str, *, username: Optional[str] = None, password: Optional[str] = None, session: Optional[requests.Session] = None):
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
        media_xaddr = self._find_text(root, ".//tds:Capabilities/tds:Media/tt:XAddr") or None
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
            manufacturer=self._find_text(root, ".//tds:GetDeviceInformationResponse/tds:Manufacturer"),
            model=self._find_text(root, ".//tds:GetDeviceInformationResponse/tds:Model"),
            firmware=self._find_text(root, ".//tds:GetDeviceInformationResponse/tds:FirmwareVersion"),
            serial=self._find_text(root, ".//tds:GetDeviceInformationResponse/tds:SerialNumber"),
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
            name = name_el.text.strip() if name_el is not None and name_el.text else token or "Profile"
            if token:
                profiles.append(OnvifProfile(token=token, name=name))
        return profiles

    def get_stream_uri(self, profile_token: str, media_xaddr: Optional[str] = None) -> Optional[str]:
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
                uri = self._find_text(root, ".//trt:GetStreamUriResponse/trt:MediaUri/tt:Uri")
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


# ---------- optional zeep-based helper (richer compatibility) ---------- #
def try_onvif_zeep_stream(xaddr: str, user: Optional[str], password: Optional[str]) -> tuple[list[dict], Optional[str], list[str]]:
    """
    Attempt to use onvif-zeep's ONVIFCamera to fetch profiles and stream URI.
    Returns (profiles, stream_uri, errors).
    profiles: list of {"token", "name"}.
    stream_uri: RTSP URL or None.
    errors: list of error strings.
    """
    errs: list[str] = []
    camera_cls = ONVIFCamera or _load_onvif_zeep_camera()
    if camera_cls is None:
        errs.append("onvif-zeep not installed")
        return [], None, errs
    # Determine WSDL directory from the pip-installed onvif package.
    wsdl_dir = None
    possible_mods = []
    if _PIP_ONVIF_MODULES:
        mod = _PIP_ONVIF_MODULES.get("onvif")
        if mod:
            possible_mods.append(mod)
    possible_mods.append(sys.modules.get("onvif"))
    ai_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    local_wsdl = os.path.join(ai_dir, "onvif", "wsdl")
    for mod in possible_mods:
        if mod and getattr(mod, "__file__", None):
            cand = os.path.join(os.path.dirname(mod.__file__), "wsdl")
            if os.path.isdir(cand):
                wsdl_dir = cand
                break
    if wsdl_dir is None and os.path.isdir(local_wsdl):
        wsdl_dir = local_wsdl
    if wsdl_dir is None or not os.path.isdir(wsdl_dir):
        errs.append("onvif-zeep wsdl folder missing; copy WSDLs into AI/onvif/wsdl or site-packages/onvif/wsdl")
        return [], None, errs

    parsed = urlparse(xaddr)
    host = parsed.hostname or ""
    if not host:
        errs.append("xaddr missing host")
        return [], None, errs
    port = parsed.port or 80
    path = parsed.path or "/onvif/device_service"
    # Temporarily install the pip onvif modules so ONVIFCamera internals can import siblings.
    backup_modules: Optional[dict[str, object]] = None
    if _PIP_ONVIF_MODULES:
        backup_modules = {k: sys.modules.get(k) for k in _PIP_ONVIF_MODULES.keys()}
        sys.modules.update(_PIP_ONVIF_MODULES)

    try:
        cam = camera_cls(host, port, user or "", password or "", wsdl_dir or None)
        media = cam.create_media_service()
        profiles_raw = media.GetProfiles()
        profiles: list[dict] = []
        for p in profiles_raw:
            token = getattr(p, "token", None)
            name = getattr(p, "Name", None) or token
            if token:
                profiles.append({"token": token, "name": name})
        stream_uri = None
        if profiles:
            for protocol in ("TCP", "RTSP"):
                try:
                    req = {
                        "StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": protocol}},
                        "ProfileToken": profiles[0]["token"],
                    }
                    uri_obj = media.GetStreamUri(req)
                    uri_val = getattr(uri_obj, "Uri", None)
                    if uri_val:
                        stream_uri = uri_val
                        break
                except Exception as e:  # pragma: no cover - device specific
                    errs.append(f"zeep GetStreamUri {protocol}: {e}")
                    continue
        return profiles, stream_uri, errs
    except Exception as e:  # pragma: no cover - device specific
        errs.append(f"zeep init failed: {e}")
        return [], None, errs
    finally:
        if backup_modules is not None:
            for k, v in backup_modules.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
