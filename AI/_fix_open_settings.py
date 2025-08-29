from pathlib import Path

p = Path('AI/mdi_app.py')
data = p.read_bytes()
lines = data.splitlines(keepends=True)

def eol_of(idx):
    try:
        return b"\r\n" if lines[idx].endswith(b"\r\n") else b"\n"
    except Exception:
        return b"\n"

# Rebuild the open_settings block at approx lines 1216..1185 (current indexing may vary)
start = 1215 - 1  # 0-based index where the header comment is
eol = eol_of(start)

block = [
    b"        # -------- Settings dialog --------" + eol,
    b"        def open_settings(self):" + eol,
    b"            dlg = SettingsDialog(self)" + eol,
    b"            # prefill from current state" + eol,
    b"            dlg.s_det_interval.setValue(getattr(self, '_det_interval', 200))" + eol,
    b"            dlg.s_hash_size.setValue(getattr(self, '_hash_size', 8))" + eol,
    b"            dlg.s_hamming.setValue(getattr(self, '_hamming', 4))" + eol,
    b"            dlg.s_ptz_interval.setValue(getattr(self, '_ptz_interval', 300))" + eol,
    b"            dlg.s_deadzone.setValue(getattr(self, '_deadzone_pct', 5))" + eol,
    b"            if dlg.exec() == QtWidgets.QDialog.Accepted:" + eol,
    b"                self._det_interval = dlg.s_det_interval.value()" + eol,
    b"                self._hash_size = dlg.s_hash_size.value()" + eol,
    b"                self._hamming = dlg.s_hamming.value()" + eol,
    b"                self._ptz_interval = dlg.s_ptz_interval.value()" + eol,
    b"                self._deadzone_pct = dlg.s_deadzone.value()" + eol,
    b"                # apply to open cameras" + eol,
    b"                for sub in self.mdi.subWindowList():" + eol,
    b"                    w=sub.widget()" + eol,
    b"                    if isinstance(w, CameraWidget):" + eol,
    b"                        w.det_thr.interval = self._det_interval" + eol,
    b"                        w.det_thr.max_skip_cycles = 1" + eol,
    b"                        w._aim_timer.setInterval(self._ptz_interval)" + eol,
    b"                QtWidgets.QMessageBox.information(self, 'Settings', 'Settings applied')" + eol,
]

# Replace lines from 'start' until before next blank line or until count fits the previous block size
end = start
while end < len(lines) and (end - start) < 30:
    # stop when encountering a blank line after starting replacements and at least 3 lines consumed
    if end > start and lines[end].strip() == b"":
        break
    end += 1

lines[start:end] = block
p.write_bytes(b"".join(lines))
print(f"Rewrote open_settings block at lines {start+1}..{start+len(block)}")

