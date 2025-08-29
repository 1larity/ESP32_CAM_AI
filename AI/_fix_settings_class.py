from pathlib import Path

p = Path('AI/mdi_app.py')
data = p.read_bytes()
lines = data.splitlines(keepends=True)

def find_index(sub: bytes):
    for i, ln in enumerate(lines):
        if sub in ln:
            return i
    return -1

start = find_index(b'class Settings')
end = find_index(b'class CullDialog')
if start != -1 and end != -1 and end > start:
    eol = b"\r\n" if lines[start].endswith(b"\r\n") else b"\n"
    block = [
        b"class SettingsDialog(QtWidgets.QDialog):" + eol,
        b"    \"\"\"App settings: detector interval, aHash grid size + Hamming, PTZ aim timing, deadzone.\"\"\"" + eol,
        b"    def __init__(self, parent=None):" + eol,
        b"        super().__init__(parent)" + eol,
        b"        self.setWindowTitle('Settings')" + eol,
        b"        self.resize(420, 240)" + eol,
        b"        form = QtWidgets.QFormLayout(self)" + eol,
        b"        self.s_det_interval = QtWidgets.QSpinBox(); self.s_det_interval.setRange(50, 2000); self.s_det_interval.setSingleStep(50); self.s_det_interval.setSuffix(' ms')" + eol,
        b"        self.s_hash_size = QtWidgets.QSpinBox(); self.s_hash_size.setRange(4, 16)" + eol,
        b"        self.s_hamming = QtWidgets.QSpinBox(); self.s_hamming.setRange(0, 32)" + eol,
        b"        self.s_ptz_interval = QtWidgets.QSpinBox(); self.s_ptz_interval.setRange(100, 2000); self.s_ptz_interval.setSingleStep(50); self.s_ptz_interval.setSuffix(' ms')" + eol,
        b"        self.s_deadzone = QtWidgets.QSpinBox(); self.s_deadzone.setRange(2, 20); self.s_deadzone.setSuffix(' %')" + eol,
        b"        form.addRow('Detector interval', self.s_det_interval)" + eol,
        b"        form.addRow('aHash grid size', self.s_hash_size)" + eol,
        b"        form.addRow('Hamming threshold', self.s_hamming)" + eol,
        b"        form.addRow('PTZ aim interval', self.s_ptz_interval)" + eol,
        b"        form.addRow('PTZ deadzone', self.s_deadzone)" + eol,
        b"        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)" + eol,
        b"        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)" + eol,
        b"        form.addRow(btns)" + eol,
        b"" + eol,
    ]
    lines[start:end] = block
    p.write_bytes(b"".join(lines))
    print(f"Rewrote SettingsDialog class at lines {start+1}..{start+len(block)}")
else:
    print('Could not locate Settings class region to patch')

