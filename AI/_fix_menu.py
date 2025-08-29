from pathlib import Path

p = Path('AI/mdi_app.py')
data = p.read_bytes()
lines = data.splitlines(keepends=True)

def eol_of(idx):
    try:
        return b"\r\n" if lines[idx].endswith(b"\r\n") else b"\n"
    except Exception:
        return b"\n"

# Adjust specific lines (1-based indexing in editor: 858, 859)
idx_settings_add = 857  # zero-based
idx_settings_conn = 858

if len(lines) > idx_settings_conn:
    eol = eol_of(idx_settings_add)
    lines[idx_settings_add] = b"        act_settings = menu_tools.addAction('Settings...')" + eol
    lines[idx_settings_conn] = b"        act_settings.triggered.connect(self.open_settings)" + eol
    p.write_bytes(b"".join(lines))
    print('Patched menu settings lines.')
else:
    print('File shorter than expected; no changes applied.')

