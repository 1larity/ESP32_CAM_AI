import importlib.util, sys
spec = importlib.util.spec_from_file_location('mdi','AI/mdi_app.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
MW = getattr(m, 'MainWindow', None)
print('MainWindow:', bool(MW))
if MW:
    for attr in ['import_faces','import_pets','cull_faces','cull_pets','open_settings']:
        print(attr, hasattr(MW, attr))
