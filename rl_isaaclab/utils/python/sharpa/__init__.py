# sharpa package __init__.py
import sys
import os
import ctypes

# Get the parent directory path - need to go up 3 levels: sharpa -> python -> SharpaWaveSDK
# Current file: SharpaWaveSDK/python/sharpa/__init__.py
# Target lib:   SharpaWaveSDK/lib/
sdk_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
lib_dir = os.path.join(sdk_root, 'lib')

# Set LD_LIBRARY_PATH for child processes
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if lib_dir not in current_ld_path.split(':'):
    os.environ['LD_LIBRARY_PATH'] = lib_dir + ':' + current_ld_path

# Preload required libraries using ctypes in correct dependency order
# Order matters! Dependencies must be loaded before libraries that depend on them
required_libs = [
    'libfmt.so.10',         # Base dependency
    'libmsgpack11.so',      # Base dependency
    'libyaml-cpp.so.0.8',   # Base dependency  
    'libspdlog.so.1.12',    # Base dependency
    'libturbojpeg.so.0',    # Base dependency
    'libtactile_sdk.so',    # Depends on msgpack11, turbojpeg, fmt
    'libSharpaHand.so.1',   # Depends on yaml-cpp, spdlog, fmt
    'libSharpaWaveSDK.so',  # Depends on SharpaHand, tactile_sdk
]

for lib_name in required_libs:
    lib_path = os.path.join(lib_dir, lib_name)
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            # If loading fails, try without RTLD_GLOBAL
            try:
                ctypes.CDLL(lib_path)
            except OSError:
                print(f"Warning: Could not preload library {lib_name}: {e}")
    else:
        print(f"Warning: Library {lib_name} not found at {lib_path}")

# Now load the Python module
python_version = f"python{sys.version_info.major}{sys.version_info.minor}"
python_dir = os.path.join(sdk_root, 'python')
module_path = os.path.join(python_dir, python_version, "sharpa.so")

if os.path.exists(module_path):
    import importlib.util
    import importlib.machinery
    
    loader = importlib.machinery.ExtensionFileLoader("sharpa", module_path)
    spec = importlib.util.spec_from_loader("sharpa", loader)
    sharpa_module = importlib.util.module_from_spec(spec)
    
    spec.loader.exec_module(sharpa_module)
    
    # Export all public attributes from the module
    for attr_name in dir(sharpa_module):
        if not attr_name.startswith('_'):
            globals()[attr_name] = getattr(sharpa_module, attr_name)
else:
    raise ImportError(f"sharpa.so not found at {module_path}")