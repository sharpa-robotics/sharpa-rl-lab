# sharpa package __init__.py
import sys
import os
import ctypes

# Get the parent directory path - need to go up 3 levels: sharpa -> python -> SharpaWaveSDK
# Current file: .../SharpaWaveSDK/python/sharpa/__init__.py
# Target lib:   .../SharpaWaveSDK/lib/
sdk_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
lib_dir = os.path.join(sdk_root, 'lib')

# Set LD_LIBRARY_PATH for child processes
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if lib_dir not in current_ld_path.split(':'):
    os.environ['LD_LIBRARY_PATH'] = lib_dir + ':' + current_ld_path

# Try to find Python library in standard locations
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
python_lib_name = f"libpython{python_version}.so.1.0"
python_lib_paths = [
    f"/usr/lib/x86_64-linux-gnu/{python_lib_name}",
    f"/usr/local/lib/{python_lib_name}",
    f"/usr/lib/{python_lib_name}",
    os.path.join(sys.prefix, 'lib', python_lib_name)
]

# Try to load Python library first
for lib_path in python_lib_paths:
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            print(f"✓ Loaded Python library from: {lib_path}")
            break
        except OSError as e:
            print(f"Warning: Failed to load {lib_path}: {e}")

# Preload required libraries using ctypes in correct dependency order
# Order matters! Dependencies must be loaded before libraries that depend on them
required_libs = [
    'libSharpaWaveSDKWrapper.so',  # Depends on SharpaHand, tactile_sdk
]

for lib_name in required_libs:
    lib_path = os.path.join(lib_dir, lib_name)
    if os.path.exists(lib_path):
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            print(f"✓ Loaded {lib_name}")
        except OSError as e:
            # If loading fails, try without RTLD_GLOBAL
            try:
                ctypes.CDLL(lib_path)
                print(f"✓ Loaded {lib_name} (without RTLD_GLOBAL)")
            except OSError:
                print(f"Warning: Could not load library {lib_name}: {e}")
    else:
        print(f"Warning: Library {lib_name} not found at {lib_path}")

# Now load the Python module
python_version = f"python{sys.version_info.major}{sys.version_info.minor}"
python_dir = os.path.join(sdk_root, 'python')
module_path = os.path.join(python_dir, python_version, "sharpa.so")

if os.path.exists(module_path):
    import importlib.util
    import importlib.machinery
    
    try:
        # Try direct import first
        from sharpa.sharpa import *
        print("✓ Successfully imported module using direct import")
    except ImportError:
        # If direct import fails, try manual loading
        print("Direct import failed, trying manual loading...")
        loader = importlib.machinery.ExtensionFileLoader("sharpa", module_path)
        spec = importlib.util.spec_from_loader("sharpa", loader)
        print(f"Spec: {spec}")
        sharpa_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sharpa_module)
        
        # Export all public attributes from the module
        for attr_name in dir(sharpa_module):
            if not attr_name.startswith('_'):
                globals()[attr_name] = getattr(sharpa_module, attr_name)
        print("✓ Successfully loaded module manually")
else:
    raise ImportError(f"sharpa.so not found at {module_path}")

# Verify that key classes are available
if 'SharpaWaveManager' not in globals():
    print("\nWarning: SharpaWaveManager not found in globals")
    print("Available symbols:", [name for name in globals() if not name.startswith('_')])