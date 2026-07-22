import subprocess
import os
import sys

def main():
    project_dir = r"c:\Users\KalleChakradhar\Downloads\music\Retrieval-based-Voice-Conversion-WebUI"
    logs_dir = os.path.join(project_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    out_log = os.path.join(logs_dir, "webui_launch_out.log")
    err_log = os.path.join(logs_dir, "webui_launch_err.log")

    # Locate the real python interpreter (bypassing the virtualenv shim)
    python_exe = getattr(sys, "_base_executable", sys.executable)

    # Set up PYTHONPATH to include the virtualenv's site-packages so the base interpreter can load Gradio, Torch, etc.
    venv_site_packages = os.path.join(project_dir, ".venv", "Lib", "site-packages")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([project_dir, venv_site_packages])

    print(f"Spawning detached WebUI server (base interpreter)...")
    print(f"Python Executable: {python_exe}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    print(f"Stdout log: {out_log}")
    print(f"Stderr log: {err_log}")

    creation_flags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS = 0x00000008, CREATE_NEW_PROCESS_GROUP = 0x00000200
        creation_flags = 0x00000008 | 0x00000200

    # Open log files and keep handles open in parent for child to inherit
    out = open(out_log, "a", encoding="utf-8")
    err = open(err_log, "a", encoding="utf-8")

    p = subprocess.Popen(
        [
            python_exe,
            "-u",
            "webui.py",
            "--pycmd",
            sys.executable,  # We can still pass the shim for sub-processes
            "--port",
            "7897",
            "--noautoopen"
        ],
        cwd=project_dir,
        env=env,
        stdout=out,
        stderr=err,
        creationflags=creation_flags
    )
    
    print(f"WebUI server successfully spawned with detached PID: {p.pid}")

if __name__ == "__main__":
    main()
