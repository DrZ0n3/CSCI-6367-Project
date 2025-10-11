# setup_env.py
import os
import sys
import subprocess

def main():
    print("Setting up virtual environment...")

    venv_dir = "venv"

    # 1. Create virtual environment
    if not os.path.exists(venv_dir):
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    else:
        print(f"✅ Virtual environment '{venv_dir}' already exists.")

    # 2. Determine platform-specific pip path
    if os.name == "nt":  # Windows
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Linux / macOS
        pip_executable = os.path.join(venv_dir, "bin", "pip")
        python_executable = os.path.join(venv_dir, "bin", "python")

    # 3. Upgrade pip
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])

    # 4. Install dependencies
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])

    print("Setup complete!")
    print(f"To activate the virtual environment, run:")
    if os.name == "nt":
        print(r"  call venv\Scripts\activate")
    else:
        print("  source venv/bin/activate")

if __name__ == "__main__":
    main()
