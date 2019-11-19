import os
import platform
from dotenv import load_dotenv


def load_env():
    load_dotenv()

    # Sets default by OS if DONKEY_SIM_PATH is not defined in .env
    if os.environ.get('DONKEY_SIM_PATH') is None:
        operating_system = platform.system()
        if operating_system == "Darwin":
            os.environ['DONKEY_SIM_PATH'] = "./donkey_sim.app/Contents/MacOS/donkey_sim"
        elif operating_system == "Windows":
            os.environ['DONKEY_SIM_PATH'] = "./DonkeySimWindows/DonkeySim.exe"
        elif operating_system == "Linux":
            os.environ['DONKEY_SIM_PATH'] = "./DonkeySimLinux/donkey_sim.x86_64"
        else:
            raise ValueError("Unknown operating")

    print(f"Set DONKEY_SIM_PATH to {os.environ.get('DONKEY_SIM_PATH')}")
    print(f"Set DONKEY_SIM_PORT to {os.environ.get('DONKEY_SIM_PORT')}")
    print(f"Set DONKEY_SIM_HEADLESS to {os.environ.get('DONKEY_SIM_HEADLESS')}")

