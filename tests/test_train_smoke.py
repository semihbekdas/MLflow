import os
import sys
import subprocess

def test_smoke_train():
    # Run a minimal training to ensure pipeline executes without error
    cmd = [sys.executable, "-m", "src.data.make_dataset", "--config", "configs/config_titanic.yaml"]
    subprocess.check_call(cmd)

    cmd = [sys.executable, "-m", "src.models.train", "--config", "configs/config_titanic.yaml", "--run-name", "smoke"]
    subprocess.check_call(cmd)

    assert os.path.exists("mlruns"), "mlruns directory should be created"
