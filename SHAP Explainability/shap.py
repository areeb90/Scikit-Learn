# Run once, then continue — no kernel restart needed
import subprocess
subprocess.run(["pip", "install", "shap", "--quiet"], check=True)
print("shap ready")