import subprocess

bash_path = r"C:\Program Files\Git\bin\bash.exe"
script_path = r"C:\Users\Admin\Desktop\wget_script_2025-7-27_13-21-28.sh"

result = subprocess.run(
    [bash_path, script_path],
    capture_output=True,
    text=True,
    encoding="utf-8",  # or "latin1" if needed
    errors="ignore"
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

if result.returncode != 0:
    print("Script failed.")
else:
    print("Script executed successfully.")
