import os
import subprocess
import math
import shutil
import csv
import sys

# --- CONFIGURATION ---
FLUENT_CMD = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\fluent\ntbin\win64\fluent.exe"
PROJECT_ROOT = r"D:\cfd-ml-project"

# Paths
BASE_CASE = os.path.join(PROJECT_ROOT, "ansys_cases", "airfoil_base.cas.h5")
TEMPLATE_JOU = os.path.join(PROJECT_ROOT, "journals", "run_airfoil_template.jou")
CASES_CSV = os.path.join(PROJECT_ROOT, "cases_to_run.csv")

# UPDATED: Pointing to "data/raw" for Day 7/8 standard
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def calculate_vectors(aoa_degrees):
    rad = math.radians(float(aoa_degrees))
    return math.cos(rad), math.sin(rad), -math.sin(rad)

def run_case(case_data):
    case_id = case_data['case_id']
    aoa = case_data['aoa']
    
    print(f"--- Starting Case: {case_id} (AoA = {aoa} deg) ---")
    
    # 1. Prepare Journal
    vx, vy, vy_neg = calculate_vectors(aoa)
    
    if not os.path.exists(TEMPLATE_JOU):
        print(f"❌ Error: Template {TEMPLATE_JOU} missing.")
        return False
        
    with open(TEMPLATE_JOU, 'r') as f:
        template = f.read()
    
    jou_content = template.replace("<BASE_CASE_PATH>", BASE_CASE) \
                          .replace("<VX>", f"{vx:.6f}") \
                          .replace("<VY>", f"{vy:.6f}") \
                          .replace("<VY_NEG>", f"{vy_neg:.6f}") \
                          .replace("<OUTPUT_PREFIX>", case_id)
    
    run_jou_path = os.path.join(LOG_DIR, f"run_{case_id}.jou")
    with open(run_jou_path, 'w') as f:
        f.write(jou_content)
    
    # 2. Run Fluent
    log_path = os.path.join(LOG_DIR, f"log_{case_id}.txt")
    cmd = f'"{FLUENT_CMD}" 2d -g -t4 -i "{run_jou_path}"'
    
    print(f"   Launching Fluent... Logs: {log_path}")
    
    with open(log_path, "w") as log_handle:
        try:
            subprocess.run(cmd, shell=True, stdout=log_handle, stderr=subprocess.STDOUT, timeout=7200)
        except subprocess.TimeoutExpired:
            print("   ⚠️ Timeout expired!")
            return False

    # 3. Check Convergence
    with open(log_path, "r") as f:
        if "solution is converged" in f.read():
            print("   ✅ Converged.")
        else:
            print("   ⚠️ Reached max iterations.")

    # 4. Move Files (Removed 'forces_*.txt')
    case_dir = os.path.join(OUTPUT_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)
    
    files_to_move = [
        f"cp_upper_{case_id}.csv", 
        f"cp_lower_{case_id}.csv", 
        f"history_{case_id}.out",       # We keep this one for Cl/Cd
        f"naca0012_{case_id}.cas.h5",
        f"naca0012_{case_id}.dat.h5"
    ]
    
    moved_count = 0
    for fname in files_to_move:
        src = os.path.join(PROJECT_ROOT, fname)
        dst = os.path.join(case_dir, fname)
        
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_count += 1
        else:
            if not os.path.exists(dst):
                print(f"   ⚠️ Missing file: {fname}")

    if moved_count > 0:
        print(f"   ✅ Moved {moved_count} files to {case_dir}/")
        return True
    else:
        print("   ❌ No output files found.")
        return False

if __name__ == "__main__":
    if not os.path.exists(CASES_CSV):
         print(f"❌ Error: {CASES_CSV} not found.")
         sys.exit(1)

    print(f"🚀 Batch Run Started. Outputs will go to: {OUTPUT_DIR}")
    with open(CASES_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            run_case(row)