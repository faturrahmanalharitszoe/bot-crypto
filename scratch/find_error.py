def find_error():
    log_path = r"c:\Personal Project\bot-crypto\logs\bot.log"
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if "Unexpected error in main loop" in line:
            print(f"Found error at line {i+1}:")
            # Print the error line and the next 20 lines (traceback) safely
            for j in range(i, min(i + 25, len(lines))):
                safe_line = lines[j].strip().encode('ascii', 'replace').decode('ascii')
                print(safe_line)
            print("=" * 60)

if __name__ == "__main__":
    find_error()
