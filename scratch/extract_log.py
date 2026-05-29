def extract():
    log_path = r"c:\Personal Project\bot-crypto\logs\bot.log"
    import re
    # We want to match timestamps between 2026-05-22 04:45:00 and 2026-05-22 04:55:00
    pattern = re.compile(r"^2026-05-22 04:(4[5-9]|5[0-5]):")
    
    output = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if pattern.match(line):
                # Only print lines mentioning FIDAUSDT or containing TRADE CLOSED / Position opened
                if "FIDAUSDT" in line or "CLOSED" in line or "opened" in line:
                    output.append(line.strip())
                    
    with open(r"c:\Personal Project\bot-crypto\scratch\log_extracted.txt", "w", encoding="utf-8") as out:
        out.write("\n".join(output))
    print("Done writing to scratch/log_extracted.txt")

if __name__ == "__main__":
    extract()
