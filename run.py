import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-e" or sys.argv[1] == "-s":
        raise SystemExit("Usage: scout_get_data.py <GW> -e -s[e]")
    
    if not sys.argv[1].isdigit():
        raise SystemExit("GW must be an integer.")
    gw = int(sys.argv[1])

    cmd = f"python3 main.py {gw};"

    if len(sys.argv) > 2:
        if sys.argv[2] == "-e":
            cmd += f"python3 evaluate.py {gw};"

            if len(sys.argv) > 3:
                if sys.argv[3] == "-s":
                    cmd += f"python3 scout_get_data.py {gw}; python3 compare.py {gw};"
                elif sys.argv[3] == "-se":
                    cmd += f"python3 scout_get_data.py {gw} -e; python3 compare.py {gw};"    

os.system(cmd)



