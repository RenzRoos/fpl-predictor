import sys 

from utils.main import main as run_main
from utils.evaluate import evaluate as run_evaluate
from utils.scout_get_data import scout_get_data as run_scout

def m(gw: int, end: int, i: int) -> int:
    if len(sys.argv) > i + 1:
        if sys.argv[i + 1].isdigit():
            end = int(sys.argv[i + 1])

            for j in range(gw, end + 1):
                run_main(j)

        else:
            raise SystemExit("End GW must be an integer.")
    else:
        raise SystemExit("End GW must be specified after -m.")
    
    return end, i+1

def e(gw: int, end: int):
    if end == -1:
        run_evaluate(gw)
    else:
        for j in range(gw, end + 1):
            run_evaluate(j)

def s(gw: int, end: int, i: int):
    evaluate = False
    if sys.argv[i] == "-se":
        evaluate = True

    if end == -1:
        run_scout(gw, evaluate)
    else:
        for j in range(gw, end + 1):
            run_scout(j, evaluate)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "-e" or sys.argv[1] == "-s":
        raise SystemExit("Usage: run.py <GW> -m [end] -e -s[e]")
    
    if not sys.argv[1].isdigit():
        raise SystemExit("GW must be an integer.")
    gw = int(sys.argv[1])

    end = -1

    if len(sys.argv) > 2:
        i = 2
        while i in range(len(sys.argv)):
            if sys.argv[i] not in ["-e", "-s", "-se", "-m"] and not sys.argv[i].isdigit():
                raise SystemExit(f"Unknown argument: {sys.argv[i]}")
        
            if "-m" not in sys.argv and (sys.argv[i] == "-e" or sys.argv[i] == "-s" or sys.argv[i] == "-se") and i == 2:
                run_main(gw)

            elif sys.argv[i] == "-m":
                end, i = m(gw, end, i)              

            elif sys.argv[i] == "-e":
                e(gw, end)

            elif sys.argv[i] in ["-s", "-se"]:
                s(gw, end, i)
            
            i += 1
    else:
        run_main(gw)



