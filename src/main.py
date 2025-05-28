import os

def run_all(run_Chapter_4 = False, run_Chapter_5 = False, run_Chapter_6 = False):
    if run_Chapter_4:
        print("=== Starting Chapter 4 Processing ===")
        os.system("python3 -m Chapter4/feature_engineering.py")
        os.system("python3 -m Chapter4/pipelineP1.py")
        os.system("python3 -m Chapter4/pipeline_tuningP1.py")
        os.system("python3 -m Chapter4/plot.py")
        os.system("python3 -m Chapter4/plot_tuning.py")
        os.system("python3 -m Chapter4/figure1.py")
        os.system("python3 -m Chapter4/figure2.py")
        os.system("python3 -m Chapter4/figure3.py")
        os.system("python3 -m Chapter4/figure4.py")
    if run_Chapter_5:
        print("\n=== Starting Chapter 5 Processing ===")
        os.system("python3 -m Chapter5/feature_engineering_time.py")
        os.system("python3 -m Chapter5/pipeline_time.py")
        os.system("python3 -m Chapter5/pipeline_time_tuning.py")
        os.system("python3 -m Chapter5/figure5.py")
        os.system("python3 -m Chapter5/figure6.py")
        os.system("python3 -m Chapter5/figure7.py")
        os.system("python3 -m Chapter5/figure8.py")
        os.system("python3 -m Chapter5/plot_china_case.py")
        os.system("python3 -m Chapter5/plot_us_case.py")
    if run_Chapter_6:
        print("\n=== Starting Chapter 6 (if applicable) ===")
        # os.system("python3 Chapter6/your_script_here.py")  # placeholder

if __name__ == "__main__":
    run_all(run_Chapter_4 = True, run_Chapter_5 = True, run_Chapter_6 = False)