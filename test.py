import sys, os

output_folder = sys.argv[1]

output_file = os.path.join(output_folder, "slideID_Summary.xlsx")

with open(output_file, "w") as f:
    f.write("Hello World!")