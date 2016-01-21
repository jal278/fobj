import os

cmd_shell = "python fool.py"
run_range = range(5,10)

optimization=False
wordnet=False
lighting=False
fixed_bg=True

os.system("sudo /usr/bin/X :1 &")

for run in run_range:
 cmd = "DISPLAY=:1 " + cmd_shell[:]
 cmd += " --seed %d" % (run + 1000)
 if optimization:
  cmd += " --map_opt"
 if wordnet:
  cmd += " --wordnet"
 if not lighting:
  cmd += " --no_lighting"
 if fixed_bg:
  cmd += " --fixed_bg"
 print cmd
 os.system(cmd)
