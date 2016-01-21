import os

cmd_shell = "python fool.py"
run_range = range(1)

optimization=True
wordnet=False
run_len = 80001
lighting=True
fixed_bg=False

#os.system("sudo /usr/bin/X :1 &")

for run in run_range:
 cmd = "DISPLAY=:1 " + cmd_shell[:]
 cmd += " --seed %d" % (run + 10000)
 if optimization:
  cmd += " --map_opt"
 if wordnet:
  cmd += " --wordnet"
 if not lighting:
  cmd += " --no_lighting"
 if fixed_bg:
  cmd += " --fixed_bg"
 cmd += " --run_len %d" % run_len
 print cmd
 os.system(cmd)
