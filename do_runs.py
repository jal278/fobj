import os

cmd_shell = "python fool.py"
run_range = range(0,5)

optimization=False
wordnet=False

for run in run_range:
 cmd = cmd_shell[:]
 cmd += " --seed %d" % (run + 1000)
 if optimization:
  cmd += " --map_opt"
 if wordnet:
  cmd += " --wordnet"
 print cmd
 os.system(cmd)
