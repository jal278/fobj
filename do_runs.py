import os

cmd_shell = "python fool.py"
run_range = range(0,5)

optimization=False
wordnet=False

os.system("sudo /usr/bin/X :1 &")

for run in run_range:
 cmd = "DISPLAY=:1 " + cmd_shell[:]
 cmd += " --seed %d" % (run + 1000)
 if optimization:
  cmd += " --map_opt"
 if wordnet:
  cmd += " --wordnet"
 print cmd
 os.system(cmd)
