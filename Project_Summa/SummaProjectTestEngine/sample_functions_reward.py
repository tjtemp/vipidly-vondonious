import subprocess
cmd = ['python3', 'sample_functions_test.py']
pipe = subprocess.Popen(cmd , stdout=subprocess.PIPE).stdout
output = pipe.read()
print(output)
output = pipe.read()
#output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
# output = pipe.read()
print(output)
