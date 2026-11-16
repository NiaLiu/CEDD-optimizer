
file1 = open('temp.txt', 'r')
Lines = file1.readlines()
start = 1826
end = 1832
flag = False
accumu = 0.0
count = 0

def is_float(string):
    string_copy = string
    if string_copy.replace('.', '').replace(' ', '').isnumeric():
        return True
    else:
        return False
for i,line in enumerate(Lines):
    if i == start:
        flag = True
    elif i == end:
        flag = False
    if flag == True:
        value = line[74:80]
        if is_float(value):
            accumu += float(value)
            count += 1
            continue
        else:
            continue
print(accumu, count)

file1.close()

