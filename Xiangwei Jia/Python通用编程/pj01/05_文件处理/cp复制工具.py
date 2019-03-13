import sys
print(sys.argv)

source_address = input('请输入被拷贝文件源地址').strip()   #strip()不带参数是移除两边的空白符，包括/n,/t,' '
target_address = input('请输入拷贝文件目标地址').strip()
sys.argv.append(source_address)
sys.argv.append(target_address)
print(sys.argv)

if len(sys.argv) != 3:
    print('usage: cp source_file target_file')

source_file,target_file=sys.argv[1],sys.argv[2]

with open(source_file,'rb') as read_f,open(target_file,'wb') as write_f:
    for line in read_f:
        write_f.write(line)
sys.exit()
