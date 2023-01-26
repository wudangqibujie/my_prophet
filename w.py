import chardet

f = open('b.txt', 'r', encoding='utf-8')
data = f.read()
print(data)

# f = open('b.txt', 'w', encoding="utf-8")
# f.write('宝马')