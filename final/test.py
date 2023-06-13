import os

# print(len(os.listdir('./model')))

count = 0
for path in os.listdir('./data/S5'):
    if path.isnumeric():
        count += 1
print(count)

