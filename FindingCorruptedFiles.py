import os
import cv2
import pandas as pd
bad_list = []
file_list = pd.read_csv(r"C:\Users\akshg\Aksh\Aksh\Senior Year\Research\COVID-Arc\Updated Dataset\0-train.csv")
print(list(file_list['Filename'].head()))
index_2 = 1
for f in file_list['Filename']:
    print(f)
    print(index_2)
    index = f.rfind('.')
    ext = f[index+1:]
    index_2 += 1
    if ext not in ['jpg', 'png', 'bmp', 'gif']:
        print(f'file {f}  has an invalid extension {ext}')
        bad_list.append(f)
    else:
        try:
            img = cv2.imread(f)
            size = img.shape
        except:
            print(f'file {f} is not a valid image file')
            bad_list.append(f)

file_list = pd.read_csv(r"C:\Users\akshg\Aksh\Aksh\Senior Year\Research\COVID-Arc\Updated Dataset\0-test.csv")
for f in file_list['Filename']:
    print(f)
    print(index_2)
    index = f.rfind('.')
    ext = f[index+1:]
    index_2 += 1
    if ext not in ['jpg', 'png', 'bmp', 'gif']:
        print(f'file {f}  has an invalid extension {ext}')
        bad_list.append(f)
    else:
        try:
            img = cv2.imread(f)
            size = img.shape
        except:
            print(f'file {f} is not a valid image file')
            bad_list.append(f)

print(bad_list)

# Bad Files:
# ['C:\\Users\\akshg\\Aksh\\Aksh\\Senior Year\\Research\\COVID-Arc\\Updated Dataset\\CleanedData\\Healthy\\99\\18.png',
# 'C:\\Users\\akshg\\Aksh\\Aksh\\Senior Year\\Research\\COVID-Arc\\Updated Dataset\\CleanedData\\Covid\\4\\39.png']