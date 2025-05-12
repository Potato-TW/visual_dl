import pathlib

a = pathlib.Path('hw4_realse_dataset/train/degraded')

# for i in a.glob('*.png'):
#     print(i)

# d = sorted(a.glob('*.png'))
# print(d[0].stem)


b = pathlib.Path('hw4_realse_dataset/train/degraded/snow-541.png')
t, n = b.stem.split('-')
print(t, n)