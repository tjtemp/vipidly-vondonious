import matplotlib.pyplot as plt

plt.figure(figsize=(10,15))
plt.subplot(211)
img = plt.imread('ted.jpg')
plt.imshow(img)
plt.subplot(212)
img = plt.imread('sky.jpg')
plt.imshow(img)
plt.show()
