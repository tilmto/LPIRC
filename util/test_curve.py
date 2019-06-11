import matplotlib.pyplot as plt

fname = 'train_log.txt'

step = [90900,177751,268871,361588,562722,601713]
loss = [0.42,0.47,0.495,0.51,0.537,0.541]

plt.plot(step,loss,'^-')
plt.grid()
plt.xlabel('global step')
plt.ylabel('test accuracy')
plt.title('Test Accuracy of MobileNetV3')
plt.savefig('test_curve.jpg')
plt.show()