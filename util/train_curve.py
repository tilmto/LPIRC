import re
import matplotlib.pyplot as plt

fname = 'train_log.txt'

step = []
loss = []
with open(fname,'r') as f:
	for line in f.readlines():
		if line.find('INFO:tensorflow:global step')!=-1:
			data = re.findall(r'\d+.?\d*',line)
			step.append(int(data[0].strip(':')))
			loss.append(float(data[1]))

print(len(step))
print(len(loss))

plt.plot(step,loss)
plt.grid()
plt.xlabel('global step')
plt.ylabel('train loss')
plt.title('Training Curve of MobileNetV3')
plt.savefig('train_curve.jpg')
plt.show()