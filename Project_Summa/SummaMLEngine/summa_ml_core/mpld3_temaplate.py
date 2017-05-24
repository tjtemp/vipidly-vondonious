import matplotlib.pyplot as plt
import mpld3


with open('test.html','w') as f:
	fig, ax = plt.subplots()
	x=[1,2,3,4,5,6,7,8]
	y=[2,3,4,5,2,3,1,4]
	ax.scatter(x,y)
	ax.scatter([-1,-2,-3],[1,2,3])
	f.write(mpld3.fig_to_html(fig))
