import numpy as np
import logging
import time
import json
import pandas as pd

logging.basicConfig(filename='app.log', filemode='w', format = '%(asctime)s  %(levelname)-10s %(processName)s  %(name)s %(message)s')

logging.debug("debug") 
logging.info("info") 
logging.warning("warning") 
logging.error("error")
logging.critical("critical")


class ComputePartition():
	def __init__(self, i1=[-10.0, 10.0], i2=[-10.0, 10.0], temp_step=5000):
		self.generate_range(i1, i2)
		self.Tmax = temp_step
		self.partition = dict()

	def generate_range(self, i1, i2):
		self.i1=np.arange(min(i1), max(i1), 0.01),
		self.i2=np.arange(min(i2), max(i2), 0.01)
		return i1, i2

	def cost_function(self, x):
		x1 = x[0]
		x2 = x[1]
    
        # function 1, levy function 
		obj = np.sin(3 * np.pi * x[0]) ** 2 + (x[0] - 1) ** 2 * (1
                + np.sin(3 * np.pi * x[1]) ** 2) + (x[1] - 1) ** 2 * (1
                + np.sin(2 * np.pi * x[1]) ** 2)

		return obj

	def calculate_partition(self):
		energy = list()
		exploration_space = [(i,j) for i in self.i1 for j in self.i2]
		super_energies = [self.cost_function(i) for i in exploration_space]
		for i in range(1, 5000+1):
			energies = super_energies.copy()
			Beta = 1/i
			energies = -Beta * np.array(energies)
			partition_function = np.sum(np.exp(energies))
			self.partition[i] = partition_function
			energy.append(energies)

		with open('partition.json', 'w') as fp:
			json.dump(self.partition, fp)

		pd.DataFrame.from_records(energy).T.to_csv("energies.csv")

if __name__ == '__main__':
	partition = ComputePartition()
	try:
		partition.calculate_partition()
	except Exception as e:

		logging.error("Exception occurred", exc_info=True)
	