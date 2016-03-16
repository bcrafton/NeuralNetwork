# NeuralNetwork
C implementation for Neural network from Stanford's machine learning course on Coursera. 
### Compilation
This is written for gcc.

It can be compiled with the following shell command: </br>
**gcc main.c matrix.c matrix_util.c NeuralNetwork.c vector.c -lm**
### Running the Code
The user also needs to specify the number of iterations that they would like to do, this is done by adding an integer after running the binary.</br>
Example: To run gradient descent for 100 iterations add '100' after running the binary </br>
**./a.out 100**
### Running the code on the discovery cluster
In this repo there is a script called NN.bash. In this script I include both the compile shell command and the command to execute the code for 100 iterations. The user only needs to **modify the script for their name and directory** on the discovery cluster and execute:</br>
**bsub< NN.bash**
###Data
The training data and some already created matrices are available in the form of csv files included in this repo. As is, the program is using the premade matrices to eliminate randomness for debugging and for grading.

**X.csv** contains the training data. Each row is a 20x20 greyscale image. So there are 400 columns (400 indexes in each row) and each one contains a pixel. There are 5000 rows, meaning 5000 training images.

**y.csv** contains what the images actually are, because this is a supervised learning problem it is necessary to have the actual classification of the data that is being trained. y.csv is a 5000x1 matrix. So there are 5000 rows, each of which contains the actual solution for the respective X row. So the 20x20 image in each row of X maps to the actual classification of that image in y.
Example: Row 20 in X contains 400 pixels that represent a handrawn '2'. Row 20 of y will contain the number 2.

**theta1.csv and theta2.csv** contain premade random classifier data. This is for debugging and grading. The user can choose to use the premade matrix or create their own random matrix in runtime with the 'matrix_random' function.

###Results
|	Iteration	|	Accuracy	|	Time Taken(s)	|
|	------------- 	|	:-------------:	|	-------------:	|
|	100	|	0.761	|	498.8	|
|	200	|	0.8562	|	995.61	|
|	300	|	0.8894	|	1491.83	|
|	400	|	0.9066	|	1988.06	|
|	500	|	0.9154	|	2485.18	|
|	600	|	0.9214	|	2981.99	|
|	700	|	0.925	|	3478.66	|
|	800	|	0.929	|	3975	|
|	900	|	0.9322	|	4471.46	|
|	1000	|	0.9358	|	4968.35	|
|	1100	|	0.9384	|	5465.45	|
|	1200	|	0.942	|	5962.7	|
|	1300	|	0.9442	|	6459.46	|
|	1400	|	0.947	|	6956.2	|
|	1500	|	0.948	|	7452.91	|
|	1600	|	0.9504	|	7949.62	|
|	1700	|	0.9526	|	8446.99	|
|	1800	|	0.9534	|	8944.57	|
|	1900	|	0.9554	|	9440.48	|
|	2000	|	0.9568	|	9937.17	|
|	2100	|	0.9576	|	10434.3	|
|	2200	|	0.9586	|	10931.54	|
|	2300	|	0.9594	|	11427.94	|
|	2400	|	0.96	|	11924.67	|
|	2500	|	0.9612	|	12421.14	|
|	2600	|	0.9622	|	12917.93	|
|	2700	|	0.9626	|	13413.99	|
|	2800	|	0.9638	|	13911.69	|
|	2900	|	0.9648	|	14408.78	|
|	3000	|	0.966	|	14904.72	|
|	3100	|	0.967	|	15401.51	|
|	3200	|	0.9678	|	15898.07	|
|	3300	|	0.9684	|	16393.96	|
|	3400	|	0.969	|	16889.57	|
|	3500	|	0.9696	|	17385.93	|
|	3600	|	0.9702	|	17883.02	|
|	3700	|	0.9716	|	18379.62	|
|	3800	|	0.9722	|	18876.54	|
|	3900	|	0.9726	|	19373.89	|
|	4000	|	0.973	|	19870.97	|
|	4100	|	0.974	|	20368	|
|	4200	|	0.9748	|	20864.48	|
|	4300	|	0.975	|	21362.37	|
|	4400	|	0.9762	|	21859.43	|
|	4500	|	0.9764	|	22356.36	|
|	4600	|	0.977	|	22853.93	|
|	4700	|	0.9774	|	23351.29	|
|	4800	|	0.9776	|	23847.43	|
|	4900	|	0.978	|	24343.09	|
|	5000	|	0.9782	|	24840.08	|
|	5100	|	0.9782	|	25337.13	|
|	5200	|	0.979	|	25832.89	|
|	5300	|	0.9796	|	26330.44	|
|	5400	|	0.9798	|	26827.28	|
|	5500	|	0.9802	|	27323.99	|
|	5600	|	0.9802	|	27820.78	|
|	5700	|	0.981	|	28317.28	|
|	5800	|	0.981	|	28815.24	|
|	5900	|	0.9812	|	29311.15	|
|	6000	|	0.9816	|	29807.15	|
|	6100	|	0.982	|	30303.42	|
|	6200	|	0.9824	|	30800.46	|
|	6300	|	0.9826	|	31297.45	|
|	6400	|	0.9826	|	31793.22	|
|	6500	|	0.9834	|	32290.28	|
|	6600	|	0.9836	|	32788.19	|
|	6700	|	0.984	|	33284.56	|
|	6800	|	0.984	|	33781.76	|
|	6900	|	0.9848	|	34279.07	|
|	7000	|	0.9848	|	34775.83	|
|	7100	|	0.9848	|	35272.45	|
|	7200	|	0.985	|	35768.83	|
|	7300	|	0.9852	|	36265.51	|
|	7400	|	0.9852	|	36762.34	|
|	7500	|	0.9852	|	37258.72	|
|	7600	|	0.9852	|	37755.94	|
|	7700	|	0.9856	|	38252.72	|
|	7800	|	0.9856	|	38749.03	|
|	7900	|	0.986	|	39246.94	|
|	8000	|	0.986	|	39743.55	|
|	8100	|	0.9864	|	40240.17	|
|	8200	|	0.9866	|	40737.25	|
|	8300	|	0.9872	|	41234.53	|
|	8400	|	0.9874	|	41731.45	|
|	8500	|	0.9874	|	42227.86	|
|	8600	|	0.9876	|	42725.11	|
|	8700	|	0.988	|	43222.07	|
|	8800	|	0.988	|	43719.61	|
|	8900	|	0.988	|	44216.54	|
|	9000	|	0.988	|	44713.61	|
|	9100	|	0.9884	|	45210.49	|
|	9200	|	0.9884	|	45706.89	|
|	9300	|	0.9884	|	46204	|
|	9400	|	0.9884	|	46700.83	|
|	9500	|	0.989	|	47197.93	|
|	9600	|	0.989	|	47695.19	|
|	9700	|	0.9892	|	48191.54	|
|	9800	|	0.9892	|	48688.88	|
|	9900	|	0.9892	|	49185.11	|
|	10000	|	0.9892	|	49681.31	|
