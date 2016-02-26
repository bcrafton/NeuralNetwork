# NeuralNetwork
C implementation for Neural network from Stanford's machine learning course on Coursera. 
### Compilation
This is written for gcc c compiler.

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
Iteration	Accuracy	Time Taken(s)	</br>
100	0.761	498.8	</br>
200	0.8562	995.61	</br>
300	0.8894	1491.83	</br>
400	0.9066	1988.06	</br>
500	0.9154	2485.18	</br>
600	0.9214	2981.99	</br>
700	0.925	3478.66	</br>
800	0.929	3975	</br>
900	0.9322	4471.46	</br>
1000	0.9358	4968.35	</br>
1100	0.9384	5465.45	</br>
1200	0.942	5962.7	</br>
1300	0.9442	6459.46	</br>
1400	0.947	6956.2	</br>
1500	0.948	7452.91	</br>
1600	0.9504	7949.62	</br>
1700	0.9526	8446.99	</br>
1800	0.9534	8944.57	</br>
1900	0.9554	9440.48	</br>
2000	0.9568	9937.17	</br>
2100	0.9576	10434.3	</br>
2200	0.9586	10931.54	</br>
2300	0.9594	11427.94	</br>
2400	0.96	11924.67	</br>
2500	0.9612	12421.14	</br>
2600	0.9622	12917.93	</br>
2700	0.9626	13413.99	</br>
2800	0.9638	13911.69	</br>
2900	0.9648	14408.78	</br>
3000	0.966	14904.72	</br>
3100	0.967	15401.51	</br>
3200	0.9678	15898.07	</br>
3300	0.9684	16393.96	</br>
3400	0.969	16889.57	</br>
3500	0.9696	17385.93	</br>
3600	0.9702	17883.02	</br>
3700	0.9716	18379.62	</br>
3800	0.9722	18876.54	</br>
3900	0.9726	19373.89	</br>
4000	0.973	19870.97	</br>
4100	0.974	20368	</br>
4200	0.9748	20864.48	</br>
4300	0.975	21362.37	</br>
4400	0.9762	21859.43	</br>
4500	0.9764	22356.36	</br>
4600	0.977	22853.93	</br>
4700	0.9774	23351.29	</br>
4800	0.9776	23847.43	</br>
4900	0.978	24343.09	</br>
5000	0.9782	24840.08	</br>
5100	0.9782	25337.13	</br>
5200	0.979	25832.89	</br>
5300	0.9796	26330.44	</br>
5400	0.9798	26827.28	</br>
5500	0.9802	27323.99	</br>
5600	0.9802	27820.78	</br>
5700	0.981	28317.28	</br>
5800	0.981	28815.24	</br>
5900	0.9812	29311.15	</br>
6000	0.9816	29807.15	</br>
6100	0.982	30303.42	</br>
6200	0.9824	30800.46	</br>
6300	0.9826	31297.45	</br>
6400	0.9826	31793.22	</br>
6500	0.9834	32290.28	</br>
6600	0.9836	32788.19	</br>
6700	0.984	33284.56	</br>
6800	0.984	33781.76	</br>
6900	0.9848	34279.07	</br>
7000	0.9848	34775.83	</br>
7100	0.9848	35272.45	</br>
7200	0.985	35768.83	</br>
7300	0.9852	36265.51	</br>
7400	0.9852	36762.34	</br>
7500	0.9852	37258.72	</br>
7600	0.9852	37755.94	</br>
7700	0.9856	38252.72	</br>
7800	0.9856	38749.03	</br>
7900	0.986	39246.94	</br>
8000	0.986	39743.55	</br>
8100	0.9864	40240.17	</br>
8200	0.9866	40737.25	</br>
8300	0.9872	41234.53	</br>
8400	0.9874	41731.45	</br>
8500	0.9874	42227.86	</br>
8600	0.9876	42725.11	</br>
8700	0.988	43222.07	</br>
8800	0.988	43719.61	</br>
8900	0.988	44216.54	</br>
9000	0.988	44713.61	</br>
9100	0.9884	45210.49	</br>
9200	0.9884	45706.89	</br>
9300	0.9884	46204	</br>
9400	0.9884	46700.83	</br>
9500	0.989	47197.93	</br>
9600	0.989	47695.19	</br>
9700	0.9892	48191.54	</br>
9800	0.9892	48688.88	</br>
9900	0.9892	49185.11	</br>
10000	0.9892	49681.31	</br>
