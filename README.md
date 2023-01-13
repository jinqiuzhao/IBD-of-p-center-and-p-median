# IBD-of-p-center-and-p-median
The code of Improved Benders decomposition algorithm for city emergency service facility location.

1. The program runs on python 3.9 and Gurobi 9.5.
2. P_center_Benders.py and P_median_Benders.py are the main programs of the IBD algorithm for the two types of problems respectively. After the data type is selected by using the data_type parameter, the file can be directly run. Except the data.py file, other files run independently and do not affect each other.
3. CARS_model.py is a production and solving program for CARS models, which can optionally solve p-center or p-meidan models (comment out corresponding statements).
4. Data. py is a production program for reading random input data without the need for separate execution.
5. The data folder contains all the data of the example used in this paper, and the relevant data is stored in.npy format.
