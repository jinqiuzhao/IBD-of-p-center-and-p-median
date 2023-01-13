# IBD-of-p-center-and-p-median


The code of Improved Benders decomposition algorithm for city emergency service facility location.

1. The program runs on python 3.9 and Gurobi 9.5.
2. P_center_Benders.py and P_median_Benders.py are the main programs of the IBD algorithm for the two types of problems respectively. After the data type is selected by using the data_type parameter, the file can be directly run. Except the data.py file, other files run independently and do not affect each other.
3. CARS_model.py is a production and solving program for CARS models, which can optionally solve p-center or p-meidan models (comment out corresponding statements).
4. Data. py is a production program for reading random input data without the need for separate execution.
5. The data folder contains a part of the data of the example used in this paper, and the relevant data is stored in.npy format. Due to github file size limitations, the full experimental data is available at https://figshare.com/articles/dataset/test_data_IBD_zip/21894240
6. The data_oms.py in the Openstreetmap folder is used to download the urban road network data and generate the.npy format recognized by the IBD algorithm.
