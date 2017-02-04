
# coding: utf-8

# In[1]:

"""
Wrapper methods around "import csv".
Created by Uki D. Lucas on Feb. 4, 2017
"""

# TODO implement batch_from, batch_to - I did not need it for 8037 rows
# TODO implement has_header_row
def read_csv(file_path):
    """
    Usage:
    headers, data = read_csv(file_path)
    Parameter: 
    - file_path: can be relative path "../../../DATA/stuff.csv"
    Returns:
    - headers: array of strings e.g. ['steering', 'throttle', 'brake', 'speed']
    - data: array of strings, you have to convert values to int, float yourself
   test_read_csv()
    """
    import csv
    # Opening spreadsheet to read in TEXT mode: 'rt'
    with open(file_path, 'rt') as csvfile:
        # Most common format of CSV, TODO improve
        payload = csv.reader(csvfile, delimiter=',', quotechar='"') 
        row_counter = 0
        headers = []
        data = []
        for row in payload:
            row_counter = row_counter + 1
            
            if row_counter == 1:
                headers = row
                # print (type(row))
                # print ( row)
                # print ('\t '.join(row))
            elif 1 < row_counter < 3:
                # print (type(row))
                # print ( row)
                # print ('\t '.join(row))
                data.append(row)
            else:
                data.append(row)
        print("row_counter", row_counter)
        return headers, data


# In[2]:

def test_read_csv():
    """
    This test is specific to Uki's enviroment.
    """
    data_dir = "../../../DATA/behavioral_cloning_data/"
    headers, data = read_csv(file_path = data_dir + 'driving_log.csv')
    print("headers \n",headers)
    print("3rd row of data \n",data[2:3])
#test_read_csv()


# In[ ]:



