import csv
import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_file', default='../myTrain.csv', help='Data file')

FLAGS = parser.parse_args()

# Read the input CSV file
input_file = FLAGS.data_file
output_file_1 = 'myTrain.csv'
output_file_2 = 'myValid.csv'

with open(input_file, 'r') as csv_file:  # r means we opened this file in read mode.
    csv_reader = csv.reader(csv_file)  
    header = next(csv_reader)  # Assuming the first row is the header
    
    rows = list(csv_reader)  # Convert the reader object to a list for random shuffling
    random.shuffle(rows)  # Shuffle the rows randomly

    # Calculate the number of rows for each output file
    total_rows = len(rows)
    rows_output_1 = total_rows * 3 // 4  # 3/4 of the total rows
    rows_output_2 = total_rows - rows_output_1  # Remaining 1/4 of the total rows

    # Write rows to output file 1
    with open(output_file_1, 'w', newline='') as csv_output_file_1:
        csv_writer_1 = csv.writer(csv_output_file_1)
        csv_writer_1.writerow(header)  # Write the header row to output file 1
        for i in range(rows_output_1):
            csv_writer_1.writerow(rows[i])  # Write the first 3/4 of the shuffled rows to output file 1

    # Write rows to output file 2
    with open(output_file_2, 'w', newline='') as csv_output_file_2:
        csv_writer_2 = csv.writer(csv_output_file_2)
        csv_writer_2.writerow(header)  # Write the header row to output file 2
        for i in range(rows_output_1, total_rows):
            csv_writer_2.writerow(rows[i])  # Write the remaining 1/4 of the shuffled rows to output file 2

print(f'Successfully divided {input_file} into {output_file_1} and {output_file_2}.')
