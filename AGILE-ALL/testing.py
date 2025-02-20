from tests.AY25.testgraph import process_data
from agile.graphing import createGraph, findAllFrequencyOfColocation, frequencyOfColocation, dwellTimeAdjacencyMatrix, mergeResults, connectNodes
import time

# Path to the CSV file
csv_file = "data/dwelltime_testset.csv"

# Read the CSV file using pandas
data, df = process_data(csv_file)
print(df)
# Create the graph
graph = createGraph(data)

connectNodes(graph, 1, df, 5, 5, 100)

for edge in graph.edges:
    print(edge.__repr__())

"""
colocations = findAllFrequencyOfColocation(df, 5, 5, 100)
print(colocations)
"""
"""
start_time = time.time()
colocations = frequencyOfColocation(df, "adid_1", "adid_2", 5, 5, 100)
end_time = time.time()
print(colocations)
elapsed_time = end_time - start_time
print(f"Execution time for frequencyOfColocation: {elapsed_time:.2f} seconds")
"""

"""
#printNodeData()
print_adjacency_matrix()
print("\n")
#testFindRelated()
start_time = time.time()
connectRelatedNodes(graph, 100, df, 1.0)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time for connectRelatedNodes: {elapsed_time:.2f} seconds")
print_adjacency_matrix()   
"""

#testing dwell time stuff
# this is in hours btw
#print(dwellTimeWithinProximity(graph.get_nodes()[0], graph.get_nodes()[1], 100))

#print(dwellTimeWithinProximity(df, "adid_1", "adid_2"))
#print(dwellTimeAdjacencyMatrix(df, 5, 5, 100))
