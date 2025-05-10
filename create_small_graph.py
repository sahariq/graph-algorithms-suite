with open('sample_graph.txt', 'r') as fin, open('small_graph.txt', 'w') as fout:
    for i, line in enumerate(fin):
        if i >= 100:
            break
        fout.write(line) 