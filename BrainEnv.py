import cuda.eann as eann

eann.init(2)

for _ in range(10):
    print(eann.think([1, 0, 1, 0]))
    eann.reward(1.2)