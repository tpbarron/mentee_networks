1: mnist conv mentor run 30 epochs
2: mnist-10 conv mentee run 100 epochs
3: mnist-1 conv mentee run 150 epochs
4: mnist-50 conv mentee run 50 epochs

5: mnist mentor run 30 with logging as every epoch

6: obedient mnist-1
7: obedient mnist-10
8: obedient mnist-50
9: obedient mnist-100
10: obedient mnist-250
11: obedient mnist-500

12: adamant mnist-1
13: adamant mnist-10
14: adamant mnist-50
15: adamant mnist-100
16: adamant mnist-250
17: adamant mnist-500

18: independent mnist-1
19: independent mnist-10
20: independent mnist-50
21: independent mnist-100
22: independent mnist-250
23: independent mnist-500

# rerunning with proper snapshotting

24: obedient mnist-1
25: obedient mnist-10
26: obedient mnist-50
27: obedient mnist-100
28: obedient mnist-250
29: obedient mnist-500

30: adamant mnist-1
31: adamant mnist-10
32: adamant mnist-50
33: adamant mnist-100
34: adamant mnist-250
35: adamant mnist-500

36: independent mnist-1
37: independent mnist-10
38: independent mnist-50
39: independent mnist-100
40: independent mnist-250
41: independent mnist-500

# rerunnign trying to get plots on same graph

42: new mentor benchmark with only 1 hidden layer 5x5 convs
43: new mentor benchmark with only 1 hidden layer 3x3 convs

44 - 89: Some random runs trying to debug