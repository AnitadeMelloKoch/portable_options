# import numpy as np
# import matplotlib.pyplot as plt
file_list = """
chihuahua_n02085620_10074.npy  chihuahua_n02085620_3838.npy  spaniel_n02085782_143.npy   spaniel_n02085782_3158.npy
chihuahua_n02085620_10131.npy  chihuahua_n02085620_3875.npy  spaniel_n02085782_1460.npy  spaniel_n02085782_3215.npy
chihuahua_n02085620_10621.npy  chihuahua_n02085620_3877.npy  spaniel_n02085782_1503.npy  spaniel_n02085782_3325.npy
chihuahua_n02085620_1073.npy   chihuahua_n02085620_3880.npy  spaniel_n02085782_1521.npy  spaniel_n02085782_3331.npy
chihuahua_n02085620_10976.npy  chihuahua_n02085620_3928.npy  spaniel_n02085782_1528.npy  spaniel_n02085782_3354.npy
chihuahua_n02085620_11140.npy  chihuahua_n02085620_3942.npy  spaniel_n02085782_1552.npy  spaniel_n02085782_3387.npy
chihuahua_n02085620_11238.npy  chihuahua_n02085620_3975.npy  spaniel_n02085782_1600.npy  spaniel_n02085782_3400.npy
chihuahua_n02085620_11258.npy  chihuahua_n02085620_4016.npy  spaniel_n02085782_1610.npy  spaniel_n02085782_3404.npy
chihuahua_n02085620_11337.npy  chihuahua_n02085620_4159.npy  spaniel_n02085782_1626.npy  spaniel_n02085782_3420.npy
chihuahua_n02085620_11477.npy  chihuahua_n02085620_4207.npy  spaniel_n02085782_1656.npy  spaniel_n02085782_3481.npy
chihuahua_n02085620_1152.npy   chihuahua_n02085620_4266.npy  spaniel_n02085782_1665.npy  spaniel_n02085782_3499.npy
chihuahua_n02085620_11696.npy  chihuahua_n02085620_4290.npy  spaniel_n02085782_1691.npy  spaniel_n02085782_3516.npy
chihuahua_n02085620_11818.npy  chihuahua_n02085620_431.npy   spaniel_n02085782_1724.npy  spaniel_n02085782_3578.npy
chihuahua_n02085620_11948.npy  chihuahua_n02085620_4441.npy  spaniel_n02085782_172.npy   spaniel_n02085782_3649.npy
chihuahua_n02085620_1205.npy   chihuahua_n02085620_4515.npy  spaniel_n02085782_1731.npy  spaniel_n02085782_3720.npy
chihuahua_n02085620_12101.npy  chihuahua_n02085620_4572.npy  spaniel_n02085782_1750.npy  spaniel_n02085782_3727.npy
chihuahua_n02085620_12334.npy  chihuahua_n02085620_4602.npy  spaniel_n02085782_1764.npy  spaniel_n02085782_3744.npy
chihuahua_n02085620_1235.npy   chihuahua_n02085620_4673.npy  spaniel_n02085782_1774.npy  spaniel_n02085782_3781.npy
chihuahua_n02085620_12718.npy  chihuahua_n02085620_4700.npy  spaniel_n02085782_1778.npy  spaniel_n02085782_3810.npy
chihuahua_n02085620_1271.npy   chihuahua_n02085620_473.npy   spaniel_n02085782_1782.npy  spaniel_n02085782_381.npy
chihuahua_n02085620_1298.npy   chihuahua_n02085620_477.npy   spaniel_n02085782_17.npy    spaniel_n02085782_382.npy
chihuahua_n02085620_13151.npy  chihuahua_n02085620_4814.npy  spaniel_n02085782_1836.npy  spaniel_n02085782_3855.npy
chihuahua_n02085620_1321.npy   chihuahua_n02085620_4875.npy  spaniel_n02085782_1848.npy  spaniel_n02085782_385.npy
chihuahua_n02085620_13383.npy  chihuahua_n02085620_4919.npy  spaniel_n02085782_1855.npy  spaniel_n02085782_3870.npy
chihuahua_n02085620_1346.npy   chihuahua_n02085620_4951.npy  spaniel_n02085782_186.npy   spaniel_n02085782_3889.npy
chihuahua_n02085620_13964.npy  chihuahua_n02085620_4980.npy  spaniel_n02085782_1890.npy  spaniel_n02085782_3896.npy
chihuahua_n02085620_14252.npy  chihuahua_n02085620_4998.npy  spaniel_n02085782_191.npy   spaniel_n02085782_3899.npy
chihuahua_n02085620_14413.npy  chihuahua_n02085620_500.npy   spaniel_n02085782_1929.npy  spaniel_n02085782_38.npy
chihuahua_n02085620_14516.npy  chihuahua_n02085620_5093.npy  spaniel_n02085782_1949.npy  spaniel_n02085782_3966.npy
chihuahua_n02085620_1455.npy   chihuahua_n02085620_5312.npy  spaniel_n02085782_1964.npy  spaniel_n02085782_3979.npy
chihuahua_n02085620_1492.npy   chihuahua_n02085620_5496.npy  spaniel_n02085782_2010.npy  spaniel_n02085782_3990.npy
chihuahua_n02085620_1502.npy   chihuahua_n02085620_5661.npy  spaniel_n02085782_2014.npy  spaniel_n02085782_4042.npy
chihuahua_n02085620_1558.npy   chihuahua_n02085620_5713.npy  spaniel_n02085782_2045.npy  spaniel_n02085782_4208.npy
chihuahua_n02085620_1569.npy   chihuahua_n02085620_574.npy   spaniel_n02085782_2074.npy  spaniel_n02085782_4269.npy
chihuahua_n02085620_1617.npy   chihuahua_n02085620_575.npy   spaniel_n02085782_2100.npy  spaniel_n02085782_4294.npy
chihuahua_n02085620_1620.npy   chihuahua_n02085620_5771.npy  spaniel_n02085782_2118.npy  spaniel_n02085782_4332.npy
chihuahua_n02085620_1765.npy   chihuahua_n02085620_588.npy   spaniel_n02085782_2128.npy  spaniel_n02085782_4351.npy
chihuahua_n02085620_1816.npy   chihuahua_n02085620_5927.npy  spaniel_n02085782_2162.npy  spaniel_n02085782_435.npy
chihuahua_n02085620_1862.npy   chihuahua_n02085620_6295.npy  spaniel_n02085782_2182.npy  spaniel_n02085782_4365.npy
chihuahua_n02085620_1916.npy   chihuahua_n02085620_6399.npy  spaniel_n02085782_2207.npy  spaniel_n02085782_440.npy
chihuahua_n02085620_199.npy    chihuahua_n02085620_6931.npy  spaniel_n02085782_2241.npy  spaniel_n02085782_4436.npy
chihuahua_n02085620_2053.npy   chihuahua_n02085620_712.npy   spaniel_n02085782_2250.npy  spaniel_n02085782_4438.npy
chihuahua_n02085620_2188.npy   chihuahua_n02085620_7292.npy  spaniel_n02085782_2255.npy  spaniel_n02085782_4458.npy
chihuahua_n02085620_2204.npy   chihuahua_n02085620_730.npy   spaniel_n02085782_2269.npy  spaniel_n02085782_4511.npy
chihuahua_n02085620_2208.npy   chihuahua_n02085620_735.npy   spaniel_n02085782_2279.npy  spaniel_n02085782_4564.npy
chihuahua_n02085620_242.npy    chihuahua_n02085620_7436.npy  spaniel_n02085782_2293.npy  spaniel_n02085782_4574.npy
chihuahua_n02085620_2479.npy   chihuahua_n02085620_7440.npy  spaniel_n02085782_230.npy   spaniel_n02085782_4579.npy
chihuahua_n02085620_2507.npy   chihuahua_n02085620_7613.npy  spaniel_n02085782_2323.npy  spaniel_n02085782_4590.npy
chihuahua_n02085620_2517.npy   chihuahua_n02085620_7700.npy  spaniel_n02085782_2345.npy  spaniel_n02085782_4616.npy
chihuahua_n02085620_2590.npy   chihuahua_n02085620_7738.npy  spaniel_n02085782_2354.npy  spaniel_n02085782_4698.npy
chihuahua_n02085620_2614.npy   chihuahua_n02085620_7.npy     spaniel_n02085782_23.npy    spaniel_n02085782_4772.npy
chihuahua_n02085620_2650.npy   chihuahua_n02085620_806.npy   spaniel_n02085782_2428.npy  spaniel_n02085782_4780.npy
chihuahua_n02085620_2693.npy   chihuahua_n02085620_8420.npy  spaniel_n02085782_2459.npy  spaniel_n02085782_4798.npy
chihuahua_n02085620_2706.npy   chihuahua_n02085620_8491.npy  spaniel_n02085782_2491.npy  spaniel_n02085782_50.npy
chihuahua_n02085620_275.npy    chihuahua_n02085620_8558.npy  spaniel_n02085782_2549.npy  spaniel_n02085782_516.npy
chihuahua_n02085620_2793.npy   chihuahua_n02085620_8578.npy  spaniel_n02085782_2584.npy  spaniel_n02085782_518.npy
chihuahua_n02085620_2815.npy   chihuahua_n02085620_8585.npy  spaniel_n02085782_2635.npy  spaniel_n02085782_564.npy
chihuahua_n02085620_2887.npy   chihuahua_n02085620_8611.npy  spaniel_n02085782_2640.npy  spaniel_n02085782_572.npy
chihuahua_n02085620_2903.npy   chihuahua_n02085620_8636.npy  spaniel_n02085782_2655.npy  spaniel_n02085782_593.npy
chihuahua_n02085620_2921.npy   chihuahua_n02085620_8637.npy  spaniel_n02085782_267.npy   spaniel_n02085782_622.npy
chihuahua_n02085620_2937.npy   chihuahua_n02085620_9351.npy  spaniel_n02085782_2690.npy  spaniel_n02085782_626.npy
chihuahua_n02085620_2973.npy   chihuahua_n02085620_9357.npy  spaniel_n02085782_2695.npy  spaniel_n02085782_664.npy
chihuahua_n02085620_2981.npy   chihuahua_n02085620_9399.npy  spaniel_n02085782_2715.npy  spaniel_n02085782_665.npy
chihuahua_n02085620_3006.npy   chihuahua_n02085620_9414.npy  spaniel_n02085782_2762.npy  spaniel_n02085782_668.npy
chihuahua_n02085620_3033.npy   chihuahua_n02085620_949.npy   spaniel_n02085782_2796.npy  spaniel_n02085782_676.npy
chihuahua_n02085620_3045.npy   chihuahua_n02085620_952.npy   spaniel_n02085782_2874.npy  spaniel_n02085782_697.npy
chihuahua_n02085620_3093.npy   chihuahua_n02085620_9654.npy  spaniel_n02085782_2886.npy  spaniel_n02085782_698.npy
chihuahua_n02085620_3110.npy   spaniel_n02085782_1039.npy    spaniel_n02085782_28.npy    spaniel_n02085782_725.npy
chihuahua_n02085620_3208.npy   spaniel_n02085782_1058.npy    spaniel_n02085782_2914.npy  spaniel_n02085782_749.npy
chihuahua_n02085620_326.npy    spaniel_n02085782_1059.npy    spaniel_n02085782_2922.npy  spaniel_n02085782_754.npy
chihuahua_n02085620_3402.npy   spaniel_n02085782_1077.npy    spaniel_n02085782_2939.npy  spaniel_n02085782_757.npy
chihuahua_n02085620_3407.npy   spaniel_n02085782_1085.npy    spaniel_n02085782_2978.npy  spaniel_n02085782_806.npy
chihuahua_n02085620_3409.npy   spaniel_n02085782_1143.npy    spaniel_n02085782_2.npy     spaniel_n02085782_80.npy
chihuahua_n02085620_3423.npy   spaniel_n02085782_1156.npy    spaniel_n02085782_3019.npy  spaniel_n02085782_810.npy
chihuahua_n02085620_3485.npy   spaniel_n02085782_1191.npy    spaniel_n02085782_3021.npy  spaniel_n02085782_82.npy
chihuahua_n02085620_3488.npy   spaniel_n02085782_1224.npy    spaniel_n02085782_3030.npy  spaniel_n02085782_845.npy
chihuahua_n02085620_3593.npy   spaniel_n02085782_1267.npy    spaniel_n02085782_3031.npy  spaniel_n02085782_866.npy
chihuahua_n02085620_3651.npy   spaniel_n02085782_126.npy     spaniel_n02085782_3052.npy  spaniel_n02085782_874.npy
chihuahua_n02085620_3677.npy   spaniel_n02085782_1284.npy    spaniel_n02085782_3065.npy  spaniel_n02085782_919.npy
chihuahua_n02085620_3681.npy   spaniel_n02085782_1348.npy    spaniel_n02085782_3071.npy  spaniel_n02085782_935.npy
chihuahua_n02085620_368.npy    spaniel_n02085782_1350.npy    spaniel_n02085782_3098.npy  spaniel_n02085782_940.npy
chihuahua_n02085620_3742.npy   spaniel_n02085782_1353.npy    spaniel_n02085782_309.npy   spaniel_n02085782_962.npy
chihuahua_n02085620_3763.npy   spaniel_n02085782_1401.npy    spaniel_n02085782_3121.npy
chihuahua_n02085620_3826.npy   spaniel_n02085782_1425.npy    spaniel_n02085782_313.npy
chihuahua_n02085620_382.npy    spaniel_n02085782_1434.npy    spaniel_n02085782_3148.npy """

# Split the text by spaces and count elements
file_count = len(file_list.split())
print(f"Total number of .npy files: {file_count}")