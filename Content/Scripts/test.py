action_map = {}
action_arr = [3, 3, 2]
index = 0

for depth1 in range(action_arr[0]):
    for depth2 in range(action_arr[1]):
        for depth3 in range(action_arr[2]):
            action_map[index] = []
            action_map[index].append(depth1)
            action_map[index].append(depth2)
            action_map[index].append(depth3)
            index += 1

print(action_map)