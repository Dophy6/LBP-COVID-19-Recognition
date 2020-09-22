import json
for i in range(12):
    print("==========TEST {} ==============".format(i))
    with open("test_results_{}.json".format(i)) as j:
        data = json.load(j)
    for key in data.keys():
        print("{};{};{};{}".format(
            data[key]["success_counter"]["first_method"],
            data[key]["success_counter"]["second_method"],
            data[key]["success_counter"]["third_method"],
            data[key]["success_counter"]["all_refs_together"]
        ))
