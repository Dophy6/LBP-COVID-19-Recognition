import json
for i in range(12):
    print("==========TEST {} ==============".format(i))
    with open("test_results_{}.json".format(i)) as j:
        data = json.load(j)
    for key in data.keys():
        print("{};{};{};{};{};{};{};{};{};{}".format(
            data[key]["method_2_avg_difference_limit"],
            data[key]["method_3_avg_divergences_limit"],
            data[key]["success_counter"]["first_method"]["COV"],
            data[key]["success_counter"]["first_method"]["NON_C"],
            data[key]["success_counter"]["second_method"]["COV"],
            data[key]["success_counter"]["second_method"]["NON_C"],
            data[key]["success_counter"]["third_method"]["COV"],
            data[key]["success_counter"]["third_method"]["NON_C"],
            data[key]["success_counter"]["all_refs_together"]["COV"],
            data[key]["success_counter"]["all_refs_together"]["NON_C"]
        ))
