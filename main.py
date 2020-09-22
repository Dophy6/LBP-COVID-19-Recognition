import random, json
import numpy as np
from threading import Thread
from skimage.feature import local_binary_pattern
from skimage import io
from skimage.color import rgb2gray
from pprint import pprint
#import multithreading
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

NON_COVID_PATH = "/path/to/COVID-CT/Images-processed/CT_NonCOVID/"
COVID_PATH = "/path/to/COVID-CT/Images-processed/CT_COVID/CT_COVID/"

TRAINING_COVID_IMGS = []
TRAINING_NON_COVID_IMGS = []
TRAINING_IMGS_NUMBER = 200

TEST_COVID_IMGS = []
TEST_NON_COVID_IMGS = []
TEST_IMGS_NUMBER = 100

LBP = {}
LBP["radius"] = 1
LBP["n_points"] = 8 * LBP["radius"]
LBP["method"] = "uniform"

REFS = {
    "first_method": None,
    "second_method": None,
    "third_method": None,
}

def load_random_training_set(imgs_number, paths=[]):
    training_files = {"COV": [], "NON_COV": []}
    for path in paths:
        imgs_counter = 0
        file_list = os.listdir(path)
        random.shuffle(file_list)
        for filename in file_list:
            try:
                image = io.imread(path + filename, plugin='matplotlib') #Load custom image
                image = rgb2gray(image) #Need to convert in grayscale if you use a custom image
                if path == NON_COVID_PATH:
                    training_files["NON_COV"].append(filename)
                    TRAINING_NON_COVID_IMGS.append(image)
                else:
                    training_files["COV"].append(filename)
                    TRAINING_COVID_IMGS.append(image)
                imgs_counter +=1
                if imgs_counter >= TRAINING_IMGS_NUMBER:
                    break
            except Exception as e:
                print("LOAD_TRAINING_SET - Image: {}, rejected due to: {}".format(filename, str(e)))
        
    return training_files

def load_random_test_set(imgs_number, paths=[], used_files=[]):
    test_files = {"COV": [], "NON_COV": []}
    for path in paths:
        imgs_counter = 0
        file_list = [filename for filename in os.listdir(path) if filename not in used_files]
        random.shuffle(file_list)
        for filename in file_list:
            try:
                image = io.imread(path + filename, plugin='matplotlib') #Load custom image
                image = rgb2gray(image) #Need to convert in grayscale if you use a custom image
                if path == NON_COVID_PATH:
                    test_files["NON_COV"].append(filename)
                    TEST_NON_COVID_IMGS.append(image)
                else:
                    test_files["COV"].append(filename)
                    TEST_COVID_IMGS.append(image)
                imgs_counter +=1
                if imgs_counter >= TEST_IMGS_NUMBER:
                    break
            except Exception as e:
                print("LOAD_TEST_SET - Image: {}, rejected due to: {}".format(filename, str(e)))
    
    return test_files

def method_1(lbps_collections = {"COV":[], "NON_COV": []}):
    refs = {"COV_M1":None, "NON_COV_M1": None}
    for key, lbps_collection in lbps_collections.items():
        key += "_M1"
        for lbp in lbps_collection:
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, density=True, bins=n_bins,range=(0, n_bins))
            if refs[key] is None:
                refs[key] = hist
            else:
                refs[key] += hist
        refs[key] /= len(lbps_collection)
    
    REFS["first_method"] = refs

def method_2(lbps_collections = {"COV":[], "NON_COV": []}, avg_difference_limit = 0.01):
    temp_refs = {"COV":{}, "NON_COV":{}}
    for key, lbps_collection in lbps_collections.items():
        goup_counter=0   
        for lbp in lbps_collection:
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, density=True, bins=n_bins,range=(0, n_bins))
            
            if len(temp_refs[key].keys()) == 0:
                temp_refs[key]["{}_{}_M2".format(key,goup_counter)] = {"h": hist, "c":1}
                goup_counter += 1
            else:
                avg_differences = []
                for key_ in temp_refs[key].keys():
                    temp = np.abs(hist - (temp_refs[key][key_]["h"]/temp_refs[key][key_]["c"]))
                    avg_diff = np.sum(temp)/len(temp)
                    avg_differences.append({"key":key_,"val":avg_diff})
                avg_differences = sorted(avg_differences, key=lambda k: k["val"]) 
                if avg_differences[0]["val"] <= avg_difference_limit:
                    temp_refs[key][avg_differences[0]["key"]]["h"]+= hist
                    temp_refs[key][avg_differences[0]["key"]]["c"]+= 1
                else:
                    temp_refs[key]["{}_{}_M2".format(key, goup_counter)] = {"h": hist, "c":1}
                    goup_counter += 1
                    
        for key_ in temp_refs[key].keys():
            temp_refs[key][key_] = temp_refs[key][key_]["h"]/temp_refs[key][key_]["c"]
    
    print("METHOD 2 - COVID Groups number found: {}".format(len(temp_refs["COV"].keys())))
    print("METHOD 2 - NON_COVID Groups number found: {}".format(len(temp_refs["NON_COV"].keys())))

    refs = temp_refs["COV"]
    refs.update(temp_refs["NON_COV"])
    REFS["second_method"] = refs

def method_3(lbps_collections = {"COV":[], "NON_COV": []}, avg_divergences_limit=0.05):
    temp_refs = {"COV":{}, "NON_COV":{}}
    for key, lbps_collection in lbps_collections.items():
        goup_counter=0   
        for lbp in lbps_collection:
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, density=True, bins=n_bins,range=(0, n_bins))
            
            if len(temp_refs[key].keys()) == 0:
                temp_refs[key]["{}_{}_M3".format(key,goup_counter)] = {"h": hist, "c":1}
                goup_counter += 1
            else:
                avg_divergences = []
                for key_ in temp_refs[key].keys():
                    temp = temp_refs[key][key_]["h"]/temp_refs[key][key_]["c"]
                    avg_div = kullback_leibler_divergence(temp, hist)
                    avg_divergences.append({"key":key_,"val":avg_div})
                avg_divergences = sorted(avg_divergences, key=lambda k: k["val"]) 
                if avg_divergences[0]["val"] <= avg_divergences_limit:
                    temp_refs[key][avg_divergences[0]["key"]]["h"]+= hist
                    temp_refs[key][avg_divergences[0]["key"]]["c"]+= 1
                else:
                    temp_refs[key]["{}_{}_M3".format(key, goup_counter)] = {"h": hist, "c":1}
                    goup_counter += 1
                    
        for key_ in temp_refs[key].keys():
            temp_refs[key][key_] = temp_refs[key][key_]["h"]/temp_refs[key][key_]["c"]
    
    print("METHOD 3 - COVID Groups number found: {}".format(len(temp_refs["COV"].keys())))
    print("METHOD 3 - NON_COVID Groups number found: {}".format(len(temp_refs["NON_COV"].keys())))

    refs = temp_refs["COV"]
    refs.update(temp_refs["NON_COV"])
    REFS["third_method"] = refs

def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, LBP["n_points"], LBP["radius"], LBP["method"])
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        score = kullback_leibler_divergence(hist, ref)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


if __name__ == "__main__":
    json_results = {}
    
    for i in range(10):
        temp_data = {}
        temp_data["training_set"] = load_random_training_set(TRAINING_IMGS_NUMBER, [NON_COVID_PATH, COVID_PATH])
        
        lbps_covid = [local_binary_pattern(image, LBP["n_points"], LBP["radius"], LBP["method"]) for image in TRAINING_COVID_IMGS]
        lbps_non_covid = [local_binary_pattern(image, LBP["n_points"], LBP["radius"], LBP["method"]) for image in TRAINING_NON_COVID_IMGS]
        lbps_collections = {
            "COV": lbps_covid,
            "NON_COV": lbps_non_covid
        }
        #TEST radius = 3 --> 0,1,2,9
        #TEST radius = 2 --> 3,4,5,10
        #TEST radius = 1 --> 6,7,8,11
        
        #TEST 0,3,6
        #temp_data["method_2_avg_difference_limit"] = 0.05
        #temp_data["method_3_avg_divergences_limit"] = 0.25
        
        #TEST 1,4,7
        #temp_data["method_2_avg_difference_limit"] = 0.01
        #temp_data["method_3_avg_divergences_limit"] = 0.05
        
        #TEST 2,5,8
        #temp_data["method_2_avg_difference_limit"] = 0.005
        #temp_data["method_3_avg_divergences_limit"] = 0.025

        #TEST 9,10,11
        temp_data["method_2_avg_difference_limit"] = 0.003
        temp_data["method_3_avg_divergences_limit"] = 0.01


        first_method_thread = Thread(target=method_1, args=(lbps_collections,))
        second_method_thread = Thread(target=method_2, args=(lbps_collections,temp_data["method_2_avg_difference_limit"]))
        third_method_thread = Thread(target=method_3, args=(lbps_collections,temp_data["method_3_avg_divergences_limit"]))

        for x in [first_method_thread, second_method_thread, third_method_thread]:
            x.start()
        
        for x in [first_method_thread, second_method_thread, third_method_thread]:
            x.join()

        temp_data["references"] = REFS
        used_files = temp_data["training_set"]["COV"] + temp_data["training_set"]["NON_COV"]
        temp_data["test_set"] = load_random_test_set(TEST_IMGS_NUMBER, [NON_COVID_PATH, COVID_PATH], used_files)

        temp_data["success_counter"] = {}
        all_refs_together = {}
        
        for key in temp_data["references"].keys():
            temp_data["success_counter"][key] = sum([ 1 for img in TEST_COVID_IMGS if str(match(temp_data["references"][key], img)).startswith("COV")])
            temp_data["success_counter"][key] += sum([ 1 for img in TEST_NON_COVID_IMGS if str(match(temp_data["references"][key], img)).startswith("NON")])
            all_refs_together.update(temp_data["references"][key])

        temp_data["success_counter"]["all_refs_together"] = sum([ 1 for img in TEST_COVID_IMGS if str(match(all_refs_together, img)).startswith("COV")])
        temp_data["success_counter"]["all_refs_together"] += sum([ 1 for img in TEST_NON_COVID_IMGS if str(match(all_refs_together, img)).startswith("NON")])

        pprint(temp_data["success_counter"])

        json_results["test_{}".format(i)] = temp_data

        #Free all common obj

        TRAINING_COVID_IMGS = []
        TRAINING_NON_COVID_IMGS = []
        TEST_COVID_IMGS = []
        TEST_NON_COVID_IMGS = []

        REFS = {
            "first_method": None,
            "second_method": None,
            "third_method": None,
        }
    counter = 0
    while os.path.exists("test_results_{}".format(counter) + ".json"):
        counter += 1
    with open("test_results_{}".format(counter) + ".json","w") as f:
        json.dump(json_results, f, indent=4, cls=NumpyEncoder)



    
    

        