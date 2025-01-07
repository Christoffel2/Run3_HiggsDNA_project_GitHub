import argparse
import awkward as ak
import json
import os

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

def Sum_genWeight_calculator(sample_JSON):
    # Load the sample JSON file.
    with open(sample_JSON, "r") as f:
        s = json.load(f)

    # Determine the process name.
    filename = os.path.splitext(os.path.basename(sample_JSON))[0]
    process_name = filename.split("_sample")[0]

    # Loop through the root files in the sample JSON, process them with coffea.nanoevents.NanoEventFactory, get the sum of genWeight of each root file 
    # and save it to samples_sum_genWeight list.
    samples_sum_genWeight = []
    key = f"{process_name}_postEE"
    print(f"{key} has {len(s[key])} root files.")

    for i in range(0, len(s[key])):
        file = s[f"{process_name}_postEE"][i]
        out = NanoEventsFactory.from_root(
            file = file,
            schemaclass = NanoAODSchema,
        ).events()
        sum_genWeight = ak.sum(out.genWeight, axis = 0)
        print(sum_genWeight)
        samples_sum_genWeight.append(sum_genWeight)

    # Sum of genWeight of each root file in this sample JSON.
    print(samples_sum_genWeight)
    
    # Sum of genWeight of all root files in this sample JSON.
    total_sample_genWeight = ak.sum(ak.Array(samples_sum_genWeight), axis = 0)
    print(f"{process_name}_postEE's total sum of genWeight is {total_sample_genWeight}")

    return key, total_sample_genWeight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Please provide the path to the sample JSON file.")
    parser.add_argument("sample_JSON", type = str, help = "Path to the sample JSON file.")
    parser.add_argument("output_JSON", type = str, help = "Path to the JSON file that saves the total sum of genWeight of the sample JSON file.")

    args = parser.parse_args()

    # To append results from multiple runs to the same JSON file, you'll need to read the existing content of the file first, 
    # update it with the new result, and then write it back.

    # Load existing data from the output JSON file if it exists.
    if os.path.exists(args.output_JSON):
        with open(args.output_JSON, "r") as output_file:
            total_sum_genWeight = json.load(output_file)
    else:
        total_sum_genWeight = {}

    # Calculate the sum of genWeight for the current sample JSON.
    name, result = Sum_genWeight_calculator(args.sample_JSON)
    total_sum_genWeight[name] = result

    # Write the updated data back to the JSON file.
    with open(args.output_JSON, "w") as output_file:
        json.dump(total_sum_genWeight, output_file, indent = 4)