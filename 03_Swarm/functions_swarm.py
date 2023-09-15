import random
import os
import pandas as pd
import shutil
import string


def calculate_distribution(min_perc=0.2, fifty_fifty=False):
    # Define the percentages for distribution fo data
    if fifty_fifty:
        host1_H_dist = 0.5
        host2_H_dist = 0.5
        host1_MDD_dist = 0.5
        host2_MDD_dist = 0.5
        return host1_H_dist, host2_H_dist, host1_MDD_dist, host2_MDD_dist
    random.seed(None)
    host1_H_dist = round(random.uniform(min_perc, 1), 1)
    if host1_H_dist == 1:
        host1_H_dist = 0.9
    host2_H_dist = round(random.uniform(min_perc, 1 - host1_H_dist), 1)
    host1_MDD_dist = round(random.uniform(min_perc, 1), 1)
    if host1_MDD_dist == 1:
        host1_MDD_dist = 0.9
    host2_MDD_dist = round(random.uniform(min_perc, 1 - host1_MDD_dist), 1)
    if host1_H_dist + host2_H_dist > 1:
        if host1_H_dist > host2_H_dist:
            host1_H_dist = host1_H_dist - 0.1
        else:
            host2_H_dist = host2_H_dist - 0.1
    if host1_MDD_dist + host2_MDD_dist > 1:
        if host1_MDD_dist > host2_MDD_dist:
            host1_MDD_dist = host1_MDD_dist - 0.1
        else:
            host2_MDD_dist = host2_MDD_dist - 0.1
    return host1_H_dist, host2_H_dist, host1_MDD_dist, host2_MDD_dist


# Delte existing files in target directories
def remove_files_in_directory(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def refill_alldata():
    # copy data from back_up_data to all_data
    sourceDirectory = "/home/ubuntu/data/back_up_data"
    targetDirectory = "/home/ubuntu/data/all_data"

    # define target directories
    tarDir_H = os.path.join(targetDirectory, "H_EC/")
    tarDir_MDD = os.path.join(targetDirectory, "MDD_EC/")

    # define source directories
    sourceDir_H = os.path.join(sourceDirectory, "H_EC/")
    sourceDir_MDD = os.path.join(sourceDirectory, "MDD_EC/")

    # Delte existing files in target directories
    remove_files_in_directory(tarDir_H)
    remove_files_in_directory(tarDir_MDD)

    # Copy files from sources to targets
    for file_name in os.listdir(sourceDir_H):
        shutil.copy(os.path.join(sourceDir_H, file_name), tarDir_H)
    for file_name in os.listdir(sourceDir_MDD):
        shutil.copy(os.path.join(sourceDir_MDD, file_name), tarDir_MDD)


def define_split_target_dir(baseDir):
    # Get the number of existing folders in the baseDir
    existing_folders = [
        name
        for name in os.listdir(baseDir)
        if os.path.isdir(os.path.join(baseDir, name))
    ]
    num_folders = len(existing_folders)

    # Create a new folder with letters as name at the end
    new_num_folders = num_folders
    letter = string.ascii_lowercase[new_num_folders]
    # Create a new folder with a numerated name
    new_folder_name = f"training_{letter}"
    new_folder_path = os.path.join(baseDir, new_folder_name)
    os.makedirs(new_folder_path)

    return new_folder_path, new_folder_name


def split_test_group(
    sourceDirectory, targetDirectory_h1, targetDirectory_h2, fixed_test_set=[]
):
    directory_path_H = os.path.join(sourceDirectory, "H_EC")
    directory_path_MDD = os.path.join(sourceDirectory, "MDD_EC")

    # Define target directories for H and MDD
    tarDir_H1 = os.path.join(targetDirectory_h1, "H_EC/")
    tarDir_MDD1 = os.path.join(targetDirectory_h1, "MDD_EC/")

    # Define target directories for H and MDD
    tarDir_H2 = os.path.join(targetDirectory_h2, "H_EC/")
    tarDir_MDD2 = os.path.join(targetDirectory_h2, "MDD_EC/")

    # make target directories (existing ok)
    os.makedirs(tarDir_H1, exist_ok=True)
    os.makedirs(tarDir_MDD1, exist_ok=True)

    # make target directories (existing ok)
    os.makedirs(tarDir_H2, exist_ok=True)
    os.makedirs(tarDir_MDD2, exist_ok=True)

    if len(fixed_test_set) == 0:
        random.seed(None)
        # Define indices for test group for H and MDD as 20% of all existing files
        file_count_H = sum(len(files) for _, _, files in os.walk(directory_path_H))
        file_count_MDD = sum(len(files) for _, _, files in os.walk(directory_path_MDD))
        num_indices_H = int(0.2 * file_count_H)
        num_indices_MDD = int(0.2 * file_count_MDD)

        # Randomly select files from sourceDirectory and copy them to targetDirectory, use the name of the file as the name of the new file and delte them from sourceDirectory
        indices_H = random.sample(range(file_count_H), num_indices_H)
        indices_MDD = random.sample(range(file_count_MDD), num_indices_MDD)
    else:
        indices_H = fixed_test_set[0]
        indices_MDD = fixed_test_set[1]

    # sort indices in reverse order
    indices_H.sort(reverse=True)
    indices_MDD.sort(reverse=True)

    # Copy from all_data to test_group
    with open("number_of_files", "a") as file:
        file.write(os.path.join("Test_Files: "))
        file.write("H: " + str(len(indices_H)))
        file.write(" MDD: " + str(len(indices_MDD)) + "\n")
    for i in indices_H:
        shutil.copy(
            os.path.join(directory_path_H, os.listdir(directory_path_H)[i]), tarDir_H1
        )
        shutil.copy(
            os.path.join(directory_path_H, os.listdir(directory_path_H)[i]), tarDir_H2
        )
        with open("number_of_files", "a") as file:
            file.write((os.listdir(directory_path_H)[i]) + ",")
    for i in indices_MDD:
        shutil.copy(
            os.path.join(directory_path_MDD, os.listdir(directory_path_MDD)[i]),
            tarDir_MDD1,
        )
        shutil.copy(
            os.path.join(directory_path_MDD, os.listdir(directory_path_MDD)[i]),
            tarDir_MDD2,
        )
        with open("number_of_files", "a") as file:
            file.write((os.listdir(directory_path_MDD)[i]) + ",")
    for i in indices_H:
        os.remove(os.path.join(directory_path_H, os.listdir(directory_path_H)[i]))
    for i in indices_MDD:
        os.remove(os.path.join(directory_path_MDD, os.listdir(directory_path_MDD)[i]))


def define_trainingdata_indexes(
    sourceDirectory, host1_H=0.5, host2_H=0.5, host1_MDD=0.5, host2_MDD=0.5
):
    random.seed(None)
    # Directory paths
    directory_path_H = os.path.join(sourceDirectory, "H_EC")
    directory_path_MDD = os.path.join(sourceDirectory, "MDD_EC")
    # Get file counts
    file_count_H = sum(len(files) for _, _, files in os.walk(directory_path_H))
    file_count_MDD = sum(len(files) for _, _, files in os.walk(directory_path_MDD))
    # Calculate indices for Host 1 and Host 2
    total_indices_H = list(range(file_count_H))
    total_indices_MDD = list(range(file_count_MDD))

    num_indices_H_host1 = int(host1_H * file_count_H)
    num_indices_H_host2 = int(host2_H * file_count_H)
    num_indices_MDD_host1 = int(host1_MDD * file_count_MDD)
    num_indices_MDD_host2 = int(host2_MDD * file_count_MDD)
    random.shuffle(total_indices_H)
    random.shuffle(total_indices_MDD)
    indices_H_host1 = total_indices_H[:num_indices_H_host1]
    indices_H_host2 = total_indices_H[
        num_indices_H_host1 : num_indices_H_host1 + num_indices_H_host2
    ]
    indices_MDD_host1 = total_indices_MDD[:num_indices_MDD_host1]
    indices_MDD_host2 = total_indices_MDD[
        num_indices_MDD_host1 : num_indices_MDD_host1 + num_indices_MDD_host2
    ]
    return indices_H_host1, indices_H_host2, indices_MDD_host1, indices_MDD_host2


def copy_selected_files(
    sourceDirectory, targetDirectory, host1_H, host2_H, host1_MDD, host2_MDD
):
    # Loop through sourceDirectory and copy files to targetDirectory

    # Directory paths
    directory_path_H = os.path.join(sourceDirectory, "H_EC/")
    directory_path_MDD = os.path.join(sourceDirectory, "MDD_EC/")

    # Define target directories for Host 1 and Host 2
    tarDir_H1_H = os.path.join(targetDirectory, "host1/H_EC/")
    tarDir_H1_MDD = os.path.join(targetDirectory, "host1/MDD_EC/")
    tarDir_H2_H = os.path.join(targetDirectory, "host2/H_EC/")
    tarDir_H2_MDD = os.path.join(targetDirectory, "host2/MDD_EC/")

    remove_files_in_directory(tarDir_H1_H)
    remove_files_in_directory(tarDir_H1_MDD)
    remove_files_in_directory(tarDir_H2_H)
    remove_files_in_directory(tarDir_H2_MDD)

    # Use indices host1_H and so on to select files from sourceDirectory and copy them to targetDirectory, use the name of the file as the name of the new file
    for i in host1_H:
        shutil.copy(
            os.path.join(directory_path_H, os.listdir(directory_path_H)[i]), tarDir_H1_H
        )
    for i in host1_MDD:
        shutil.copy(
            os.path.join(directory_path_MDD, os.listdir(directory_path_MDD)[i]),
            tarDir_H1_MDD,
        )
    for i in host2_H:
        shutil.copy(
            os.path.join(directory_path_H, os.listdir(directory_path_H)[i]), tarDir_H2_H
        )
    for i in host2_MDD:
        shutil.copy(
            os.path.join(directory_path_MDD, os.listdir(directory_path_MDD)[i]),
            tarDir_H2_MDD,
        )
    with open("number_of_files", "a") as file:
        file.write(targetDirectory + " ")
        file.write("host1_H: " + str(len(host1_H)))
        file.write("host1_MDD: " + str(len(host1_MDD)))
        file.write("host2_H: " + str(len(host2_H)))
        file.write("host2_MDD: " + str(len(host2_MDD)) + "\n")


def copy_selected_files_for_comparision(
    sourceDir, refDir, host1_H, host2_H, host1_MDD, host2_MDD
):
    # Loop through sourceDirectory and copy files to targetDirectory
    # Directory paths
    directory_path_H = os.path.join(sourceDir, "H_EC/")
    directory_path_MDD = os.path.join(sourceDir, "MDD_EC/")

    # Define target directories for Host 1 and Host 2
    tarDir_H = os.path.join(refDir, "train_data/H_EC/")
    tarDir_MDD = os.path.join(refDir, "train_data/MDD_EC/")

    # make target directories (existing ok)
    os.makedirs(os.path.join(tarDir_H), exist_ok=True)
    os.makedirs(os.path.join(tarDir_MDD), exist_ok=True)

    # Use indices to put selected files in training data
    for i in host1_H:
        shutil.copy(
            os.path.join(directory_path_H, os.listdir(directory_path_H)[i]), tarDir_H
        )
    for i in host1_MDD:
        shutil.copy(
            os.path.join(directory_path_MDD, os.listdir(directory_path_MDD)[i]),
            tarDir_MDD,
        )
    for i in host2_H:
        shutil.copy(
            os.path.join(directory_path_H, os.listdir(directory_path_H)[i]), tarDir_H
        )
    for i in host2_MDD:
        shutil.copy(
            os.path.join(directory_path_MDD, os.listdir(directory_path_MDD)[i]),
            tarDir_MDD,
        )
