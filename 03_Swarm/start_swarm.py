import swarmlearning.swci as sw
import sys
import os
import random
import string
import re
import functions_swarm as f
import time
import helper as h
import datetime
import multiprocessing
import local_training as l


def do_task(name, task_path, taskrunnername, s):
    print(task_path)
    print(name)
    print(s.createTaskFrom(task_path))
    print(s.finalizeTask(name))
    print(s.resetTaskRunner(trName=taskrunnername))
    print(s.assignTask(name, peers=2, trName=taskrunnername))
    while not s.isTaskDone(trName=taskrunnername):
        print("Waiting for build task to complete")
        s.sleep(5)
    s.resetTaskRunner(trName=taskrunnername)


def start_swarm(swci_IP, csv_path):
    # define taskrunnername
    taskrunnername = "defaulttaskbb.taskdb.sml.hpe"
    # Build a socket
    s = sw.Swci(swci_IP, port=30306)
    # Create a context
    s.createContext("testContext", ip=swci_IP)
    # switch to context
    s.switchContext("testContext")
    # Create tasknames
    name1, name2 = h.generate_task_names(2)
    # print task names to control
    print("TASKNAME TO CONTROL", name1)
    print("TASKNAME TO CONTROL", name2)
    # Write code to csv
    h.write_to_csv([name2, "\n"], csv_path)
    # Update templates
    h.update_template(
        name1,
        "<VAR1>",
        "/home/ubuntu/data/swci_configs/user_env_tf_build_task.yaml",
        "/home/ubuntu/data/swci_configs/mod_user_build_task.yaml",
    )
    h.update_template(
        name1,
        "<VAR1>",
        "/home/ubuntu/data/swci_configs/swarm_mnist_task.yaml",
        "/home/ubuntu/data/swci_configs/mod_swarm_task.yaml",
    )
    h.update_template(
        name2,
        "<VAR2>",
        "/home/ubuntu/data/swci_configs/mod_swarm_task.yaml",
        "/home/ubuntu/data/swci_configs/mod_swarm_task.yaml",
    )
    # change to swarm directory
    s.cd("/platform/swarm/usr")
    # define paths from docker container for tasks
    envbuild_task_path = "taskdefs/mod_user_build_task.yaml"
    swarm_tast_path = "taskdefs/mod_swarm_task.yaml"
    # Create and assign envbuild tast
    do_task(name1, envbuild_task_path, taskrunnername, s)
    # Create and assign swarm task
    do_task(name2, swarm_tast_path, taskrunnername, s)
    print(s.getTrainingContractStatus(ctName="defaultbb.cqdb.sml.hpe"))
    # Wait for swarm task to finish
    s.sleep(15)
    s.resetTaskRunner(trName=taskrunnername)
    # status vom Contract
    s.resetTrainingContract(ctName="defaultbb.cqdb.sml.hpe")
    print("Process one start_swarm finished")
    return 0


def set_up_data(dataDir, targetDir, csv_path, fixed_test_set=[]):
    # creates and gives back the adress of training dir with data
    split_target_dir1, new_folder_name = f.define_split_target_dir(
        "/home/esralenz/Dokumente/13_Praktikum/08_all_data/hpe_praktikum/Diffusion_27/alltheotherstuff/host1/evaluations"
    )
    split_target_dir2, _ = f.define_split_target_dir(
        "/home/esralenz/Dokumente/13_Praktikum/08_all_data/hpe_praktikum/Diffusion_27/alltheotherstuff/host2/evaluations"
    )
    if len(fixed_test_set) != 0:
        f.split_test_group(
            sourceDirectory=dataDir,
            targetDirectory_h1=split_target_dir1,
            targetDirectory_h2=split_target_dir2,
            fixed_test_set=fixed_test_set,
        )
    else:
        f.split_test_group(
            sourceDirectory=dataDir,
            targetDirectory_h1=split_target_dir1,
            targetDirectory_h2=split_target_dir2,
        )
    (
        host1_H_dist,
        host2_H_dist,
        host1_MDD_dist,
        host2_MDD_dist,
    ) = f.calculate_distribution(fifty_fifty=False)
    used_MDD = host1_MDD_dist + host2_MDD_dist
    used_H = host1_H_dist + host2_H_dist
    # write metrics to csv
    h.write_to_csv(
        [
            host1_H_dist,
            host2_H_dist,
            used_H,
            host1_MDD_dist,
            host2_MDD_dist,
            used_MDD,
            new_folder_name,
        ],
        csv_path,
    )
    # define indexes for data distribution
    idx_H_h1, idx_H_h2, idx_MDD_h1, idx_MDD_h2 = f.define_trainingdata_indexes(
        sourceDirectory=dataDir,
        host1_H=host1_H_dist,
        host2_H=host2_H_dist,
        host1_MDD=host1_MDD_dist,
        host2_MDD=host2_MDD_dist,
    )
    # Copy files to host1 and host2 with defined distribution
    f.copy_selected_files(
        sourceDirectory=dataDir,
        targetDirectory=targetDir,
        host1_H=idx_H_h1,
        host2_H=idx_H_h2,
        host1_MDD=idx_MDD_h1,
        host2_MDD=idx_MDD_h2,
    )
    f.copy_selected_files_for_comparision(
        dataDir,
        split_target_dir1,
        host1_H=idx_H_h1,
        host2_H=idx_H_h2,
        host1_MDD=idx_MDD_h1,
        host2_MDD=idx_MDD_h2,
    )
    return split_target_dir1


def start_trainings(csv_path, controll_dir):
    p1 = multiprocessing.Process(
        target=start_swarm(swci_IP="10.1.23.25", csv_path=csv_path)
    )
    p2 = multiprocessing.Process(target=l.local_training(controll_dir=controll_dir))
    # Start processes
    p1.start()
    p2.start()
    # Wait for processes to finish
    p1.join()
    p2.join()
    return 0


if __name__ == "__main__":
    rounds = 10
    dataDir = "/home/ubuntu/data/all_data"
    targetDir = "/home/ubuntu/data"
    csv_path = "/home/ubuntu/data/host1/evaluations/metrics.csv"
    fixed_test_set = [[6, 7, 8, 9], [6, 7, 8, 9]]
    f.refill_alldata()
    for i in range(0, rounds):
        controll_dir = set_up_data(
            dataDir, targetDir, csv_path, fixed_test_set=fixed_test_set
        )
        print("Data distribution finished")
        start_trainings(csv_path, controll_dir)
        print("Training finished")
        # write to a file that the round is finished
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Round {i+1} finished at {timestamp}\n"
        with open("round_status.txt", "a") as file:
            file.write(message)
        f.refill_alldata()
    print("Finished")
