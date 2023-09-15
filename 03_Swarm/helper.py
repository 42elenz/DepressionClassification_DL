import os
import random
import string


def generate_task_names(number_of_names=1):
    list_of_names = []
    #reset random seed
    random.seed(None)
    for i in range(number_of_names):
        list_of_names.append(''.join(random.choice(string.ascii_letters) for i in range(10)))
    return list_of_names

def update_template(task_name, to_replace, template_path, new_template_path):
    with open(template_path) as org_build:
        newText=org_build.read().replace(to_replace,task_name)
    with open(new_template_path,"w") as mod_build:
        mod_build.write(newText)
    return new_template_path

def write_to_csv(data, filename):
    # Write data to csv file
        with open(filename, 'a') as f:
            for i in data:
                if i == '\n':
                    f.write(str(i))
                else:
                    f.write(str(i) + ',')