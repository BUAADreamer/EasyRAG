import yaml


def get_yaml_data(yaml_file):
    # 打开yaml文件
    print("加载yaml文件:", yaml_file)
    with open(yaml_file, encoding="utf-8") as f:
        data = yaml.full_load(f.read())
    return data
