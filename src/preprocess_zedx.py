"""
本文件用于处理.zedx压缩包文档解压后的东西
"""
import os
from xml.etree import ElementTree
from bs4 import BeautifulSoup
from tqdm import tqdm

def dfs_tree(url2path: dict, node, parents: tuple):
    for child in node:
        sub_parents = parents + (child.get('name'), )
        url = child.get('url')
        url = url.replace('\\', '/')
        url2path[url] = sub_parents
        dfs_tree(url2path, child, sub_parents)

def load_content(meta_dir, url):
    load_path = os.path.join(meta_dir, 'documents', url)
    if os.path.exists(load_path):
        try:
            html_doc = open(load_path, 'r', encoding='utf-8').read()
        except:
            html_doc = open(load_path, 'r', encoding='gb2312').read()
        soup = BeautifulSoup(html_doc, 'lxml')
        texts = soup.find_all(string=True)
        all_text = '\n'.join(texts)
        return all_text
    else:
        print('文档不存在: ', load_path)
        return None

def format_content(content, path):
    new_content = []
    last_line = None
    for line in content.split('\n'):
        if last_line == line:
            continue
        last_line = line
        line = line.strip()
        if line.startswith('html'):
            continue
        if line:
            new_content.append(line)
    new_str = '###\n'
    if path:
        new_str += '文档路径: ' + '/'.join(path) + '\n\n'
    
    if new_content:
        new_str += '\n'.join(new_content) + '\n'
    else:
        print('空文档: ', path)
        new_str += '<文档为空>\n'
    return new_str
    
        
def fill_document(meta_dir, build_data_dir, url2path):
    for url, path in tqdm(url2path.items()):
        content = load_content(meta_dir, url)
        if content is None:
            continue
        if url.endswith('.html') or url.endswith('.htm'):
            url = url.replace('.html', '.txt').replace('.htm', '.txt')
            build_file = os.path.join(build_data_dir, url)
            build_dir = os.path.dirname(build_file)
            os.makedirs(build_dir, exist_ok=True)
            with open(build_file, 'w', encoding='utf-8') as fin:
                fin.write(format_content(content, path))
        else:
            print('未知的url后缀: ', url)

def process_package(package_name):
    meta_dir = os.path.join(base_origin_dir, package_name)
    build_data_dir = os.path.join(base_processed_data_dir, package_name)
    os.makedirs(build_data_dir, exist_ok=True)
    
    node_tree_path = os.path.join(meta_dir, 'nodetree.xml')
    
    element_tree = ElementTree.fromstring(open(node_tree_path, 'r', encoding='utf-8').read())
    
    url2path = {}
    dfs_tree(url2path, element_tree, (package_name, ))
    fill_document(meta_dir, build_data_dir, url2path)
    

if __name__ == '__main__':
    base_origin_dir = '/data/origin_data'
    base_processed_data_dir = '/home/zhangrichong/data/fengzc/rag/wzy_rag/RAG-AIOps/data/new_build_data'
    package_list = ['director', 'emsplus', 'rcp', 'umac']
    for package_name in package_list:
        process_package(package_name)


    
    